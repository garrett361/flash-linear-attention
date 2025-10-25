#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Context-Parallel Language Model Training (WikiText-103)
========================================================
Launch:
    torchrun --nproc_per_node=4 train_cp_realtext.py \
        --cp_size=4 \
        --seq_len=2048 \
        --batch_size=4 \
        --num_steps=1500 \
        --learning_rate=1e-4
"""

import os
import time
import random
import numpy as np
import math
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, random_split
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet_cp import GatedDeltaNetForCausalLMCP


# -----------------------
#   Reproducibility
# -----------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
#   Distributed setup (CP-only)
# -----------------------
def setup_distributed_cp(cp_size: int):
    """Setup distributed training with context parallelism."""
    dist.init_process_group(backend="nccl")

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size != cp_size:
        raise ValueError(
            f"For CP-only training, world_size ({world_size}) must equal cp_size ({cp_size}). "
            f"Launch with: torchrun --nproc_per_node={cp_size}"
        )

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Single CP group with all ranks
    cp_group = dist.new_group(list(range(world_size)))
    cp_rank = rank

    return rank, world_size, local_rank, device, cp_rank, cp_size, cp_group


# -----------------------
#   Dataset utilities (FIXED: line-by-line collection)
# -----------------------
def build_wikitext_chunks(tokenizer, seq_len=1024, split="train"):
    """
    Build WikiText chunks properly.
    Returns tensor [N, seq_len]
    """
    print(f"  Loading WikiText-103 ({split}) dataset...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    # Collect ALL tokens line by line
    all_tokens = []
    for line in ds["text"]:
        if not line.strip():
            continue
        ids = tokenizer(
            line,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False
        )["input_ids"].squeeze(0)
        all_tokens.append(ids)

    # Concatenate all into one long sequence
    full_tokens = torch.cat(all_tokens, dim=0)
    print(f"    Total tokens: {len(full_tokens)}")

    # Trim to multiple of seq_len and reshape
    n_chunks = len(full_tokens) // seq_len
    trimmed = full_tokens[:n_chunks * seq_len]
    token_tensor = trimmed.view(-1, seq_len)

    print(f"    Final shape: {token_tensor.shape}")
    return token_tensor


class LMTextDataset(Dataset):
    """Dataset for next-token prediction"""
    def __init__(self, tokens, eos_id):
        self.tokens = tokens  # [N, seq_len]
        self.eos_id = eos_id

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        x = self.tokens[i].clone()
        y = x.clone()
        y[:-1] = x[1:]
        y[-1] = self.eos_id
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x, dtype=torch.long),
            "labels": y
        }


# -----------------------
#   CP Collator (shards sequences across ranks)
# -----------------------
class CPCollator:
    """
    Receives full sequences, shards them by cp_rank into contiguous chunks.
    All ranks see the SAME data (important for CP).
    """
    def __init__(self, cp_rank: int, cp_size: int, pad_token_id: int = 0):
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])

        batch_size, seq_len = input_ids.shape

        # Pad to be divisible by cp_size if needed
        if seq_len % self.cp_size != 0:
            pad_len = self.cp_size - (seq_len % self.cp_size)
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
            seq_len = seq_len + pad_len

        # Shard: each rank gets its chunk
        chunk_size = seq_len // self.cp_size
        start_idx = self.cp_rank * chunk_size
        end_idx = start_idx + chunk_size

        return {
            "input_ids": input_ids[:, start_idx:end_idx],
            "attention_mask": attention_mask[:, start_idx:end_idx],
            "labels": labels[:, start_idx:end_idx],
            "full_seq_len": seq_len,
            "chunk_start": start_idx,
            "chunk_end": end_idx,
        }


# -----------------------
#   Training Monitor (sanity checks)
# -----------------------
class TrainingMonitor:
    """Monitor training for NaN, gradient health, loss divergence"""
    def __init__(self, window=10):
        self.window = window
        self.losses = []

    def check_loss(self, loss, step):
        """Check loss sanity"""
        loss_val = loss.item()
        self.losses.append(loss_val)

        if torch.isnan(loss) or torch.isinf(loss):
            return f"❌ NaN/Inf at step {step}: {loss_val}"

        if loss_val < 0:
            return f"❌ Negative loss at step {step}: {loss_val}"

        # Detect sudden spikes
        if len(self.losses) > 5:
            prev_avg = np.mean(self.losses[-6:-1])
            if loss_val > prev_avg * 100:
                return f"⚠️ Loss spike: {prev_avg:.4f} → {loss_val:.4f}"

        return None

    def check_gradients(self, model):
        """Check gradient health"""
        total_norm = 0.0
        has_grad = False

        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
                if p.grad.abs().sum() > 0:
                    has_grad = True

        total_norm = total_norm ** 0.5

        warnings = []
        if not has_grad:
            warnings.append("⚠️ No gradient flow detected!")
        if total_norm > 100:
            warnings.append(f"⚠️ Large grad norm: {total_norm:.2f}")

        return total_norm, warnings


# -----------------------
#   Loss computation with CP
# -----------------------
def compute_loss_with_cp(model, batch, cp_rank, cp_size, cp_group, device):
    """Forward pass with CP."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    if torch.isnan(input_ids.float()).any():
        print(f"❌ Rank {cp_rank}: NaN in input_ids")
        return torch.tensor(0.0, device=device, requires_grad=True)

    outputs = model(
        input_ids=input_ids,
        attention_mask=None,
        labels=labels,
        cp_rank=cp_rank,
        cp_size=cp_size,
        cp_group=cp_group,
    )

    loss = outputs.loss

    if torch.isnan(loss) or torch.isinf(loss):
        print(f"❌ Rank {cp_rank}: NaN/Inf loss detected")
        return torch.tensor(0.0, device=device, requires_grad=True)

    return loss


# -----------------------
#   Checkpoint helpers
# -----------------------
def save_checkpoint(model, optimizer, step, checkpoint_dir, rank):
    """Save checkpoint (rank 0 only)"""
    if rank != 0:
        return

    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"step_{step}.pt")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    print(f"  💾 Checkpoint saved → {path}")


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load checkpoint"""
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0

    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    step = ckpt["step"]

    print(f"  ✓ Loaded checkpoint from step {step}")
    return step


# -----------------------
#   Validation
# -----------------------
@torch.inference_mode()
def evaluate(model, val_loader, cp_rank, cp_size, cp_group, device):
    """Compute validation perplexity"""
    model.eval()
    total_loss = 0.0
    count = 0

    for batch in val_loader:
        loss = compute_loss_with_cp(model, batch, cp_rank, cp_size, cp_group, device)
        total_loss += loss.item()
        count += 1

    model.train()

    avg_loss = total_loss / max(1, count)
    ppl = math.exp(avg_loss)

    if cp_rank == 0:
        print(f"  🧪 Validation Loss={avg_loss:.4f} | Perplexity={ppl:.2f}")

    return ppl


# -----------------------
#   Main training
# -----------------------
def main():
    parser = ArgumentParser(description="Context-Parallel LM Training (WikiText-103)")

    # Distributed
    parser.add_argument("--cp_size", type=int, required=True, help="Context parallel size")

    # Data
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=2)

    # Training
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=200)

    # Model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)

    # Checkpoint
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_cp")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    rank, world_size, local_rank, device, cp_rank, cp_size, cp_group = setup_distributed_cp(args.cp_size)

    if rank == 0:
        print("\n" + "="*80)
        print("CONTEXT-PARALLEL LM TRAINING (WikiText-103)")
        print("="*80)
        print(f"\n🔧 Distributed Setup:")
        print(f"  World size: {world_size}")
        print(f"  CP size: {cp_size}")
        print(f"  CP rank: {cp_rank}")
        print(f"  Device: {device}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Model config
    config = GatedDeltaNetConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        vocab_size=tokenizer.vocab_size,
        attn_mode="chunk",
        use_gate=True,
        use_short_conv=True,
        expand_v=1.0,
    )

    if rank == 0:
        n_params = sum(p.numel() for p in GatedDeltaNetForCausalLMCP(config).parameters())
        print(f"\n📐 Model Configuration:")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_layers: {config.num_hidden_layers}")
        print(f"  num_heads: {config.num_heads}")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  Parameters: {n_params:,}")

    # Build model
    model = GatedDeltaNetForCausalLMCP(config).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    start_step = load_checkpoint(model, optimizer, args.resume_from, device) if args.resume_from else 0

    # Dataset
    if rank == 0:
        print(f"\n📚 Dataset:")
        print(f"  Seq length: {args.seq_len}")
        print(f"  Chunk per rank: {args.seq_len // cp_size}")
        print(f"  Batch size: {args.batch_size}")

    token_chunks_train = build_wikitext_chunks(tokenizer, seq_len=args.seq_len, split="train")
    token_chunks_val = build_wikitext_chunks(tokenizer, seq_len=args.seq_len, split="validation")

    train_dataset = LMTextDataset(token_chunks_train, eos_id=tokenizer.eos_token_id)
    val_dataset = LMTextDataset(token_chunks_val, eos_id=tokenizer.eos_token_id)

    # Dataloaders with CP collator
    collator = CPCollator(cp_rank=cp_rank, cp_size=cp_size, pad_token_id=tokenizer.pad_token_id)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,  # Keep aligned across ranks
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    if rank == 0:
        print(f"\n🚀 Training Configuration:")
        print(f"  Total steps: {args.num_steps}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Checkpoint every: {args.save_every} steps")
        print(f"  Validate every: {args.val_every} steps")
        print("\n" + "="*80 + "\n")

    # Training loop
    model.train()
    step = start_step
    total_loss = 0.0
    monitor = TrainingMonitor(window=10)

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break

            # Forward pass
            loss = compute_loss_with_cp(model, batch, cp_rank, cp_size, cp_group, device)

            # Check loss sanity
            loss_warning = monitor.check_loss(loss, step)
            if loss_warning and rank == 0:
                print(loss_warning)

            # Backward pass
            loss.backward()

            # Check gradients
            total_norm, grad_warnings = monitor.check_gradients(model)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            step_time = time.time() - start_time

            # Logging
            if (step + 1) % 10 == 0:
                torch.cuda.synchronize()
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                avg_loss = total_loss / 10

                if rank == 0:
                    print(
                        f"[Step {step+1:4d}] Loss={avg_loss:.4f} | "
                        f"GradNorm={total_norm:.4f} | "
                        f"Time={step_time:.3f}s | "
                        f"Mem={mem_alloc:.2f}GB | Peak={mem_peak:.2f}GB"
                    )
                    for w in grad_warnings:
                        print(f"  {w}")

                total_loss = 0.0
                start_time = time.time()

            # Validation
            if (step + 1) % args.val_every == 0:
                evaluate(model, val_loader, cp_rank, cp_size, cp_group, device)

            # Checkpoint
            if (step + 1) % args.save_every == 0:
                save_checkpoint(model, optimizer, step + 1, args.checkpoint_dir, rank)

            step += 1

    # Final checkpoint
    if rank == 0:
        print("\n" + "="*80)
        print("✅ Training complete!")
        save_checkpoint(model, optimizer, step, args.checkpoint_dir, rank)
        print("="*80 + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()