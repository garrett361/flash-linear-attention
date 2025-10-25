#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean Non-CP Training Script (WikiText-103)
============================================
Usage:
    python train_single_realtext.py \
        --num_steps 1500 \
        --seq_len 2048 \
        --batch_size 4 \
        --learning_rate 1e-4
"""

import os
import time
import random
import numpy as np
import math
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from argparse import ArgumentParser
from datasets import load_dataset
from transformers import AutoTokenizer

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM


# -----------------------
#   Reproducibility
# -----------------------
def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------
#   Dataset utilities
# -----------------------
def get_wikitext_dataset(seq_len=2048, max_samples=5000):
    """
    Load WikiText-103-raw, tokenize into fixed-length chunks.
    Returns tensor [N, seq_len] and tokenizer.
    
    FIX: Properly handle token buffer and ensure all chunks are 2D
    """
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"  Loading WikiText-103 dataset...")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    # Collect ALL tokens first
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

    # Concatenate all tokens into one long sequence
    full_tokens = torch.cat(all_tokens, dim=0)
    print(f"  Total tokens collected: {len(full_tokens)}")

    # Trim to multiple of seq_len and reshape
    n_chunks = len(full_tokens) // seq_len
    trimmed_tokens = full_tokens[:n_chunks * seq_len]
    token_tensor = trimmed_tokens.view(-1, seq_len)  # [N, seq_len]

    # Take first max_samples chunks
    token_tensor = token_tensor[:max_samples]
    print(f"  Final shape: {token_tensor.shape} (samples={len(token_tensor)}, seq_len={seq_len})")

    return token_tensor, tokenizer


class LMTextDataset(Dataset):
    """Dataset for next-token prediction"""
    def __init__(self, tokens, eos_id):
        self.tokens = tokens  # [N, seq_len]
        self.eos_id = eos_id

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, i):
        x = self.tokens[i].clone()  # [seq_len]
        y = x.clone()
        y[:-1] = x[1:]
        y[-1] = self.eos_id
        return {
            "input_ids": x,
            "attention_mask": torch.ones_like(x, dtype=torch.long),
            "labels": y
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
#   Training loop
# -----------------------
def main():
    parser = ArgumentParser(description="Non-CP LM Training (WikiText-103)")
    parser.add_argument("--seq_len", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=2000)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--vocab_size", type=int, default=50257)  # GPT-2 vocab
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_ref")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--save_every", type=int, default=500)
    parser.add_argument("--val_every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print("\n" + "="*80)
    print("NON-CP BASELINE TRAINING (WikiText-103)")
    print("="*80)

    # -----------------------
    #   Load dataset
    # -----------------------
    print("\n📚 Loading dataset...")
    toks, tokenizer = get_wikitext_dataset(seq_len=args.seq_len, max_samples=5000)
    dataset = LMTextDataset(toks, eos_id=tokenizer.eos_token_id)

    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )

    print(f"  Total samples: {len(dataset)}")
    print(f"  Train: {n_train} | Val: {n_val}")
    print(f"  Seq length: {args.seq_len}")
    print(f"  Batch size: {args.batch_size}")

    # -----------------------
    #   Model setup
    # -----------------------
    print("\n📐 Building model...")
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

    model = GatedDeltaNetForCausalLM(config).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")
    print(f"  Device: {device}")

    print(f"\n🚀 Training for {args.num_steps} steps")
    print("="*80 + "\n")

    # -----------------------
    #   Training state
    # -----------------------
    torch.cuda.reset_peak_memory_stats()
    model.train()
    step = 0
    total_loss = 0.0
    monitor = TrainingMonitor(window=10)

    start_time = time.time()

    while step < args.num_steps:
        for batch in train_loader:
            if step >= args.num_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss

            # Check loss sanity
            loss_warning = monitor.check_loss(loss, step)
            if loss_warning:
                print(loss_warning)

            # Backward pass
            loss.backward()

            # Check gradients
            total_norm, grad_warnings = monitor.check_gradients(model)

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += loss.item()
            step_time = time.time() - start_time

            # Logging
            if (step + 1) % 10 == 0:
                mem_alloc = torch.cuda.memory_allocated() / 1e9
                mem_peak = torch.cuda.max_memory_allocated() / 1e9
                avg_loss = total_loss / 10

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
                model.eval()
                val_loss = 0.0
                val_count = 0

                with torch.no_grad():
                    for vb in val_loader:
                        v_in = vb["input_ids"].to(device)
                        v_lbl = vb["labels"].to(device)
                        v_out = model(input_ids=v_in, labels=v_lbl)
                        val_loss += v_out.loss.item()
                        val_count += 1

                val_loss /= val_count
                ppl = math.exp(val_loss)
                print(f"  🧪 Validation Loss={val_loss:.4f} | Perplexity={ppl:.2f}")
                model.train()

            # Checkpoint
            if (step + 1) % args.save_every == 0:
                ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step+1}.pt")
                torch.save(
                    {
                        "step": step + 1,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config.to_dict()
                    },
                    ckpt_path
                )
                print(f"  💾 Checkpoint saved → {ckpt_path}")

            step += 1

    # Final checkpoint
    print("\n" + "="*80)
    print("✅ Training complete!")
    final_ckpt = os.path.join(args.checkpoint_dir, "final_model.pt")
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict()
        },
        final_ckpt
    )
    print(f"✅ Final checkpoint saved → {final_ckpt}")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()