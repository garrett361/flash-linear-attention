#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Reference Training Script (Non-CP)
Usage:
    python train_cp_single.py
"""

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM
from torch.utils.data import Dataset
import time
from argparse import ArgumentParser
import numpy as np
import random


torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

class RealisticTextDataset(Dataset):
    """
    Realistic dataset that:
    1. Loads full sequences from storage
    2. Each worker/rank loads the SAME data (important for CP!)
    3. Returns full sequences (sharding happens in collate_fn)
    """
    
    def __init__(
        self,
        num_samples: int,
        seq_len: int,
        vocab_size: int,
        mode: str = 'random'
    ):
        """
        Args:
            num_samples: Number of training samples
            seq_len: Full sequence length (will be split across CP ranks)
            vocab_size: Vocabulary size
            mode: 'random' for synthetic data, 'file' for loading from disk
        """
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.mode = mode
        
        # For reproducibility: all ranks must see the same data!
        # In real scenarios, this would load from shared storage (NFS, S3, etc.)
        torch.manual_seed(42)
        
        print(f"    Dataset: {num_samples} samples, seq_len={seq_len}, vocab_size={vocab_size}")
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        """
        Load a full sequence.
        In real-world: this would load from disk, database, or cloud storage.
        All ranks load the SAME data.
        """
        # Use idx as seed for consistent data across ranks
        torch.manual_seed(42 + idx)
        
        # # Simulate loading from storage
        # input_ids = torch.randint(0, self.vocab_size, (self.seq_len,), dtype=torch.long)
        
        # # Create labels (shifted by 1 for next-token prediction)
        # labels = torch.cat([input_ids[1:], torch.tensor([self.vocab_size - 1])])

        seq = torch.arange(self.seq_len) % 100  # pattern repeats every 100 tokens
        input_ids = seq.clone()
        labels = (seq + 1) % 100  # next-token = next integer mod 100
        
        # Attention mask (all 1s for now, could have padding in real scenarios)
        attention_mask = torch.ones(self.seq_len, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'sample_id': idx  # For debugging
        }

def main():
    # ---- Config ----
    # num_samples = 1000
    # seq_len = 2048
    # batch_size = 4
    # num_steps = 100
    # vocab_size = 32000
    # learning_rate = 1e-4
    # checkpoint_dir = "./checkpoints_ref"

    parser = ArgumentParser(description="Reference Training Script (Non-CP)")


    # Data
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of training samples')
    parser.add_argument('--seq_len', type=int, default=2048, help='Full sequence length')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size (same on all ranks)')
    
    # Training
    parser.add_argument('--num_steps', type=int, default=100, help='Number of training steps')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help='Gradient clipping')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N steps')
    
    # Model
    parser.add_argument('--hidden_size', type=int, default=512, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=8, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')

    # Checkpoint
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints_ref', help='Checkpoint directory')
    parser.add_argument('--resume_from', type=str, default=None, help='Resume from checkpoint')
    
    args = parser.parse_args()
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Model ----
    config = GatedDeltaNetConfig(
        hidden_size=512,
        num_hidden_layers=8,
        num_heads=8,
        head_dim=64,
        vocab_size=args.vocab_size,
        attn_mode="chunk",
        use_gate=True,
        use_short_conv=False,
        expand_v=1.0,
    )
    model = GatedDeltaNetForCausalLM(config).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    # ---- Dataset & Loader ----
    dataset = RealisticTextDataset(args.num_samples, args.seq_len, args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # ---- Training ----
    print(f"\n🚀 Training Reference (Non-CP)")
    model.train()
    total_loss = 0.0

    torch.cuda.reset_peak_memory_stats()
    start_time = time.time()

    for step, batch in enumerate(dataloader):
        if step >= args.num_steps:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        step_time = time.time() - start_time
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_peak = torch.cuda.max_memory_allocated() / 1e9

        print(f"[Step {step+1}] Loss={loss.item():.4f} | "
            f"StepTime={step_time:.3f}s | "
            f"MemAlloc={mem_alloc:.2f}GB | "
            f"MemPeak={mem_peak:.2f}GB")

        start_time = time.time()  # reset timer

        optimizer.zero_grad()


        optimizer.zero_grad()

        total_loss += loss.item()

        if (step + 1) % 10 == 0:
            print(f"Step {step+1}/{args.num_steps} | Loss: {total_loss / 10:.4f}")
            total_loss = 0.0

    torch.save(model.state_dict(), f"{args.checkpoint_dir}/final_reference.pt")
    print("✅ Reference training complete and checkpoint saved.")

if __name__ == "__main__":
    main()