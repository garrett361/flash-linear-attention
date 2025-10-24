#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Comprehensive Training Benchmark: CP vs Non-CP
Measures all critical metrics for comparing training performance

Usage:
    # Single GPU (non-CP)
    python benchmark_training.py --mode single --seq_len 8192 --batch_size 2
    
    # Multi-GPU with CP
    torchrun --nproc_per_node=4 benchmark_training.py --mode cp --cp_size 4 --seq_len 32768 --batch_size 2
    
    # Automated sweep
    ./benchmark_training_sweep.sh
"""

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset

from fla.models.gated_deltanet.configuration_gated_deltanet import GatedDeltaNetConfig
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetForCausalLM
from fla.models.gated_deltanet.modeling_gated_deltanet_cp import GatedDeltaNetForCausalLMCP


@dataclass
class TrainingMetrics:
    """Comprehensive training metrics"""
    # Configuration
    mode: str  # 'single' or 'cp'
    cp_size: int
    seq_len: int
    batch_size: int
    num_steps: int
    hidden_size: int
    num_layers: int
    
    # Memory metrics (per GPU)
    peak_memory_gb: float
    avg_memory_gb: float
    model_memory_gb: float
    
    # Performance metrics
    total_time_sec: float
    avg_step_time_sec: float
    tokens_per_sec: float
    samples_per_sec: float
    
    # Training metrics
    initial_loss: float
    final_loss: float
    avg_loss: float
    loss_std: float
    
    # Efficiency metrics (for CP)
    memory_per_gpu_ratio: Optional[float] = None  # vs single GPU
    throughput_ratio: Optional[float] = None  # vs single GPU
    speed_efficiency: Optional[float] = None  # actual speedup / ideal speedup
    
    # Resource utilization
    forward_time_sec: Optional[float] = None
    backward_time_sec: Optional[float] = None
    optimizer_time_sec: Optional[float] = None
    
    # Status
    success: bool = True
    error_msg: Optional[str] = None
    
    def to_dict(self):
        return asdict(self)


class SimpleDataset(Dataset):
    """Simple deterministic dataset for benchmarking"""
    
    def __init__(self, num_samples: int, seq_len: int, vocab_size: int):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        torch.manual_seed(42)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        torch.manual_seed(42 + idx)
        seq = torch.arange(self.seq_len) % 100
        input_ids = seq.clone()
        labels = (seq + 1) % 100
        
        return {
            'input_ids': input_ids,
            'attention_mask': torch.ones(self.seq_len, dtype=torch.long),
            'labels': labels,
        }


class CPCollator:
    """Collator that shards sequences for CP"""
    
    def __init__(self, cp_rank: int, cp_size: int, pad_token_id: int = 0):
        self.cp_rank = cp_rank
        self.cp_size = cp_size
        self.pad_token_id = pad_token_id
        
    def __call__(self, batch):
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        batch_size, seq_len = input_ids.shape
        
        # Pad if needed
        if seq_len % self.cp_size != 0:
            pad_len = self.cp_size - (seq_len % self.cp_size)
            input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=self.pad_token_id)
            attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)
            labels = torch.nn.functional.pad(labels, (0, pad_len), value=-100)
            seq_len = seq_len + pad_len
        
        # Shard
        chunk_size = seq_len // self.cp_size
        start_idx = self.cp_rank * chunk_size
        end_idx = start_idx + chunk_size
        
        return {
            'input_ids': input_ids[:, start_idx:end_idx],
            'attention_mask': attention_mask[:, start_idx:end_idx],
            'labels': labels[:, start_idx:end_idx],
        }


def setup_distributed_cp(cp_size: int):
    """Setup distributed for CP"""
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    
    assert world_size == cp_size, f"World size {world_size} != cp_size {cp_size}"
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    
    cp_group = dist.new_group(list(range(world_size)))
    
    return rank, local_rank, device, cp_group


def benchmark_single_gpu(args) -> TrainingMetrics:
    """Benchmark single GPU training (no CP)"""
    
    print("\n" + "="*80)
    print("BENCHMARKING: Single GPU (No CP)")
    print("="*80)
    
    device = torch.device("cuda:0")
    
    # Model config
    config = GatedDeltaNetConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        attn_mode="chunk",
        use_gate=True,
        use_short_conv=False,
        expand_v=1.0,
    )
    
    print(f"\n📐 Model: {sum(p.numel() for p in GatedDeltaNetForCausalLM(config).parameters()):,} parameters")
    print(f"📊 Config: seq_len={args.seq_len}, batch={args.batch_size}, steps={args.num_steps}")
    
    # Create model
    model = GatedDeltaNetForCausalLM(config).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Measure model memory
    torch.cuda.reset_peak_memory_stats()
    model_memory = torch.cuda.max_memory_allocated() / 1e9
    
    # Dataset
    dataset = SimpleDataset(args.num_samples, args.seq_len, args.vocab_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    # Training
    model.train()
    losses = []
    step_times = []
    memory_readings = []
    
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    torch.cuda.reset_peak_memory_stats()
    total_start = time.time()
    
    print("\n🚀 Starting training...")
    
    for step, batch in enumerate(dataloader):
        if step >= args.num_steps:
            break
        
        step_start = time.time()
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward
        fwd_start = time.time()
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        torch.cuda.synchronize()
        fwd_time = time.time() - fwd_start
        forward_times.append(fwd_time)
        
        # Backward
        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time = time.time() - bwd_start
        backward_times.append(bwd_time)
        
        # Optimizer
        opt_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        opt_time = time.time() - opt_start
        optimizer_times.append(opt_time)
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        losses.append(loss.item())
        memory_readings.append(torch.cuda.memory_allocated() / 1e9)
        
        if (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{args.num_steps} | Loss: {loss.item():.4f} | "
                  f"Time: {step_time:.3f}s | Mem: {memory_readings[-1]:.2f}GB")
    
    total_time = time.time() - total_start
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    # Calculate metrics
    total_tokens = args.num_steps * args.batch_size * args.seq_len
    tokens_per_sec = total_tokens / total_time
    samples_per_sec = (args.num_steps * args.batch_size) / total_time
    
    print("\n✓ Training complete!")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Throughput: {tokens_per_sec:.0f} tokens/s")
    print(f"  Peak memory: {peak_memory:.2f}GB")
    
    return TrainingMetrics(
        mode='single',
        cp_size=1,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        peak_memory_gb=peak_memory,
        avg_memory_gb=np.mean(memory_readings),
        model_memory_gb=model_memory,
        total_time_sec=total_time,
        avg_step_time_sec=np.mean(step_times),
        tokens_per_sec=tokens_per_sec,
        samples_per_sec=samples_per_sec,
        initial_loss=losses[0],
        final_loss=losses[-1],
        avg_loss=np.mean(losses),
        loss_std=np.std(losses),
        forward_time_sec=np.sum(forward_times),
        backward_time_sec=np.sum(backward_times),
        optimizer_time_sec=np.sum(optimizer_times),
        success=True
    )


def benchmark_cp(args) -> TrainingMetrics:
    """Benchmark multi-GPU training with CP"""
    
    rank, local_rank, device, cp_group = setup_distributed_cp(args.cp_size)
    
    if rank == 0:
        print("\n" + "="*80)
        print(f"BENCHMARKING: Context Parallelism (CP={args.cp_size})")
        print("="*80)
    
    # Model config
    config = GatedDeltaNetConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        vocab_size=args.vocab_size,
        attn_mode="chunk",
        use_gate=True,
        use_short_conv=False,
        expand_v=1.0,
    )
    
    if rank == 0:
        print(f"\n📐 Model: {sum(p.numel() for p in GatedDeltaNetForCausalLMCP(config).parameters()):,} parameters")
        print(f"📊 Config: seq_len={args.seq_len}, batch={args.batch_size}, steps={args.num_steps}, cp_size={args.cp_size}")
        print(f"   Chunk per GPU: {args.seq_len // args.cp_size}")
    
    # Create model
    model = GatedDeltaNetForCausalLMCP(config).to(device).to(torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    # Measure model memory
    torch.cuda.reset_peak_memory_stats()
    model_memory = torch.cuda.max_memory_allocated() / 1e9
    
    # Dataset with CP collator
    dataset = SimpleDataset(args.num_samples, args.seq_len, args.vocab_size)
    collator = CPCollator(cp_rank=rank, cp_size=args.cp_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collator)
    
    # Training
    model.train()
    losses = []
    step_times = []
    memory_readings = []
    
    forward_times = []
    backward_times = []
    optimizer_times = []
    
    torch.cuda.reset_peak_memory_stats()
    total_start = time.time()
    
    if rank == 0:
        print("\n🚀 Starting training...")
    
    for step, batch in enumerate(dataloader):
        if step >= args.num_steps:
            break
        
        step_start = time.time()
        
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        
        # Forward
        fwd_start = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=None,
            labels=labels,
            cp_rank=rank,
            cp_size=args.cp_size,
            cp_group=cp_group
        )
        loss = outputs.loss
        torch.cuda.synchronize()
        fwd_time = time.time() - fwd_start
        forward_times.append(fwd_time)
        
        # Backward
        bwd_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        bwd_time = time.time() - bwd_start
        backward_times.append(bwd_time)
        
        # Optimizer
        opt_start = time.time()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        opt_time = time.time() - opt_start
        optimizer_times.append(opt_time)
        
        step_time = time.time() - step_start
        step_times.append(step_time)
        
        losses.append(loss.item())
        memory_readings.append(torch.cuda.memory_allocated() / 1e9)
        
        if rank == 0 and (step + 1) % 10 == 0:
            print(f"  Step {step+1}/{args.num_steps} | Loss: {loss.item():.4f} | "
                  f"Time: {step_time:.3f}s | Mem: {memory_readings[-1]:.2f}GB")
    
    total_time = time.time() - total_start
    peak_memory = torch.cuda.max_memory_allocated() / 1e9
    
    # Calculate metrics
    total_tokens = args.num_steps * args.batch_size * args.seq_len
    tokens_per_sec = total_tokens / total_time
    samples_per_sec = (args.num_steps * args.batch_size) / total_time
    
    if rank == 0:
        print("\n✓ Training complete!")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  Throughput: {tokens_per_sec:.0f} tokens/s")
        print(f"  Peak memory per GPU: {peak_memory:.2f}GB")
    
    result = TrainingMetrics(
        mode='cp',
        cp_size=args.cp_size,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        peak_memory_gb=peak_memory,
        avg_memory_gb=np.mean(memory_readings),
        model_memory_gb=model_memory,
        total_time_sec=total_time,
        avg_step_time_sec=np.mean(step_times),
        tokens_per_sec=tokens_per_sec,
        samples_per_sec=samples_per_sec,
        initial_loss=losses[0],
        final_loss=losses[-1],
        avg_loss=np.mean(losses),
        loss_std=np.std(losses),
        forward_time_sec=np.sum(forward_times),
        backward_time_sec=np.sum(backward_times),
        optimizer_time_sec=np.sum(optimizer_times),
        success=True
    )
    
    dist.destroy_process_group()
    
    return result


def main():
    parser = argparse.ArgumentParser(description='Benchmark Training: CP vs Non-CP')
    
    # Mode
    parser.add_argument('--mode', type=str, required=True, choices=['single', 'cp'],
                        help='Training mode: single GPU or CP')
    parser.add_argument('--cp_size', type=int, default=1, help='CP size (for CP mode)')
    
    # Data
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--seq_len', type=int, default=2048, help='Sequence length')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of training steps')
    
    # Model
    parser.add_argument('--hidden_size', type=int, default=2048, help='Hidden size')
    parser.add_argument('--num_layers', type=int, default=16, help='Number of layers')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of heads')
    parser.add_argument('--head_dim', type=int, default=64, help='Head dimension')
    parser.add_argument('--vocab_size', type=int, default=32000, help='Vocabulary size')
    
    # Training
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    # Output
    parser.add_argument('--output', type=str, default='training_benchmark.json',
                        help='Output JSON file')
    
    args = parser.parse_args()
    
    # Run benchmark
    if args.mode == 'single':
        metrics = benchmark_single_gpu(args)
    else:
        metrics = benchmark_cp(args)
    
    # Save results (rank 0 only for CP)
    if args.mode == 'single' or (args.mode == 'cp' and int(os.environ.get('RANK', 0)) == 0):
        with open(args.output, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
        print(f"\n✓ Results saved to {args.output}")
        print("="*80 + "\n")


if __name__ == '__main__':
    main()

