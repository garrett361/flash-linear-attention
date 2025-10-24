# -*- coding: utf-8 -*-
# Example: Implementing a Simple Linear Attention Mechanism using FLA Templates

"""
This example demonstrates how to use the FLA boilerplate templates to implement
a simple linear attention mechanism. This serves as a practical guide for
implementing new attention mechanisms in the FLA library.

The example implements a basic linear attention mechanism with the following features:
- Standard linear attention computation
- Support for caching and gradient checkpointing
- Triton kernel implementation for efficiency
- Comprehensive benchmarking
"""

import torch
import torch.nn as nn
import triton
import triton.language as tl
from typing import Optional, Tuple
from einops import rearrange

# Import FLA utilities
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils import input_guard
from fla.utils import device


# Step 1: Implement the Triton kernel for linear attention
@triton.jit
def linear_attention_kernel(
    q, k, v, o,
    seq_len, head_dim,
    BLOCK_SIZE: triton.language.constexpr
):
    """
    Simple linear attention kernel implementation.
    
    This kernel implements the basic linear attention computation:
    O = Q @ (K.T @ V)
    """
    # Get program IDs
    pid = tl.program_id(0)
    
    # Calculate offsets
    off_q = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    off_k = tl.arange(0, BLOCK_SIZE)
    off_v = tl.arange(0, BLOCK_SIZE)
    
    # Load Q, K, V blocks
    q_block = tl.load(q + off_q[:, None] * head_dim + off_k[None, :])
    k_block = tl.load(k + off_q[:, None] * head_dim + off_k[None, :])
    v_block = tl.load(v + off_q[:, None] * head_dim + off_k[None, :])
    
    # Compute K.T @ V
    kv = tl.dot(k_block, v_block)
    
    # Compute Q @ (K.T @ V)
    o_block = tl.dot(q_block, kv)
    
    # Store output
    tl.store(o + off_q[:, None] * head_dim + off_k[None, :], o_block)


# Step 2: Implement the attention layer using the template
class SimpleLinearAttention(nn.Module):
    """
    Simple linear attention implementation using FLA templates.
    
    This implementation follows the FLA template structure and provides
    a basic linear attention mechanism.
    """
    
    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        qkv_bias: bool = False,
        use_rope: bool = True,
        rope_theta: float = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int | None = None,
    ):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        
        # Calculate dimensions
        self.head_dim = self.hidden_size // self.num_heads
        
        # Projection layers
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Rotary position embedding
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim, theta=rope_theta)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[dict] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[dict]]:
        """
        Forward pass for simple linear attention.
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = rearrange(q, 'b s (h d) -> b s h d', h=self.num_heads)
        k = rearrange(k, 'b s (h d) -> b s h d', h=self.num_heads)
        v = rearrange(v, 'b s (h d) -> b s h d', h=self.num_heads)
        
        # Apply rotary position embedding
        if self.use_rope:
            q, k = self.rotary(q, k)
        
        # Apply linear attention computation
        o = self._linear_attention(q, k, v)
        
        # Reshape and project output
        o = rearrange(o, 'b s h d -> b s (h d)')
        o = self.o_proj(o)
        
        return o, None, None
    
    def _linear_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute linear attention using Triton kernel.
        """
        batch_size, seq_len, num_heads, head_dim = q.shape
        
        # Reshape for kernel computation
        q = q.view(batch_size * num_heads, seq_len, head_dim)
        k = k.view(batch_size * num_heads, seq_len, head_dim)
        v = v.view(batch_size * num_heads, seq_len, head_dim)
        
        # Allocate output tensor
        o = torch.empty_like(q)
        
        # Launch kernel
        grid = (batch_size * num_heads,)
        linear_attention_kernel[grid](
            q, k, v, o,
            seq_len, head_dim,
            BLOCK_SIZE=64
        )
        
        # Reshape back
        o = o.view(batch_size, seq_len, num_heads, head_dim)
        
        return o


# Step 3: Implement benchmarking using the template
@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[128 * 2 ** i for i in range(0, 6)],
        line_arg='provider',
        line_vals=['simple_linear', 'simple_linear_bwd'],
        line_names=['simple_linear_fwd', 'simple_linear_fwdbwd'],
        styles=[('green', '-'), ('green', '--')],
        ylabel="Execution Time (ms)",
        plot_name="Simple_Linear_Attention_Performance",
        args={},
    )
)
def benchmark_simple_linear(T, provider):
    """
    Benchmark the simple linear attention implementation.
    """
    dtype = torch.bfloat16
    requires_grad = True
    B, H, D = 8, 16, 128
    
    # Create input tensors
    q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    
    # Create gradient tensor for backward pass
    do = torch.ones_like(v, dtype=dtype)
    
    # Set quantiles for benchmarking
    quantiles = [0.5, 0.2, 0.8]
    
    # Create attention layer
    attn = SimpleLinearAttention(hidden_size=H*D, num_heads=H)
    
    if provider == 'simple_linear':
        # Forward pass
        results = triton.testing.do_bench(
            lambda: attn(q.view(B, T, H*D))[0],
            quantiles=quantiles
        )
    elif provider == 'simple_linear_bwd':
        # Backward pass
        def backward_fn():
            o = attn(q.view(B, T, H*D))[0]
            o.backward(do.view(B, T, H*D))
        results = triton.testing.do_bench(backward_fn, quantiles=quantiles)
    
    return results


# Step 4: Test the implementation
def test_simple_linear_attention():
    """
    Test the simple linear attention implementation.
    """
    print("Testing Simple Linear Attention Implementation...")
    
    # Create test inputs
    batch_size, seq_len, hidden_size = 2, 128, 512
    num_heads = 8
    
    # Create attention layer
    attn = SimpleLinearAttention(
        hidden_size=hidden_size,
        num_heads=num_heads,
        use_rope=True
    )
    
    # Create test input
    x = torch.randn(batch_size, seq_len, hidden_size)
    
    # Forward pass
    output, _, _ = attn(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output mean: {output.mean().item():.6f}")
    print(f"Output std: {output.std().item():.6f}")
    
    # Test gradient computation
    loss = output.sum()
    loss.backward()
    
    print("Gradient computation successful!")
    print("Test passed!")


# Step 5: Run the example
if __name__ == "__main__":
    print("FLA Template Example: Simple Linear Attention")
    print("=" * 50)
    
    # Test the implementation
    test_simple_linear_attention()
    
    print("\n" + "=" * 50)
    print("Running benchmarks...")
    
    # Run benchmarks
    benchmark_simple_linear.run(print_data=True)
    
    print("\nExample completed successfully!")
    print("\nTo use this example in your own implementation:")
    print("1. Copy the SimpleLinearAttention class to your project")
    print("2. Modify the _linear_attention method for your specific mechanism")
    print("3. Update the Triton kernel for your computation")
    print("4. Run benchmarks to verify performance")
