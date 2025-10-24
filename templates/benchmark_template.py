# -*- coding: utf-8 -*-
# Template for benchmarking new attention mechanisms

import torch
import triton
from torch.nn import functional as F

# Import your attention operation here
from fla.ops.template import chunk_template  # Replace with actual operation

# Optional: Import FlashAttention for comparison
try:
    from flash_attn import flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

# Optional: Import other attention mechanisms for comparison
try:
    from fla.ops.gla import chunk_gla
    from fla.ops.retention import chunk_retention
    HAS_COMPARISON_OPS = True
except ImportError:
    HAS_COMPARISON_OPS = False


@triton.testing.perf_report(
    triton.testing.Benchmark(
        # argument names to use as an x-axis for the plot
        x_names=['T'],
        # different possible values for `x_name`
        x_vals=[128 * 2 ** i for i in range(0, 8)],  # Adjust range as needed
        # argument name whose value corresponds to a different line in the plot
        line_arg='provider',
        # possible values for `line_arg`
        line_vals=['template', 'template_bwd'] + (
            ['flash', 'flash_bwd'] if HAS_FLASH else []
        ) + (
            ['gla', 'gla_bwd', 'retention', 'retention_bwd'] if HAS_COMPARISON_OPS else []
        ),
        # label name for the lines
        line_names=['template_fwd', 'template_fwdbwd'] + (
            ['flash_fwd', 'flash_fwdbwd'] if HAS_FLASH else []
        ) + (
            ['gla_fwd', 'gla_fwdbwd', 'retention_fwd', 'retention_fwdbwd'] if HAS_COMPARISON_OPS else []
        ),
        # line styles
        styles=[('green', '-'), ('green', '--')] + (
            [('blue', '-'), ('blue', '--')] if HAS_FLASH else []
        ) + (
            [('red', '-'), ('red', '--'), ('orange', '-'), ('orange', '--')] if HAS_COMPARISON_OPS else []
        ),
        ylabel="Execution Time (ms)",  # label name for the y-axis
        # name for the plot. Used also as a file name for saving the plot.
        plot_name="Template_Performance",  # Replace with your mechanism name
        args={},
    )
)
def benchmark(T, provider):
    """
    Benchmark function for attention mechanisms.
    
    This template provides a standardized benchmarking approach for new attention mechanisms.
    Modify the parameters and operations to match your specific implementation.
    
    Args:
        T: Sequence length
        provider: Which implementation to benchmark
        
    Returns:
        Benchmark results
    """
    from fla.utils import device
    
    # Configuration parameters - adjust these for your mechanism
    dtype = torch.bfloat16
    requires_grad = True
    B, H, D = 8, 16, 128  # Batch size, num heads, head dimension
    
    # Additional parameters specific to your mechanism
    # Add any extra parameters your mechanism needs
    M = 64  # Example: number of slots for ABC, etc.
    
    # Create input tensors
    q = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
    
    # Create additional inputs specific to your mechanism
    # Examples for different mechanisms:
    if provider.startswith('gla'):
        # GLA needs gating values
        g = F.logsigmoid(torch.randn(B, T, H, D, device=device, dtype=dtype))
        g = g.clamp_min(-5).requires_grad_(requires_grad)
    elif provider.startswith('abc'):
        # ABC needs slot values
        s = torch.randn(B, T, H, M, device=device, requires_grad=requires_grad, dtype=dtype)
    elif provider.startswith('template'):
        # Add any specific inputs your mechanism needs
        # Example: additional_gate = torch.randn(B, T, H, D, device=device, requires_grad=requires_grad, dtype=dtype)
        pass
    
    # Create gradient tensor for backward pass
    do = torch.ones_like(v, dtype=dtype)
    
    # Set quantiles for benchmarking
    quantiles = [0.5, 0.2, 0.8]
    
    # Benchmark different implementations
    if provider == 'template':
        # Forward pass of your mechanism
        results = triton.testing.do_bench(
            lambda: chunk_template(q, k, v),  # Replace with your operation
            quantiles=quantiles
        )
    elif provider == 'template_bwd':
        # Backward pass of your mechanism
        results = triton.testing.do_bench(
            lambda: chunk_template(q, k, v)[0].backward(do),  # Replace with your operation
            quantiles=quantiles
        )
    elif provider == 'flash' and HAS_FLASH:
        # FlashAttention forward pass
        results = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, causal=True),
            quantiles=quantiles
        )
    elif provider == 'flash_bwd' and HAS_FLASH:
        # FlashAttention backward pass
        results = triton.testing.do_bench(
            lambda: flash_attn_func(q, k, v, causal=True).backward(do),
            quantiles=quantiles
        )
    elif provider == 'gla' and HAS_COMPARISON_OPS:
        # GLA forward pass
        results = triton.testing.do_bench(
            lambda: chunk_gla(q, k, v, g),
            quantiles=quantiles
        )
    elif provider == 'gla_bwd' and HAS_COMPARISON_OPS:
        # GLA backward pass
        results = triton.testing.do_bench(
            lambda: chunk_gla(q, k, v, g)[0].backward(do),
            quantiles=quantiles
        )
    elif provider == 'retention' and HAS_COMPARISON_OPS:
        # Retention forward pass
        results = triton.testing.do_bench(
            lambda: chunk_retention(q, k, v),
            quantiles=quantiles
        )
    elif provider == 'retention_bwd' and HAS_COMPARISON_OPS:
        # Retention backward pass
        results = triton.testing.do_bench(
            lambda: chunk_retention(q, k, v)[0].backward(do),
            quantiles=quantiles
        )
    else:
        raise ValueError(f"Unknown provider: {provider}")
    
    return results


def benchmark_memory_usage():
    """
    Benchmark memory usage of your attention mechanism.
    
    This function helps you understand the memory requirements of your implementation.
    """
    from fla.utils import device
    
    dtype = torch.bfloat16
    B, H, D, T = 8, 16, 128, 2048  # Adjust parameters as needed
    
    # Create inputs
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    
    # Clear cache and reset memory stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    # Run your operation
    with torch.no_grad():
        output = chunk_template(q, k, v)  # Replace with your operation
    
    torch.cuda.synchronize()
    
    # Get memory usage
    memory_used = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
    print(f"Memory usage: {memory_used:.2f} GB")
    
    return memory_used


def benchmark_accuracy():
    """
    Benchmark numerical accuracy of your attention mechanism.
    
    Compare your implementation against a reference implementation to ensure correctness.
    """
    from fla.utils import device
    
    dtype = torch.float32  # Use float32 for accuracy testing
    B, H, D, T = 2, 4, 64, 128  # Smaller sizes for accuracy testing
    
    # Create inputs
    q = torch.randn(B, T, H, D, device=device, dtype=dtype)
    k = torch.randn(B, T, H, D, device=device, dtype=dtype)
    v = torch.randn(B, T, H, D, device=device, dtype=dtype)
    
    # Run your implementation
    output_yours = chunk_template(q, k, v)  # Replace with your operation
    
    # Run reference implementation (if available)
    # Example: output_ref = reference_implementation(q, k, v)
    
    # Calculate differences
    # max_diff = torch.max(torch.abs(output_yours - output_ref)).item()
    # mean_diff = torch.mean(torch.abs(output_yours - output_ref)).item()
    
    # print(f"Max difference: {max_diff:.6f}")
    # print(f"Mean difference: {mean_diff:.6f}")
    
    return output_yours


if __name__ == '__main__':
    # Run the main benchmark
    benchmark.run(print_data=True)
    
    # Run additional benchmarks
    print("\n=== Memory Usage Benchmark ===")
    benchmark_memory_usage()
    
    print("\n=== Accuracy Benchmark ===")
    benchmark_accuracy()
