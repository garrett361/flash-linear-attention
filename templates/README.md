# FLA Boilerplate Templates

This directory contains comprehensive boilerplate templates for implementing new attention mechanisms in the FLA (Flash Linear Attention) library. These templates provide standardized structures and patterns to help you quickly implement and benchmark new attention mechanisms.

## Overview

The FLA library provides a modular architecture for implementing various attention mechanisms. The boilerplate templates cover:

1. **Layer Templates** - Standardized attention layer implementations
2. **Model Templates** - Complete model configurations and implementations
3. **Operations Templates** - Triton kernel implementations for efficient computation
4. **Benchmarking Templates** - Performance testing and comparison tools

## Quick Start

### 1. Implementing a New Attention Layer

To implement a new attention mechanism, follow these steps:

1. **Copy the layer template**:
   ```bash
   cp templates/layer_template.py fla/layers/your_attention.py
   ```

2. **Modify the template**:
   - Replace `TemplateAttention` with your attention mechanism name
   - Update the docstring with your mechanism's description and paper reference
   - Modify the `__init__` parameters to match your mechanism's requirements
   - Implement the `forward` method with your specific attention computation
   - Update the `state_size` method for caching

3. **Create corresponding operations**:
   ```bash
   cp templates/ops_template.py fla/ops/your_attention/
   ```

4. **Add to exports**:
   Update `fla/layers/__init__.py` and `fla/ops/__init__.py` to include your new components.

### 2. Creating a Complete Model

1. **Copy model templates**:
   ```bash
   cp templates/model_config_template.py fla/models/your_model/configuration_your_model.py
   cp templates/model_template.py fla/models/your_model/modeling_your_model.py
   ```

2. **Modify the templates**:
   - Update class names and configuration parameters
   - Modify the model architecture to match your attention mechanism
   - Update imports and references

3. **Add model exports**:
   Update `fla/models/__init__.py` to include your new model classes.

### 3. Setting Up Benchmarking

1. **Copy benchmark template**:
   ```bash
   cp templates/benchmark_template.py benchmarks/ops/benchmark_your_attention.py
   ```

2. **Modify the benchmark**:
   - Update imports to use your attention operations
   - Modify benchmark parameters and configurations
   - Add any specific inputs your mechanism requires

3. **Run benchmarks**:
   ```bash
   python benchmarks/ops/benchmark_your_attention.py
   ```

## Template Structure

### Layer Template (`layer_template.py`)

The layer template provides a standardized structure for attention layers:

- **Standardized initialization** with common parameters
- **Flexible architecture** supporting various attention mechanisms
- **Caching support** for efficient inference
- **Gradient checkpointing** compatibility
- **Comprehensive documentation** and type hints

Key features:
- Support for grouped query attention (GQA)
- Optional short convolution
- Rotary position embedding (RoPE)
- Input/output gating
- Normalization options

### Model Template (`model_template.py`)

The model template provides complete model implementations:

- **Configuration class** (`TemplateConfig`) for model parameters
- **Block implementation** (`TemplateBlock`) for transformer blocks
- **Model implementation** (`TemplateModel`) for the base model
- **Causal LM implementation** (`TemplateForCausalLM`) for language modeling

Key features:
- Hugging Face transformers compatibility
- Gradient checkpointing support
- Caching for efficient generation
- Standardized input/output handling

### Operations Template (`ops_template.py`)

The operations template provides Triton kernel implementations:

- **Forward kernels** for attention computation
- **Backward kernels** for gradient computation
- **Chunk-based computation** for efficient memory usage
- **Fused implementations** for optimal performance

Key features:
- Triton JIT compilation
- Block-based computation
- Memory-efficient implementations
- Gradient computation support

### Benchmark Template (`benchmark_template.py`)

The benchmark template provides performance testing tools:

- **Triton benchmarking** with performance reports
- **Memory usage testing** for memory efficiency
- **Accuracy testing** for numerical correctness
- **Comparison with other mechanisms** (FlashAttention, GLA, etc.)

Key features:
- Automated performance reporting
- Memory usage analysis
- Accuracy verification
- Comparative benchmarking

## Best Practices

### 1. Naming Conventions

- Use descriptive names for your attention mechanism
- Follow the existing naming patterns in the codebase
- Use consistent naming across layers, models, and operations

### 2. Documentation

- Provide comprehensive docstrings for all classes and methods
- Include paper references and mathematical descriptions
- Document all parameters and their effects
- Provide usage examples

### 3. Testing

- Implement comprehensive unit tests
- Test both forward and backward passes
- Verify numerical accuracy against reference implementations
- Test edge cases and error conditions

### 4. Performance

- Optimize Triton kernels for your specific use case
- Use appropriate block sizes for your hardware
- Implement fused operations where possible
- Profile and benchmark your implementation

### 5. Integration

- Follow the existing code patterns and conventions
- Ensure compatibility with the FLA ecosystem
- Support standard features like caching and gradient checkpointing
- Maintain backward compatibility when possible

## Examples

### Example 1: Simple Linear Attention

Here's a minimal example of implementing a simple linear attention mechanism:

```python
# fla/layers/simple_linear.py
from fla.layers.template import TemplateAttention

class SimpleLinearAttention(TemplateAttention):
    def forward(self, hidden_states, **kwargs):
        # Implement simple linear attention
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Simple linear attention computation
        o = torch.matmul(q, torch.matmul(k.transpose(-2, -1), v))
        
        return self.o_proj(o), None, None
```

### Example 2: Custom Triton Kernel

Here's an example of implementing a custom Triton kernel:

```python
# fla/ops/custom_attention/kernel.py
import triton
import triton.language as tl

@triton.jit
def custom_attention_kernel(q, k, v, o, seq_len, head_dim, BLOCK_SIZE: tl.constexpr):
    # Implement your custom attention kernel
    pass
```

### Example 3: Benchmarking Setup

Here's an example of setting up benchmarking:

```python
# benchmarks/ops/benchmark_custom_attention.py
import torch
import triton
from fla.ops.custom_attention import custom_attention

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['T'],
        x_vals=[128 * 2 ** i for i in range(0, 8)],
        line_arg='provider',
        line_vals=['custom', 'flash'],
        line_names=['custom_fwd', 'flash_fwd'],
        styles=[('green', '-'), ('blue', '-')],
        ylabel="Execution Time (ms)",
        plot_name="Custom_Attention_Performance",
        args={},
    )
)
def benchmark(T, provider):
    # Implement your benchmark
    pass
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Make sure to update all import statements when copying templates
2. **Shape Mismatches**: Verify tensor shapes match your attention mechanism's requirements
3. **Memory Issues**: Adjust block sizes and batch sizes for your hardware
4. **Performance Issues**: Profile your implementation and optimize bottlenecks

### Getting Help

- Check existing implementations in the FLA codebase for reference
- Review the documentation for similar attention mechanisms
- Test your implementation with small examples before scaling up
- Use the benchmarking tools to identify performance issues

## Contributing

When contributing new attention mechanisms:

1. Follow the boilerplate templates
2. Add comprehensive tests
3. Include benchmarking results
4. Update documentation
5. Ensure compatibility with existing code

## License

These templates are provided under the same license as the FLA library. See the main repository for license details.
