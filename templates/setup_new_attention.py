#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FLA Template Setup Script

This script helps you set up a new attention mechanism using the FLA boilerplate templates.
It automates the process of copying templates, updating names, and setting up the basic structure.

Usage:
    python setup_new_attention.py --name MyAttention --description "My custom attention mechanism"
"""

import argparse
import os
import re
import shutil
from pathlib import Path


def replace_in_file(file_path: Path, replacements: dict):
    """Replace placeholders in a file with actual values."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    for placeholder, value in replacements.items():
        content = content.replace(placeholder, value)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)


def create_directory_structure(base_path: Path, name: str):
    """Create the directory structure for the new attention mechanism."""
    # Create main directories
    layers_dir = base_path / "fla" / "layers"
    models_dir = base_path / "fla" / "models" / name.lower()
    ops_dir = base_path / "fla" / "ops" / name.lower()
    benchmarks_dir = base_path / "benchmarks" / "ops"
    
    # Create directories
    models_dir.mkdir(parents=True, exist_ok=True)
    ops_dir.mkdir(parents=True, exist_ok=True)
    
    return layers_dir, models_dir, ops_dir, benchmarks_dir


def setup_attention_layer(layers_dir: Path, name: str, description: str):
    """Set up the attention layer."""
    template_path = Path(__file__).parent / "layer_template.py"
    layer_path = layers_dir / f"{name.lower()}.py"
    
    # Copy template
    shutil.copy(template_path, layer_path)
    
    # Replace placeholders
    replacements = {
        "TemplateAttention": name,
        "template": name.lower(),
        "Template for implementing new attention layers": description,
        "from fla.ops.template import chunk_template": f"from fla.ops.{name.lower()} import chunk_{name.lower()}",
        "chunk_template": f"chunk_{name.lower()}",
    }
    
    replace_in_file(layer_path, replacements)
    
    print(f"✓ Created attention layer: {layer_path}")


def setup_model_config(models_dir: Path, name: str, description: str):
    """Set up the model configuration."""
    template_path = Path(__file__).parent / "model_config_template.py"
    config_path = models_dir / f"configuration_{name.lower()}.py"
    
    # Copy template
    shutil.copy(template_path, config_path)
    
    # Replace placeholders
    replacements = {
        "TemplateConfig": f"{name}Config",
        "template": name.lower(),
        "Template model": f"{name} model",
        "Template-specific parameters": f"{name}-specific parameters",
    }
    
    replace_in_file(config_path, replacements)
    
    print(f"✓ Created model configuration: {config_path}")


def setup_model_implementation(models_dir: Path, name: str, description: str):
    """Set up the model implementation."""
    template_path = Path(__file__).parent / "model_template.py"
    model_path = models_dir / f"modeling_{name.lower()}.py"
    
    # Copy template
    shutil.copy(template_path, model_path)
    
    # Replace placeholders
    replacements = {
        "TemplateConfig": f"{name}Config",
        "TemplateBlock": f"{name}Block",
        "TemplateModel": f"{name}Model",
        "TemplateForCausalLM": f"{name}ForCausalLM",
        "TemplateAttention": name,
        "template": name.lower(),
        "Template block implementation": f"{name} block implementation",
        "Template model implementation": f"{name} model implementation",
        "Template model for causal language modeling": f"{name} model for causal language modeling",
    }
    
    replace_in_file(model_path, replacements)
    
    print(f"✓ Created model implementation: {model_path}")


def setup_operations(ops_dir: Path, name: str, description: str):
    """Set up the operations."""
    template_path = Path(__file__).parent / "ops_template.py"
    ops_path = ops_dir / "chunk.py"
    
    # Copy template
    shutil.copy(template_path, ops_path)
    
    # Replace placeholders
    replacements = {
        "template": name.lower(),
        "Template": name,
        "Template forward kernel": f"{name} forward kernel",
        "Template backward kernel": f"{name} backward kernel",
        "Template chunk-based attention computation": f"{name} chunk-based attention computation",
        "Template chunk-based attention mechanism": f"{name} chunk-based attention mechanism",
        "Template recurrent attention computation": f"{name} recurrent attention computation",
        "Template recurrent attention mechanism": f"{name} recurrent attention mechanism",
        "chunk_template": f"chunk_{name.lower()}",
        "fused_chunk_template": f"fused_chunk_{name.lower()}",
        "fused_recurrent_template": f"fused_recurrent_{name.lower()}",
    }
    
    replace_in_file(ops_path, replacements)
    
    # Create __init__.py
    init_path = ops_dir / "__init__.py"
    with open(init_path, 'w') as f:
        f.write(f"""# -*- coding: utf-8 -*-

from .chunk import chunk_{name.lower()}, fused_chunk_{name.lower()}, fused_recurrent_{name.lower()}

__all__ = [
    'chunk_{name.lower()}',
    'fused_chunk_{name.lower()}',
    'fused_recurrent_{name.lower()}'
]
""")
    
    print(f"✓ Created operations: {ops_path}")
    print(f"✓ Created operations __init__.py: {init_path}")


def setup_benchmark(benchmarks_dir: Path, name: str, description: str):
    """Set up the benchmark."""
    template_path = Path(__file__).parent / "benchmark_template.py"
    benchmark_path = benchmarks_dir / f"benchmark_{name.lower()}.py"
    
    # Copy template
    shutil.copy(template_path, benchmark_path)
    
    # Replace placeholders
    replacements = {
        "template": name.lower(),
        "Template": name,
        "Template chunk-based attention computation": f"{name} chunk-based attention computation",
        "Template chunk-based attention mechanism": f"{name} chunk-based attention mechanism",
        "Template recurrent attention computation": f"{name} recurrent attention computation",
        "Template recurrent attention mechanism": f"{name} recurrent attention mechanism",
        "Template_Performance": f"{name}_Performance",
        "chunk_template": f"chunk_{name.lower()}",
        "from fla.ops.template import chunk_template": f"from fla.ops.{name.lower()} import chunk_{name.lower()}",
    }
    
    replace_in_file(benchmark_path, replacements)
    
    print(f"✓ Created benchmark: {benchmark_path}")


def update_exports(base_path: Path, name: str):
    """Update the __init__.py files to include the new components."""
    
    # Update layers __init__.py
    layers_init = base_path / "fla" / "layers" / "__init__.py"
    if layers_init.exists():
        with open(layers_init, 'r') as f:
            content = f.read()
        
        # Add import
        import_line = f"from .{name.lower()} import {name}"
        if import_line not in content:
            content = content.replace(
                "from .abc import ABCAttention",
                f"from .abc import ABCAttention\nfrom .{name.lower()} import {name}"
            )
        
        # Add to __all__
        if f"'{name}'" not in content:
            content = content.replace(
                "__all__ = [",
                f"__all__ = [\n    '{name}',"
            )
        
        with open(layers_init, 'w') as f:
            f.write(content)
        
        print(f"✓ Updated layers __init__.py")
    
    # Update ops __init__.py
    ops_init = base_path / "fla" / "ops" / "__init__.py"
    if ops_init.exists():
        with open(ops_init, 'r') as f:
            content = f.read()
        
        # Add import
        import_line = f"from .{name.lower()} import chunk_{name.lower()}, fused_chunk_{name.lower()}, fused_recurrent_{name.lower()}"
        if import_line not in content:
            content = content.replace(
                "from .abc import chunk_abc",
                f"from .abc import chunk_abc\n{import_line}"
            )
        
        # Add to __all__
        for func_name in [f"chunk_{name.lower()}", f"fused_chunk_{name.lower()}", f"fused_recurrent_{name.lower()}"]:
            if f"'{func_name}'" not in content:
                content = content.replace(
                    "__all__ = [",
                    f"__all__ = [\n    '{func_name}',"
                )
        
        with open(ops_init, 'w') as f:
            f.write(content)
        
        print(f"✓ Updated ops __init__.py")


def create_example_usage(name: str, description: str):
    """Create an example usage file."""
    example_content = f"""# -*- coding: utf-8 -*-
# Example usage of {name}

import torch
from fla.layers import {name}
from fla.ops.{name.lower()} import chunk_{name.lower()}

# Create the attention layer
attention = {name}(
    hidden_size=2048,
    num_heads=32,
    use_rope=True
)

# Create test input
batch_size, seq_len, hidden_size = 2, 128, 2048
x = torch.randn(batch_size, seq_len, hidden_size)

# Forward pass
output, _, _ = attention(x)
print(f"Output shape: {{output.shape}}")

# Test gradient computation
loss = output.sum()
loss.backward()
print("Gradient computation successful!")

print("Example completed successfully!")
"""
    
    example_path = Path(__file__).parent / f"example_{name.lower()}_usage.py"
    with open(example_path, 'w') as f:
        f.write(example_content)
    
    print(f"✓ Created example usage: {example_path}")


def main():
    """Main function to set up a new attention mechanism."""
    parser = argparse.ArgumentParser(description="Set up a new attention mechanism using FLA templates")
    parser.add_argument("--name", required=True, help="Name of the attention mechanism (e.g., MyAttention)")
    parser.add_argument("--description", required=True, help="Description of the attention mechanism")
    parser.add_argument("--base-path", default=".", help="Base path of the FLA project")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not re.match(r'^[A-Z][a-zA-Z0-9]*$', args.name):
        print("Error: Name must start with uppercase letter and contain only alphanumeric characters")
        return
    
    base_path = Path(args.base_path)
    if not (base_path / "fla").exists():
        print("Error: FLA project not found. Please run this script from the FLA project root.")
        return
    
    print(f"Setting up {args.name} attention mechanism...")
    print(f"Description: {args.description}")
    print(f"Base path: {base_path}")
    print()
    
    try:
        # Create directory structure
        layers_dir, models_dir, ops_dir, benchmarks_dir = create_directory_structure(base_path, args.name)
        
        # Set up components
        setup_attention_layer(layers_dir, args.name, args.description)
        setup_model_config(models_dir, args.name, args.description)
        setup_model_implementation(models_dir, args.name, args.description)
        setup_operations(ops_dir, args.name, args.description)
        setup_benchmark(benchmarks_dir, args.name, args.description)
        
        # Update exports
        update_exports(base_path, args.name)
        
        # Create example usage
        create_example_usage(args.name, args.description)
        
        print()
        print("=" * 60)
        print(f"Successfully set up {args.name} attention mechanism!")
        print("=" * 60)
        print()
        print("Next steps:")
        print(f"1. Implement your attention computation in fla/ops/{args.name.lower()}/chunk.py")
        print(f"2. Modify the attention layer in fla/layers/{args.name.lower()}.py")
        print(f"3. Update model configuration in fla/models/{args.name.lower()}/configuration_{args.name.lower()}.py")
        print(f"4. Test your implementation with the example: python templates/example_{args.name.lower()}_usage.py")
        print(f"5. Run benchmarks: python benchmarks/ops/benchmark_{args.name.lower()}.py")
        print()
        print("Happy coding!")
        
    except Exception as e:
        print(f"Error: {e}")
        return


if __name__ == "__main__":
    main()
