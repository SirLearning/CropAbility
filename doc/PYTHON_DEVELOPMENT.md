# Python development

> **Note:** The installable package is **`cropability`** (`src/main/python/cropability/`).
> Legacy **PGL** code lives under `archive/legacy/pgl/`. GPU work uses PyTorch directly; CPU/NGS uses `cropability.ngs` → Rust `_core`.
> Tests: **[TESTING.md](TESTING.md)** (`src/test/python/` only; no `scripts/` folder).

# PGL - Performance GPU Library (legacy reference)

PGL is a high-performance GPU computing library that provides GPU-accelerated operators based on Triton and supports exporting TorchScript models for use from Rust/C++.

## ✨ Features

- 🚀 **High performance**: GPU-accelerated operators built on Triton
- 🔧 **Easy integration**: Export TorchScript models for Rust/C++ consumption
- 📊 **Complete testing**: Built-in performance benchmarks and correctness validation
- 🎯 **Simple to use**: Concise Python API and command-line tools

## 📦 Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install PGL package (development mode)
pip install -e .
```

## 🚀 Quick Start

### Python API

```python
import torch
from pgl import add

# Create GPU tensors
x = torch.randn(1000, device='cuda')
y = torch.randn(1000, device='cuda')

# Use PGL add operator (auto-selects best implementation)
result = add(x, y)

print(f"Result shape: {result.shape}")
```

### Command-line usage

```bash
# Show system info
python pgl_main.py info

# Run correctness validation
python pgl_main.py test --correctness

# Run performance benchmark
python pgl_main.py test --benchmark

# Export TorchScript model
python pgl_main.py export --output model.pt

# Test exported model
python pgl_main.py test --model model.pt
```

## 📁 Project Structure

```
pgl/
├── __init__.py          # Main package entry
├── ops/                 # Operator implementations
│   ├── __init__.py     # Export all operators
│   ├── add.py          # Add operator implementation
│   ├── export.py       # Model export utilities
│   └── test.py         # Tests and benchmarks
├── pgl_main.py         # CLI entry point
├── requirements.txt    # Dependencies
└── setup.py           # Package configuration
```

## 🔧 Operator API

### add function

```python
add(x: torch.Tensor, y: torch.Tensor, use_triton: bool = True) -> torch.Tensor
```

Smart addition that automatically selects the best implementation:
- When `use_triton=True` and inputs are on a CUDA device, uses Triton acceleration
- Otherwise uses the native PyTorch implementation
- Automatically handles exceptions and falls back to PyTorch

### triton_add function

```python
triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
```

Addition implemented directly with a Triton kernel:
- Requires input tensors on a CUDA device
- Requires matching tensor shapes and dtypes
- Delivers best GPU performance

### pytorch_add function

```python
pytorch_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor
```

Native PyTorch addition for comparison and fallback.

## GPU models

Use PyTorch `nn.Module` directly in Python (training and inference). There is no TorchScript export path and no Rust-side model loading. For CPU/NGS, use `cropability.ngs` and [RUST_DEVELOPMENT.md](RUST_DEVELOPMENT.md).

## 📊 Performance Testing

### Run benchmarks

```python
from pgl.ops import benchmark_add_operations

results = benchmark_add_operations(
    sizes=[1000, 10000, 100000], 
    num_runs=10
)

# View results
for i, size in enumerate(results['sizes']):
    print(f"Size {size}: Triton {results['triton_times'][i]:.3f}ms, "
          f"PyTorch {results['pytorch_times'][i]:.3f}ms, "
          f"speedup {results['speedup_ratios'][i]:.2f}x")
```

### Validate correctness

```python
from pgl.ops import validate_correctness

# Validate operator correctness
success = validate_correctness([100, 1000, 10000])
print(f"Correctness check: {'passed' if success else 'failed'}")
```

## 🐛 Troubleshooting

### Common issues

1. **CUDA unavailable**
   - Verify CUDA installation and GPU drivers
   - Confirm PyTorch CUDA support: `torch.cuda.is_available()`

2. **Triton import errors**
   - Install Triton: `pip install triton>=2.0.0`
   - Confirm CUDA compatibility

3. **Performance below expectations**
   - Ensure data is on the GPU: `tensor.is_cuda`
   - Use sufficiently large tensors to fully utilize the GPU
   - Check GPU memory usage

### Debug mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Verbose debug output is now enabled
result = add(x, y)
```

## 📈 Performance Profiling

PGL automatically generates the Triton performance profile file `triton_profile.json`, which can be used to analyze GPU utilization.

## 🤝 Contributing

1. Fork the project
2. Create a feature branch: `git checkout -b feature/new-op`
3. Add test cases
4. Commit changes: `git commit -am 'Add new operator'`
5. Push to the branch: `git push origin feature/new-op`
6. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](../../../LICENSE) file for details.
