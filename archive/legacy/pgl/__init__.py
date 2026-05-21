"""
PGL (Performance GPU Library) - high-performance GPU computing library.

This library provides Triton-based high-performance GPU operator implementations
and TorchScript model export for Java/C++ integration.
"""

# Import main APIs
from .ops import (
    add,
    export_torchscript_model,
    benchmark_add_operations,
    validate_correctness
)

__version__ = "1.0.0"
__author__ = "CropAbility Team"

__all__ = [
    'add',
    'export_torchscript_model', 
    'benchmark_add_operations',
    'validate_correctness'
]
