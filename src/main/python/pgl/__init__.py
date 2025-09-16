"""
PGL (Performance GPU Library) - 高性能GPU计算库

这个库提供了基于Triton的高性能GPU算子实现，
以及与Java/C++集成的TorchScript模型导出功能。
"""

# 导入主要功能
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