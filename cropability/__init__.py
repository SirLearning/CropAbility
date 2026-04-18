"""
CropAbility — 植物基因组高性能GPU计算框架
============================================

面向植物基因组学的高性能GPU计算工具包，支持多GPU（H100/A2 PCIe）并行计算。
提供序列分析、变异检测、基因组统计等核心算法的GPU加速实现。

主要模块:
    - cropability.gpu      : GPU设备管理与资源调度
    - cropability.kernels  : Triton GPU内核 (序列处理 / 统计 / 矩阵)
    - cropability.genomics : 植物基因组分析算法
    - cropability.models   : 可导出的TorchScript模型
    - cropability.io       : 基因组数据读写 (FASTA/VCF/BAM)
    - cropability.utils    : 配置、日志与通用工具
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cropability")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "CropAbility Team"
__email__ = "cropability@example.com"
__license__ = "MIT"

# 顶层便捷导入
from cropability.gpu import DeviceManager, get_device_manager
from cropability.utils.config import Config, get_config
from cropability.utils.logging import get_logger

__all__ = [
    "__version__",
    "DeviceManager",
    "get_device_manager",
    "Config",
    "get_config",
    "get_logger",
]
