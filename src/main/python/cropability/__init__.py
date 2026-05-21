"""
CropAbility — high-performance GPU framework for plant genomics
================================================================

High-performance GPU computing toolkit for plant genomics, with multi-GPU
(H100/A2 PCIe) parallel support. Provides GPU-accelerated implementations of
core algorithms for sequence analysis, variant detection, and genomic statistics.

Main modules:
    - cropability.gpu      : GPU device management and resource scheduling
    - cropability.kernels  : Triton GPU kernels (sequence / stats / matrix)
    - cropability.genomics : plant genomics analysis algorithms
    - cropability.models   : exportable TorchScript models
    - cropability.io       : genomic data I/O (FASTA/VCF/BAM)
    - cropability.utils    : configuration, logging, and shared utilities
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cropability")
except PackageNotFoundError:
    __version__ = "0.1.0"

__author__ = "CropAbility Team"
__email__ = "cropability@example.com"
__license__ = "MIT"

# Top-level convenience imports
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
