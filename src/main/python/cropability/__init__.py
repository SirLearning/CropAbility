"""
CropAbility — GPU compute (Python) + CPU/logic (Rust)
====================================================

- **Python**: `gpu`, `kernels`, GPU `genomics`, `viz`
- **Python** (`ngs`): thin facade over Rust for I/O and pipelines
- **Rust** (`cropability.native._core`): CPU I/O, NGS, TorchScript inference
"""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("cropability")
except PackageNotFoundError:
    __version__ = "0.1.0"

from cropability.gpu import DeviceManager, get_device_manager

__all__ = [
    "__version__",
    "DeviceManager",
    "get_device_manager",
]
