"""GPU device management: discovery, memory monitoring, multi-GPU scheduling, and distributed support."""

from cropability.gpu.device_manager import DeviceManager, get_device_manager

__all__ = ["DeviceManager", "get_device_manager"]
