"""GPU 设备管理层：设备发现、内存监控、多GPU调度与分布式支持。"""

from cropability.gpu.device_manager import DeviceManager, get_device_manager, DeviceInfo
from cropability.gpu.distributed import (
    launch_ddp,
    wrap_ddp,
    is_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    all_reduce_mean,
    main_process_only,
)

__all__ = [
    "DeviceManager",
    "get_device_manager",
    "DeviceInfo",
    "launch_ddp",
    "wrap_ddp",
    "is_distributed",
    "get_rank",
    "get_world_size",
    "is_main_process",
    "all_reduce_mean",
    "main_process_only",
]
