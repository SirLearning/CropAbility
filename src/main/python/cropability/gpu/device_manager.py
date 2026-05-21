"""
GPU device manager
==================
Enumerates CUDA devices in the system (H100 PCIe, A2, etc.) and provides memory
monitoring, device selection policies, and resource lifecycle management.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cropability.utils.logging import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """Static and dynamic attributes of a single GPU device."""
    index: int
    name: str
    total_memory_gb: float
    compute_capability: Tuple[int, int]
    is_h100: bool = field(init=False)
    is_a2: bool = field(init=False)
    supports_bf16: bool = field(init=False)

    def __post_init__(self) -> None:
        name_lower = self.name.lower()
        self.is_h100 = "h100" in name_lower
        self.is_a2 = "a2" in name_lower and "a20" not in name_lower
        # BF16 supported on Ampere (sm_80+) and newer
        major, _ = self.compute_capability
        self.supports_bf16 = major >= 8

    @property
    def torch_device(self) -> "torch.device":  # type: ignore[name-defined]
        import torch
        return torch.device(f"cuda:{self.index}")

    def __repr__(self) -> str:
        cc = ".".join(map(str, self.compute_capability))
        return (
            f"DeviceInfo(index={self.index}, name={self.name!r}, "
            f"mem={self.total_memory_gb:.1f}GB, cc={cc}, "
            f"bf16={self.supports_bf16})"
        )


# -----------------------------------------------------------------------
# Device manager
# -----------------------------------------------------------------------

class DeviceManager:
    """
    GPU device manager (singleton).

    Features:
    - Enumerate and cache all available CUDA device info
    - Filter target devices by configured device_ids
    - "Pick freest GPU" selection policy
    - Manage multi-GPU contexts (DataParallel / DistributedDataParallel)
    """

    def __init__(
        self,
        device_ids: Optional[List[int]] = None,
        memory_fraction: float = 0.90,
        allow_growth: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._device_ids = device_ids  # None → use all available devices
        self._memory_fraction = memory_fraction
        self._allow_growth = allow_growth
        self._devices: List[DeviceInfo] = []
        self._initialized = False

        try:
            import torch
            self._torch_available = True
            self._cuda_available = torch.cuda.is_available()
        except ImportError:
            self._torch_available = False
            self._cuda_available = False

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self) -> "DeviceManager":
        """Enumerate devices and set memory limits; idempotent."""
        with self._lock:
            if self._initialized:
                return self
            self._discover_devices()
            self._configure_memory()
            self._initialized = True
        return self

    def _discover_devices(self) -> None:
        if not self._cuda_available:
            logger.warning("CUDA unavailable; running on CPU.")
            return

        import torch
        n = torch.cuda.device_count()
        logger.info(f"Found {n} CUDA device(s)")

        candidate_ids = self._device_ids if self._device_ids is not None else list(range(n))
        for idx in candidate_ids:
            if idx >= n:
                logger.warning(f"Device {idx} out of range ({n} total); skipped.")
                continue
            props = torch.cuda.get_device_properties(idx)
            info = DeviceInfo(
                index=idx,
                name=props.name,
                total_memory_gb=props.total_memory / 1024 ** 3,
                compute_capability=(props.major, props.minor),
            )
            self._devices.append(info)
            logger.info(f"  [cuda:{idx}] {info}")

    def _configure_memory(self) -> None:
        if not self._devices:
            return
        import torch
        for dev in self._devices:
            try:
                torch.cuda.set_per_process_memory_fraction(
                    self._memory_fraction, device=dev.index
                )
            except Exception as exc:
                logger.debug(f"set_per_process_memory_fraction failed on cuda:{dev.index}: {exc}")

    # ------------------------------------------------------------------
    # Device queries
    # ------------------------------------------------------------------

    @property
    def devices(self) -> List[DeviceInfo]:
        if not self._initialized:
            self.initialize()
        return list(self._devices)

    @property
    def num_gpus(self) -> int:
        return len(self.devices)

    @property
    def has_gpu(self) -> bool:
        return self.num_gpus > 0

    def get_device(self, idx: int = 0) -> "torch.device":
        """Return torch.device for the idx-th device in the managed list."""
        import torch
        if not self.has_gpu:
            return torch.device("cpu")
        managed_idx = self.devices[idx % self.num_gpus].index
        return torch.device(f"cuda:{managed_idx}")

    def get_primary_device(self) -> "torch.device":
        """Return primary GPU (prefer H100, then A2, otherwise the first device)."""
        import torch
        if not self.has_gpu:
            return torch.device("cpu")
        for dev in self.devices:
            if dev.is_h100:
                return dev.torch_device
        for dev in self.devices:
            if dev.is_a2:
                return dev.torch_device
        return self.devices[0].torch_device

    def get_freest_device(self) -> "torch.device":
        """Select the GPU with the most free memory."""
        import torch
        if not self.has_gpu:
            return torch.device("cpu")
        best_idx = self.devices[0].index
        best_free = 0
        for dev in self.devices:
            try:
                free, _ = torch.cuda.mem_get_info(dev.index)
                if free > best_free:
                    best_free = free
                    best_idx = dev.index
            except Exception:
                pass
        return torch.device(f"cuda:{best_idx}")

    # ------------------------------------------------------------------
    # Memory stats
    # ------------------------------------------------------------------

    def memory_stats(self) -> Dict[str, Dict[str, float]]:
        """Return per-GPU memory usage in GB."""
        import torch
        stats: Dict[str, Dict[str, float]] = {}
        for dev in self.devices:
            try:
                free, total = torch.cuda.mem_get_info(dev.index)
                used = total - free
                stats[f"cuda:{dev.index}"] = {
                    "total_gb": total / 1024 ** 3,
                    "used_gb": used / 1024 ** 3,
                    "free_gb": free / 1024 ** 3,
                    "utilization": used / total,
                }
            except Exception as exc:
                stats[f"cuda:{dev.index}"] = {"error": str(exc)}
        return stats

    def print_memory_stats(self) -> None:
        for name, s in self.memory_stats().items():
            if "error" in s:
                logger.warning(f"{name}: {s['error']}")
            else:
                logger.info(
                    f"{name}: {s['used_gb']:.2f}/{s['total_gb']:.2f} GB "
                    f"({s['utilization']*100:.1f}% used)"
                )

    # ------------------------------------------------------------------
    # Multi-GPU helpers
    # ------------------------------------------------------------------

    def wrap_data_parallel(self, module: "torch.nn.Module") -> "torch.nn.Module":
        """Wrap a model in DataParallel for single-node multi-GPU inference/training."""
        import torch
        import torch.nn as nn
        if self.num_gpus > 1:
            device_ids = [d.index for d in self.devices]
            logger.info(f"DataParallel on devices: {device_ids}")
            module = nn.DataParallel(module, device_ids=device_ids)
        return module.to(self.get_primary_device())

    def setup_distributed(
        self,
        rank: int,
        world_size: int,
        backend: str = "nccl",
        master_addr: str = "localhost",
        master_port: int = 29500,
    ) -> None:
        """Initialize torch.distributed process group for DDP multi-process training."""
        import torch.distributed as dist
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        logger.info(f"Distributed initialized: rank={rank}/{world_size}, backend={backend}")

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "DeviceManager":
        return self.initialize()

    def __exit__(self, *_) -> None:
        pass

    def __repr__(self) -> str:
        return (
            f"DeviceManager(gpus={self.num_gpus}, "
            f"devices={[d.name for d in self._devices]})"
        )


# -----------------------------------------------------------------------
# Global singleton
# -----------------------------------------------------------------------

_manager_lock = threading.Lock()
_global_manager: Optional[DeviceManager] = None


def get_device_manager(
    device_ids: Optional[List[int]] = None,
    memory_fraction: float = 0.90,
) -> DeviceManager:
    """Return the global DeviceManager singleton; auto-initializes on first call."""
    global _global_manager
    with _manager_lock:
        if _global_manager is None:
            _global_manager = DeviceManager(
                device_ids=device_ids,
                memory_fraction=memory_fraction,
            ).initialize()
    return _global_manager
