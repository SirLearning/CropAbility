"""
GPU 设备管理器
==============
负责枚举系统中的 CUDA 设备（H100 PCIe、A2 等），
提供内存监控、设备选择策略和资源生命周期管理。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from cropability.utils.logging import get_logger

logger = get_logger(__name__)

# -----------------------------------------------------------------------
# 数据类
# -----------------------------------------------------------------------

@dataclass
class DeviceInfo:
    """单块 GPU 设备的静态与动态属性。"""
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
        # BF16 在 Ampere (sm_80+) 及以上支持
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
# 设备管理器
# -----------------------------------------------------------------------

class DeviceManager:
    """
    GPU 设备管理器（单例）。

    功能：
    - 枚举并缓存所有可用 CUDA 设备信息
    - 按配置列表（device_ids）筛选目标设备
    - 提供"选最空闲 GPU"策略
    - 管理多 GPU 上下文（DataParallel / DistributedDataParallel）
    """

    def __init__(
        self,
        device_ids: Optional[List[int]] = None,
        memory_fraction: float = 0.90,
        allow_growth: bool = True,
    ) -> None:
        self._lock = threading.Lock()
        self._device_ids = device_ids  # None → 使用全部可用设备
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
    # 初始化
    # ------------------------------------------------------------------

    def initialize(self) -> "DeviceManager":
        """枚举设备并设置内存限制，幂等操作。"""
        with self._lock:
            if self._initialized:
                return self
            self._discover_devices()
            self._configure_memory()
            self._initialized = True
        return self

    def _discover_devices(self) -> None:
        if not self._cuda_available:
            logger.warning("CUDA 不可用，将使用 CPU 运行。")
            return

        import torch
        n = torch.cuda.device_count()
        logger.info(f"发现 {n} 块 CUDA 设备")

        candidate_ids = self._device_ids if self._device_ids is not None else list(range(n))
        for idx in candidate_ids:
            if idx >= n:
                logger.warning(f"设备 {idx} 超出范围（共 {n} 块），已跳过。")
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
                logger.debug(f"set_per_process_memory_fraction 在 cuda:{dev.index} 失败: {exc}")

    # ------------------------------------------------------------------
    # 设备查询
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
        """获取第 idx 号（在 managed 列表中）设备的 torch.device。"""
        import torch
        if not self.has_gpu:
            return torch.device("cpu")
        managed_idx = self.devices[idx % self.num_gpus].index
        return torch.device(f"cuda:{managed_idx}")

    def get_primary_device(self) -> "torch.device":
        """获取主 GPU（优先选 H100，其次 A2，否则第一块）。"""
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
        """选择当前空闲显存最多的 GPU。"""
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
    # 内存统计
    # ------------------------------------------------------------------

    def memory_stats(self) -> Dict[str, Dict[str, float]]:
        """返回每块 GPU 的显存使用情况（GB）。"""
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
    # 多 GPU 辅助
    # ------------------------------------------------------------------

    def wrap_data_parallel(self, module: "torch.nn.Module") -> "torch.nn.Module":
        """将模型包装为 DataParallel（用于单机多GPU推理/训练）。"""
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
        """初始化 torch.distributed 进程组（用于 DDP 多进程训练）。"""
        import torch.distributed as dist
        os.environ["MASTER_ADDR"] = master_addr
        os.environ["MASTER_PORT"] = str(master_port)
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        logger.info(f"Distributed initialized: rank={rank}/{world_size}, backend={backend}")

    # ------------------------------------------------------------------
    # 上下文管理器
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
# 全局单例
# -----------------------------------------------------------------------

_manager_lock = threading.Lock()
_global_manager: Optional[DeviceManager] = None


def get_device_manager(
    device_ids: Optional[List[int]] = None,
    memory_fraction: float = 0.90,
) -> DeviceManager:
    """获取全局 DeviceManager 单例，首次调用时自动初始化。"""
    global _global_manager
    with _manager_lock:
        if _global_manager is None:
            _global_manager = DeviceManager(
                device_ids=device_ids,
                memory_fraction=memory_fraction,
            ).initialize()
    return _global_manager
