"""
分布式多 GPU 计算支持
=====================
封装 torch.distributed (DDP) 和 torch.multiprocessing 的启动逻辑，
用于在 2× H100 或 2× A2 上并行训练/推理。
"""

from __future__ import annotations

import os
import functools
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from cropability.utils.logging import get_logger

logger = get_logger(__name__)


def is_distributed() -> bool:
    """当前进程是否处于 torch.distributed 上下文中。"""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def all_reduce_mean(tensor: torch.Tensor) -> torch.Tensor:
    """跨进程求平均（常用于同步 loss）。"""
    if not is_distributed():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / get_world_size()


def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """从所有进程收集张量，返回列表（长度 = world_size）。"""
    if not is_distributed():
        return [tensor]
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


# ---------------------------------------------------------------------------
# DDP 训练循环启动器
# ---------------------------------------------------------------------------

def _ddp_worker(
    rank: int,
    world_size: int,
    fn: Callable,
    fn_kwargs: Dict[str, Any],
    backend: str,
    master_addr: str,
    master_port: int,
) -> None:
    """每个 DDP 进程的入口函数。"""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["LOCAL_RANK"] = str(rank)

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    logger.info(f"DDP worker rank={rank}/{world_size} started on cuda:{rank}")
    try:
        fn(rank=rank, world_size=world_size, **fn_kwargs)
    finally:
        dist.destroy_process_group()


def launch_ddp(
    fn: Callable,
    num_gpus: int = 2,
    backend: str = "nccl",
    master_addr: str = "localhost",
    master_port: int = 29500,
    **fn_kwargs,
) -> None:
    """
    用 mp.spawn 在本机 num_gpus 块 GPU 上启动 DDP 训练。

    Args:
        fn          : 工作函数，签名 fn(rank, world_size, **kwargs)
        num_gpus    : GPU 数量（world_size）
        backend     : 通信后端（nccl 推荐）
        master_addr : 主节点地址
        master_port : 主节点端口
        **fn_kwargs : 传给 fn 的额外关键字参数
    """
    available = torch.cuda.device_count()
    if available < num_gpus:
        logger.warning(
            f"请求 {num_gpus} 块 GPU，但系统只有 {available} 块，"
            f"自动降级到 {available} 块。"
        )
        num_gpus = available

    if num_gpus == 0:
        logger.warning("无可用 GPU，以单进程 CPU 模式运行。")
        fn(rank=0, world_size=1, **fn_kwargs)
        return

    logger.info(f"Launching DDP: {num_gpus} processes, backend={backend}")
    mp.spawn(
        _ddp_worker,
        args=(num_gpus, fn, fn_kwargs, backend, master_addr, master_port),
        nprocs=num_gpus,
        join=True,
    )


def wrap_ddp(
    model: "torch.nn.Module",
    device_ids: Optional[List[int]] = None,
    find_unused_parameters: bool = False,
) -> "torch.nn.parallel.DistributedDataParallel":
    """
    将模型包装为 DistributedDataParallel。
    应在 launch_ddp 回调内调用（dist 已初始化时）。
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    rank = get_rank()
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    ids = device_ids or [rank]
    return DDP(model, device_ids=ids, find_unused_parameters=find_unused_parameters)


# ---------------------------------------------------------------------------
# 装饰器：仅主进程执行
# ---------------------------------------------------------------------------

def main_process_only(fn: Callable) -> Callable:
    """装饰器：仅 rank-0 进程执行被装饰函数，其他进程等待同步后跳过。"""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if is_main_process():
            result = fn(*args, **kwargs)
        else:
            result = None
        barrier()
        return result
    return wrapper
