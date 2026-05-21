"""
Distributed multi-GPU support
=============================
Wraps torch.distributed (DDP) and torch.multiprocessing launch logic for parallel
training/inference on 2× H100 or 2× A2.
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
    """Return whether the current process is in a torch.distributed context."""
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
    """Average a tensor across processes (commonly used to sync loss)."""
    if not is_distributed():
        return tensor
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / get_world_size()


def all_gather(tensor: torch.Tensor) -> List[torch.Tensor]:
    """Gather tensors from all processes; returns a list of length world_size."""
    if not is_distributed():
        return [tensor]
    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return gathered


# ---------------------------------------------------------------------------
# DDP training loop launcher
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
    """Entry point for each DDP process."""
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
    Launch DDP training on num_gpus local GPUs via mp.spawn.

    Args:
        fn          : Worker function with signature fn(rank, world_size, **kwargs)
        num_gpus    : Number of GPUs (world_size)
        backend     : Communication backend (nccl recommended)
        master_addr : Master node address
        master_port : Master node port
        **fn_kwargs : Extra keyword arguments passed to fn
    """
    available = torch.cuda.device_count()
    if available < num_gpus:
        logger.warning(
            f"Requested {num_gpus} GPU(s), but only {available} available; "
            f"falling back to {available} GPU(s)."
        )
        num_gpus = available

    if num_gpus == 0:
        logger.warning("No GPU available; running in single-process CPU mode.")
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
    Wrap a model in DistributedDataParallel.
    Call inside a launch_ddp callback after dist is initialized.
    """
    from torch.nn.parallel import DistributedDataParallel as DDP
    rank = get_rank()
    device = torch.device(f"cuda:{rank}")
    model = model.to(device)
    ids = device_ids or [rank]
    return DDP(model, device_ids=ids, find_unused_parameters=find_unused_parameters)


# ---------------------------------------------------------------------------
# Decorator: main process only
# ---------------------------------------------------------------------------

def main_process_only(fn: Callable) -> Callable:
    """Decorator: run the wrapped function only on rank-0; other ranks skip after sync."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if is_main_process():
            result = fn(*args, **kwargs)
        else:
            result = None
        barrier()
        return result
    return wrapper
