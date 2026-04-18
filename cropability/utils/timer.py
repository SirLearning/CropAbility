"""
计时工具
========
提供上下文管理器 Timer 和装饰器 timeit，支持 CUDA 事件精确计时。
"""

from __future__ import annotations

import functools
import time
from typing import Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable)


class Timer:
    """高精度计时器，GPU 可用时使用 CUDA 事件。

    使用方式::

        with Timer("my_op") as t:
            do_something()
        print(t.elapsed_ms)
    """

    def __init__(self, name: str = "", use_cuda: bool = True) -> None:
        self.name = name
        self._use_cuda = use_cuda
        self.elapsed_ms: float = 0.0
        self._start: Optional[float] = None
        self._cuda_start = None
        self._cuda_end = None

    def __enter__(self) -> "Timer":
        self._cuda_start = None
        self._cuda_end = None
        try:
            if self._use_cuda:
                import torch
                if torch.cuda.is_available():
                    self._cuda_start = torch.cuda.Event(enable_timing=True)
                    self._cuda_end = torch.cuda.Event(enable_timing=True)
                    self._cuda_start.record()  # type: ignore[union-attr]
                    return self
        except ImportError:
            pass
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        if self._cuda_start is not None:
            import torch
            self._cuda_end.record()  # type: ignore[union-attr]
            torch.cuda.synchronize()
            self.elapsed_ms = self._cuda_start.elapsed_time(self._cuda_end)  # type: ignore[union-attr]
        else:
            assert self._start is not None
            self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0

    @property
    def elapsed_s(self) -> float:
        return self.elapsed_ms / 1000.0

    def __repr__(self) -> str:
        return f"Timer(name={self.name!r}, elapsed={self.elapsed_ms:.3f} ms)"


def timeit(func: Optional[F] = None, *, use_cuda: bool = True) -> F:
    """装饰器：打印函数运行时间。

    ::

        @timeit
        def my_fn(): ...

        @timeit(use_cuda=False)
        def my_fn(): ...
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            with Timer(fn.__qualname__, use_cuda=use_cuda) as t:
                result = fn(*args, **kwargs)
            print(f"[timeit] {fn.__qualname__}: {t.elapsed_ms:.3f} ms")
            return result
        return wrapper  # type: ignore[return-value]

    if func is not None:
        return decorator(func)  # type: ignore[arg-type]
    return decorator  # type: ignore[return-value]
