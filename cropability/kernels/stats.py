"""
统计计算 GPU 内核
=================
Welford 在线均值/方差、z-score 标准化、Pearson 相关系数。
支持 Triton GPU 路径和 PyTorch CPU/GPU fallback。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from cropability.utils.logging import get_logger

logger = get_logger(__name__)

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


# ---------------------------------------------------------------------------
# Triton 内核：逐行 Welford 均值与方差
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _welford_row_kernel(
        x_ptr,          # [M, N] float32 输入
        mean_ptr,       # [M]    float32 输出均值
        var_ptr,        # [M]    float32 输出方差
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """对 [M, N] 矩阵的每行计算均值和无偏方差（Welford 算法）。"""
        row = tl.program_id(0)
        mean_acc = tl.zeros([1], dtype=tl.float32)
        m2_acc = tl.zeros([1], dtype=tl.float32)
        count = tl.zeros([1], dtype=tl.int32)

        for start in range(0, N, BLOCK_N):
            offsets = start + tl.arange(0, BLOCK_N)
            mask = offsets < N
            x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0)
            # Welford online update（Triton 向量化形式）
            valid = mask.to(tl.int32)
            delta = x - mean_acc
            count += valid
            mean_acc += tl.sum(tl.where(mask, delta / count.to(tl.float32), 0.0), axis=0)
            delta2 = x - mean_acc
            m2_acc += tl.sum(tl.where(mask, delta * delta2, 0.0), axis=0)

        var = tl.where(count > 1, m2_acc / (count - 1).to(tl.float32), 0.0)
        tl.store(mean_ptr + row, mean_acc)
        tl.store(var_ptr + row, var)

    @triton.jit
    def _zscore_kernel(
        x_ptr,
        out_ptr,
        mean_ptr,
        std_ptr,
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row = tl.program_id(0)
        mean_val = tl.load(mean_ptr + row)
        std_val = tl.load(std_ptr + row)
        std_safe = tl.where(std_val > 1e-8, std_val, 1.0)

        for start in range(0, N, BLOCK_N):
            offsets = start + tl.arange(0, BLOCK_N)
            mask = offsets < N
            x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0)
            z = (x - mean_val) / std_safe
            tl.store(out_ptr + row * N + offsets, z, mask=mask)


def welford_mean_var(
    x: torch.Tensor,
    dim: int = -1,
    unbiased: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    沿指定维度计算均值和方差（数值稳定的 Welford 算法）。

    Args:
        x       : 输入张量，任意形状
        dim     : 计算方向（默认最后一维）
        unbiased: 是否无偏方差

    Returns:
        (mean, var) 相同形状（去掉 dim 维度）的张量对
    """
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()

    orig_shape = x.shape
    M = x.numel() // orig_shape[-1]
    N = orig_shape[-1]
    x_2d = x.reshape(M, N).float().contiguous()

    if _TRITON_AVAILABLE and x.is_cuda:
        mean = torch.empty(M, dtype=torch.float32, device=x.device)
        var = torch.empty(M, dtype=torch.float32, device=x.device)
        BLOCK_N = min(1024, triton.next_power_of_2(N))
        _welford_row_kernel[(M,)](x_2d, mean, var, N=N, BLOCK_N=BLOCK_N)
    else:
        mean = x_2d.mean(dim=1)
        var = x_2d.var(dim=1, unbiased=unbiased)

    out_shape = orig_shape[:-1]
    if not out_shape:
        out_shape = torch.Size([1])
    return mean.reshape(out_shape), var.reshape(out_shape)


def zscore_normalize(
    x: torch.Tensor,
    dim: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    沿指定维度进行 z-score 标准化。

    Args:
        x  : 输入张量
        dim: 标准化方向
        eps: 防止除零的小量

    Returns:
        标准化后的张量，形状与 x 相同
    """
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()
        transposed = True
    else:
        transposed = False

    orig_shape = x.shape
    M = x.numel() // orig_shape[-1]
    N = orig_shape[-1]
    x_2d = x.reshape(M, N).float().contiguous()
    out_2d = torch.empty_like(x_2d)

    if _TRITON_AVAILABLE and x.is_cuda:
        mean, var = welford_mean_var(x_2d, dim=-1)
        std = (var + eps).sqrt()
        BLOCK_N = min(1024, triton.next_power_of_2(N))
        _zscore_kernel[(M,)](x_2d, out_2d, mean, std, N=N, BLOCK_N=BLOCK_N)
    else:
        mean = x_2d.mean(dim=1, keepdim=True)
        std = (x_2d.var(dim=1, keepdim=True) + eps).sqrt()
        out_2d = (x_2d - mean) / std

    result = out_2d.reshape(orig_shape)
    if transposed:
        result = result.transpose(dim, -1)
    return result


def pearson_correlation(
    x: torch.Tensor,
    y: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    计算两组特征向量之间的 Pearson 相关系数矩阵。

    Args:
        x  : [M, D] 张量
        y  : [N, D] 张量
        eps: 防止除零

    Returns:
        [M, N] 相关系数矩阵，值域 [-1, 1]
    """
    x = x.float()
    y = y.float()
    x_norm = zscore_normalize(x, dim=-1)
    y_norm = zscore_normalize(y, dim=-1)
    D = x.shape[-1]
    return (x_norm @ y_norm.T) / max(D - 1, 1)
