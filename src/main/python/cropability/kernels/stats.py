"""
Statistical GPU kernels
=======================
Welford online mean/variance, z-score normalization, Pearson correlation.
Supports Triton GPU paths and PyTorch CPU/GPU fallback.
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
# Triton kernel: per-row Welford mean and variance
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _welford_row_kernel(
        x_ptr,          # [M, N] float32 input
        mean_ptr,       # [M]    float32 output mean
        var_ptr,        # [M]    float32 output variance
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """Compute mean and unbiased variance per row of [M, N] (Welford algorithm)."""
        row = tl.program_id(0)
        mean_acc = tl.zeros([1], dtype=tl.float32)
        m2_acc = tl.zeros([1], dtype=tl.float32)
        count = tl.zeros([1], dtype=tl.int32)

        for start in range(0, N, BLOCK_N):
            offsets = start + tl.arange(0, BLOCK_N)
            mask = offsets < N
            x = tl.load(x_ptr + row * N + offsets, mask=mask, other=0.0)
            # Welford online update (vectorized Triton form)
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
    Compute mean and variance along a dimension (numerically stable Welford algorithm).

    Args:
        x       : Input tensor of any shape
        dim     : Reduction dimension (default last)
        unbiased: Whether to use unbiased variance

    Returns:
        (mean, var) tensor pair with dim removed
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
    Apply z-score normalization along a dimension.

    Args:
        x  : Input tensor
        dim: Normalization dimension
        eps: Small constant to prevent division by zero

    Returns:
        Normalized tensor with the same shape as x
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
    Compute Pearson correlation matrix between two feature matrices.

    Args:
        x  : [M, D] tensor
        y  : [N, D] tensor
        eps: Small constant to prevent division by zero

    Returns:
        [M, N] correlation matrix with values in [-1, 1]
    """
    x = x.float()
    y = y.float()
    x_norm = zscore_normalize(x, dim=-1)
    y_norm = zscore_normalize(y, dim=-1)
    D = x.shape[-1]
    return (x_norm @ y_norm.T) / max(D - 1, 1)
