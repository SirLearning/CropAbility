"""
Matrix operation GPU kernels
============================
Symmetric matrix-vector product, batch outer product, upper-triangular sum.
These operations appear frequently in genomic LD and covariance analysis.
"""

from __future__ import annotations

from typing import Optional

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
# Triton kernel: symmetric matrix-vector product (upper triangle only)
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _symm_matvec_kernel(
        A_ptr,          # [N, N] symmetric float32 (upper triangle stored)
        x_ptr,          # [N] float32
        y_ptr,          # [N] float32 output
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """y = A @ x for symmetric A; reads upper triangle only to reduce memory traffic."""
        row = tl.program_id(0)
        acc = tl.zeros([1], dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < N
            # Symmetry: A[row,col] == A[col,row]
            idx = tl.where(cols >= row, row * N + cols, cols * N + row)
            a = tl.load(A_ptr + idx, mask=mask, other=0.0)
            xv = tl.load(x_ptr + cols, mask=mask, other=0.0)
            acc += tl.sum(a * xv, axis=0)
        tl.store(y_ptr + row, acc)

    @triton.jit
    def _upper_tri_sum_kernel(
        A_ptr,          # [N, N] float32
        out_ptr,        # scalar float32
        N: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Sum strict upper triangle (excluding diagonal) of [N, N] matrix."""
        pid = tl.program_id(0)
        row = pid
        acc = tl.zeros([1], dtype=tl.float32)
        for col_start in range(row + 1, N, BLOCK):
            cols = col_start + tl.arange(0, BLOCK)
            mask = cols < N
            val = tl.load(A_ptr + row * N + cols, mask=mask, other=0.0)
            acc += tl.sum(val, axis=0)
        # Atomic add (concurrent rows)
        tl.atomic_add(out_ptr, acc)


def symm_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    Symmetric matrix multiply via cuBLAS SSYMM semantics (through torch.mm).
    When both sides are square and identical, equivalent to A @ A^T (common in genomic covariance).

    Args:
        A: [M, K] or [N, N] symmetric matrix
        B: [K, N]

    Returns:
        [M, N] product
    """
    A = A.float()
    B = B.float()
    if A.is_cuda and B.is_cuda:
        # torch.mm on CUDA calls highly optimized cuBLAS SGEMM
        return torch.mm(A, B)
    return torch.mm(A, B)


def batch_outer_product(
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    Batch outer product: u[i] ⊗ v[i] for each pair of row vectors.

    Args:
        u: [B, M] tensor
        v: [B, N] tensor

    Returns:
        [B, M, N] outer product tensor
    """
    u = u.float()
    v = v.float()
    return torch.bmm(u.unsqueeze(2), v.unsqueeze(1))


def triangular_sum(A: torch.Tensor, include_diagonal: bool = False) -> torch.Tensor:
    """
    Sum upper-triangular elements of a square matrix (common LD matrix statistic).

    Args:
        A               : [N, N] tensor
        include_diagonal: Whether to include the main diagonal

    Returns:
        Scalar tensor
    """
    N = A.shape[0]
    assert A.shape == (N, N), "triangular_sum requires square matrix"
    if include_diagonal:
        mask = torch.ones(N, N, dtype=torch.bool, device=A.device).triu()
    else:
        mask = torch.ones(N, N, dtype=torch.bool, device=A.device).triu(diagonal=1)
    return A[mask].sum()
