"""
矩阵运算 GPU 内核
=================
对称矩阵-向量积、批量外积、上三角求和。
这些操作在基因组 LD（连锁不平衡）和协方差分析中频繁出现。
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
# Triton 内核：对称矩阵向量积（利用对称性只读上三角）
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _symm_matvec_kernel(
        A_ptr,          # [N, N] symmetric float32（只存储上三角）
        x_ptr,          # [N] float32
        y_ptr,          # [N] float32 输出
        N: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        """y = A @ x，A 为对称矩阵，只读取上三角减少内存访问。"""
        row = tl.program_id(0)
        acc = tl.zeros([1], dtype=tl.float32)
        for start in range(0, N, BLOCK_N):
            cols = start + tl.arange(0, BLOCK_N)
            mask = cols < N
            # 利用对称性：A[row,col] == A[col,row]
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
        """对 [N, N] 矩阵严格上三角（不含对角线）求和。"""
        pid = tl.program_id(0)
        row = pid
        acc = tl.zeros([1], dtype=tl.float32)
        for col_start in range(row + 1, N, BLOCK):
            cols = col_start + tl.arange(0, BLOCK)
            mask = cols < N
            val = tl.load(A_ptr + row * N + cols, mask=mask, other=0.0)
            acc += tl.sum(val, axis=0)
        # 原子加（多行并发）
        tl.atomic_add(out_ptr, acc)


def symm_matmul(
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """
    利用 cuBLAS SSYMM 语义（通过 torch.mm）执行对称矩阵乘法。
    当两侧矩阵都是方阵且相同时，等同于 A @ A^T（基因组协方差常见模式）。

    Args:
        A: [M, K] 或 [N, N] 对称矩阵
        B: [K, N]

    Returns:
        [M, N] 乘积
    """
    A = A.float()
    B = B.float()
    if A.is_cuda and B.is_cuda:
        # torch.mm 在 CUDA 上调用 cuBLAS SGEMM，已高度优化
        return torch.mm(A, B)
    return torch.mm(A, B)


def batch_outer_product(
    u: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """
    批量外积：每对行向量 u[i] ⊗ v[i]。

    Args:
        u: [B, M] 张量
        v: [B, N] 张量

    Returns:
        [B, M, N] 外积张量
    """
    u = u.float()
    v = v.float()
    return torch.bmm(u.unsqueeze(2), v.unsqueeze(1))


def triangular_sum(A: torch.Tensor, include_diagonal: bool = False) -> torch.Tensor:
    """
    计算方阵上三角元素之和（常用于基因组 LD 矩阵的统计量）。

    Args:
        A               : [N, N] 张量
        include_diagonal: 是否包含主对角线

    Returns:
        标量 tensor
    """
    N = A.shape[0]
    assert A.shape == (N, N), "triangular_sum requires square matrix"
    if include_diagonal:
        mask = torch.ones(N, N, dtype=torch.bool, device=A.device).triu()
    else:
        mask = torch.ones(N, N, dtype=torch.bool, device=A.device).triu(diagonal=1)
    return A[mask].sum()
