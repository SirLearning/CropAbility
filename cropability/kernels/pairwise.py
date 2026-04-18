"""
成对距离/相似度矩阵 GPU 内核
=============================
用于基因组序列比较的成对距离计算：
- Hamming 距离（SNP 差异计数）
- Jaccard 相似度（k-mer 集合比较）
"""

from __future__ import annotations

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
# Triton 内核：批量 Hamming 距离矩阵
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    @triton.jit
    def _hamming_kernel(
        A_ptr,        # [M, L] int8
        B_ptr,        # [N, L] int8
        out_ptr,      # [M, N] int32
        L: tl.constexpr,
        BLOCK_L: tl.constexpr,
    ):
        """计算 A 和 B 所有行对之间的 Hamming 距离（忽略 N=4）。"""
        m_idx = tl.program_id(0)
        n_idx = tl.program_id(1)
        dist = tl.zeros([1], dtype=tl.int32)
        for start in range(0, L, BLOCK_L):
            offsets = start + tl.arange(0, BLOCK_L)
            mask = offsets < L
            a = tl.load(A_ptr + m_idx * L + offsets, mask=mask, other=4)
            b = tl.load(B_ptr + n_idx * L + offsets, mask=mask, other=4)
            valid = ((a != 4) & (b != 4)) & mask
            diff = (a != b) & valid
            dist += tl.sum(diff.to(tl.int32), axis=0)
        tl.store(out_ptr + m_idx * tl.num_programs(1) + n_idx, dist)


def hamming_distance_matrix(
    A: torch.Tensor,
    B: Optional[torch.Tensor] = None,
    normalize: bool = True,
) -> torch.Tensor:
    """
    计算编码序列的成对 Hamming 距离矩阵。

    Args:
        A        : [M, L] int8 编码序列
        B        : [N, L] int8 编码序列（None 则与 A 自身比较）
        normalize: 是否归一化到 [0, 1]（除以有效碱基位数）

    Returns:
        [M, N] 距离矩阵（normalize=True 时为 float32，否则为 int32）
    """
    if B is None:
        B = A
    M, L = A.shape
    N = B.shape[0]
    assert B.shape[1] == L, "序列长度必须相同"

    A_c = A.contiguous()
    B_c = B.contiguous()

    if _TRITON_AVAILABLE and A.is_cuda:
        out = torch.zeros(M, N, dtype=torch.int32, device=A.device)
        BLOCK_L = min(1024, triton.next_power_of_2(L))
        _hamming_kernel[(M, N)](A_c, B_c, out, L=L, BLOCK_L=BLOCK_L)
    else:
        # CPU / non-Triton fallback: 广播向量化
        out = ((A_c.unsqueeze(1) != B_c.unsqueeze(0)) &
               (A_c.unsqueeze(1) != 4) &
               (B_c.unsqueeze(0) != 4)).int().sum(dim=2)

    if normalize:
        valid_a = (A_c != 4).int().sum(dim=1, keepdim=True).float()
        valid_b = (B_c != 4).int().sum(dim=1).float().unsqueeze(0)
        denom = torch.min(valid_a.expand(M, N), valid_b.expand(M, N)).clamp(min=1)
        return out.float() / denom
    return out


def jaccard_similarity_matrix(
    A: torch.Tensor,
    B: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    基于 k-mer 频率向量计算 Jaccard 相似度矩阵（使用 min-over-max 近似）。

    Args:
        A: [M, K] float32 k-mer 频率矩阵（来自 kmer_count_kernel）
        B: [N, K] float32（None 则与 A 自身比较）

    Returns:
        [M, N] float32 相似度矩阵，值域 [0, 1]
    """
    if B is None:
        B = A
    A = A.float()
    B = B.float()

    # min(a_i, b_j) 的和 / max(a_i, b_j) 的和
    # 广播: [M, 1, K] vs [1, N, K]
    A_exp = A.unsqueeze(1)
    B_exp = B.unsqueeze(0)
    intersection = torch.min(A_exp, B_exp).sum(dim=2)
    union = torch.max(A_exp, B_exp).sum(dim=2).clamp(min=1e-8)
    return intersection / union


# 允许 Optional 引用
from typing import Optional
