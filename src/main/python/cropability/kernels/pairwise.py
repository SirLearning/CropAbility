"""
Pairwise distance/similarity matrix GPU kernels
===============================================
Pairwise distance computation for genomic sequence comparison:
- Hamming distance (SNP difference count)
- Jaccard similarity (k-mer set comparison)
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
# Triton kernel: batch Hamming distance matrix
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
        """Compute Hamming distance between all row pairs of A and B (ignore N=4)."""
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
    Compute pairwise Hamming distance matrix for encoded sequences.

    Args:
        A        : [M, L] int8 encoded sequences
        B        : [N, L] int8 encoded sequences (None compares A to itself)
        normalize: Whether to normalize to [0, 1] (divide by valid base count)

    Returns:
        [M, N] distance matrix (float32 if normalize=True, else int32)
    """
    if B is None:
        B = A
    M, L = A.shape
    N = B.shape[0]
    assert B.shape[1] == L, "Sequence lengths must match"

    A_c = A.contiguous()
    B_c = B.contiguous()

    if _TRITON_AVAILABLE and A.is_cuda:
        out = torch.zeros(M, N, dtype=torch.int32, device=A.device)
        BLOCK_L = min(1024, triton.next_power_of_2(L))
        _hamming_kernel[(M, N)](A_c, B_c, out, L=L, BLOCK_L=BLOCK_L)
    else:
        # CPU / non-Triton fallback: broadcast vectorization
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
    Compute Jaccard similarity matrix from k-mer frequency vectors (min-over-max approximation).

    Args:
        A: [M, K] float32 k-mer frequency matrix (from kmer_count_kernel)
        B: [N, K] float32 (None compares A to itself)

    Returns:
        [M, N] float32 similarity matrix with values in [0, 1]
    """
    if B is None:
        B = A
    A = A.float()
    B = B.float()

    # sum of min(a_i, b_j) / sum of max(a_i, b_j)
    # Broadcast: [M, 1, K] vs [1, N, K]
    A_exp = A.unsqueeze(1)
    B_exp = B.unsqueeze(0)
    intersection = torch.min(A_exp, B_exp).sum(dim=2)
    union = torch.max(A_exp, B_exp).sum(dim=2).clamp(min=1e-8)
    return intersection / union


# Allow Optional reference
from typing import Optional
