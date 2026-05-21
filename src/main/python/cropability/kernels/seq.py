"""
Sequence processing GPU kernels
===============================
All kernels accept int8-encoded base sequences (A=0 C=1 G=2 T=3 N=4).
CPU fallback is used automatically when Triton/CUDA is unavailable.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch

from cropability.utils.logging import get_logger

logger = get_logger(__name__)

# Base encoding dictionary (ASCII → 0-4)
_BASE_TO_INT: dict[int, int] = {
    ord("A"): 0, ord("a"): 0,
    ord("C"): 1, ord("c"): 1,
    ord("G"): 2, ord("g"): 2,
    ord("T"): 3, ord("t"): 3,
    ord("U"): 3, ord("u"): 3,  # RNA
    ord("N"): 4, ord("n"): 4,
}
_INT_TO_BASE: list[str] = ["A", "C", "G", "T", "N"]
_COMPLEMENT: dict[int, int] = {0: 3, 1: 2, 2: 1, 3: 0, 4: 4}  # A↔T, C↔G

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False
    logger.debug("Triton unavailable; sequence kernels will use PyTorch implementation.")


# ---------------------------------------------------------------------------
# Triton kernel: GC content
# ---------------------------------------------------------------------------

if _TRITON_AVAILABLE:
    import triton
    import triton.language as tl

    @triton.jit
    def _gc_content_triton(
        seq_ptr,       # [N, L] int8
        out_ptr,       # [N] float32
        seq_len: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """Compute GC content per sequence (G=2, C=1 counted as GC)."""
        row = tl.program_id(0)
        gc_count = tl.zeros([1], dtype=tl.int32)
        valid_count = tl.zeros([1], dtype=tl.int32)

        for start in range(0, seq_len, BLOCK):
            offsets = start + tl.arange(0, BLOCK)
            mask = offsets < seq_len
            base = tl.load(seq_ptr + row * seq_len + offsets, mask=mask, other=4)
            is_gc = ((base == 1) | (base == 2)) & mask
            is_valid = (base != 4) & mask
            gc_count += tl.sum(is_gc.to(tl.int32), axis=0)
            valid_count += tl.sum(is_valid.to(tl.int32), axis=0)

        ratio = tl.where(valid_count > 0, gc_count.to(tl.float32) / valid_count.to(tl.float32), 0.0)
        tl.store(out_ptr + row, ratio)

    @triton.jit
    def _encode_seq_triton(
        ascii_ptr,    # [N, L] uint8  (ASCII bytes)
        out_ptr,      # [N, L] int8   (0-4 encoding)
        n_elements: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offsets = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        ch = tl.load(ascii_ptr + offsets, mask=mask, other=ord("N"))
        # Simplified character mapping (uppercase)
        encoded = tl.where(ch == ord("A"), 0,
                  tl.where(ch == ord("C"), 1,
                  tl.where(ch == ord("G"), 2,
                  tl.where(ch == ord("T"), 3,
                  tl.where(ch == ord("U"), 3, 4)))))
        # Lowercase
        encoded = tl.where(ch == ord("a"), 0,
                  tl.where(ch == ord("c"), 1,
                  tl.where(ch == ord("g"), 2,
                  tl.where(ch == ord("t"), 3, encoded))))
        tl.store(out_ptr + offsets, encoded.to(tl.int8), mask=mask)


def encode_sequences(
    sequences: List[str],
    device: Optional[torch.device] = None,
    pad_value: int = 4,
) -> torch.Tensor:
    """
    Encode a list of string sequences as an int8 tensor [N, max_len].

    Args:
        sequences: DNA/RNA sequence list (may differ in length)
        device   : Target device; auto-selected if None
        pad_value: Padding value (default 4 = N)

    Returns:
        torch.Tensor of shape [N, max_len], dtype=torch.int8
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n = len(sequences)
    if n == 0:
        return torch.zeros(0, 0, dtype=torch.int8, device=device)

    max_len = max(len(s) for s in sequences)
    result = np.full((n, max_len), pad_value, dtype=np.int8)

    for i, seq in enumerate(sequences):
        for j, ch in enumerate(seq):
            result[i, j] = _BASE_TO_INT.get(ord(ch), 4)

    tensor = torch.from_numpy(result).to(device)

    if _TRITON_AVAILABLE and device.type == "cuda":
        logger.debug(f"encode_sequences: Triton CUDA path, N={n}, L={max_len}")
    else:
        logger.debug(f"encode_sequences: CPU/numpy path, N={n}, L={max_len}")

    return tensor


def gc_content_kernel(
    encoded: torch.Tensor,
    block_size: int = 1024,
) -> torch.Tensor:
    """
    Compute per-row GC content for an encoded sequence matrix.

    Args:
        encoded   : [N, L] int8 tensor (from encode_sequences)
        block_size: Triton block size

    Returns:
        [N] float32 tensor with values in [0, 1]
    """
    n, seq_len = encoded.shape
    out = torch.zeros(n, dtype=torch.float32, device=encoded.device)

    if _TRITON_AVAILABLE and encoded.is_cuda:
        _gc_content_triton[(n,)](
            encoded,
            out,
            seq_len=seq_len,
            BLOCK=block_size,
        )
    else:
        # CPU fallback
        gc_mask = (encoded == 1) | (encoded == 2)
        valid_mask = encoded != 4
        gc_sum = gc_mask.float().sum(dim=1)
        valid_sum = valid_mask.float().sum(dim=1).clamp(min=1)
        out = gc_sum / valid_sum

    return out


def reverse_complement_kernel(encoded: torch.Tensor) -> torch.Tensor:
    """
    Compute the reverse complement of encoded sequences.

    Args:
        encoded: [N, L] int8

    Returns:
        [N, L] int8, reverse-complemented per row
    """
    comp_table = torch.tensor([3, 2, 1, 0, 4], dtype=torch.int8, device=encoded.device)
    # PyTorch index lookup (no Triton needed; memory is contiguous)
    complemented = comp_table[encoded.long()]
    return complemented.flip(dims=[1]).to(torch.int8)


def kmer_count_kernel(
    encoded: torch.Tensor,
    k: int = 4,
) -> torch.Tensor:
    """
    Count k-mer frequencies per sequence (only k-mers without N).

    Args:
        encoded: [N, L] int8
        k      : k-mer length

    Returns:
        [N, 4^k] float32 frequency tensor
    """
    n, seq_len = encoded.shape
    vocab = 4 ** k
    counts = torch.zeros(n, vocab, dtype=torch.float32, device=encoded.device)

    if seq_len < k:
        return counts

    # Build sliding window indices (positions without N)
    for start in range(seq_len - k + 1):
        window = encoded[:, start : start + k].long()  # [N, k]
        has_n = (window == 4).any(dim=1)               # [N]
        # Encode k-mer as a single integer index (base-4)
        powers = (4 ** torch.arange(k - 1, -1, -1, device=encoded.device)).long()
        idx = (window * powers.unsqueeze(0)).sum(dim=1)  # [N]
        idx = idx.masked_fill(has_n, -1)
        valid_mask = idx >= 0
        # Safe scatter_add
        valid_n = valid_mask.sum().item()
        if valid_n > 0:
            rows = torch.where(valid_mask)[0]
            cols = idx[valid_mask]
            counts[rows, cols] += 1.0

    # Normalize
    total = counts.sum(dim=1, keepdim=True).clamp(min=1)
    return counts / total
