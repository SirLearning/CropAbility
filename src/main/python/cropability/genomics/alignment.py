"""
GPU-accelerated sequence alignment
==================================
Batch GPU implementation of Smith-Waterman local alignment.
Computes fast alignment score matrices for bulk query-vs-database comparisons.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from cropability.kernels.seq import encode_sequences
from cropability.utils.logging import get_logger
from cropability.utils.timer import Timer

logger = get_logger(__name__)


@dataclass
class AlignmentResult:
    """Local alignment result."""
    query_idx: int
    target_idx: int
    score: float
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    cigar: Optional[str] = None  # Placeholder for now

    def __repr__(self) -> str:
        return (
            f"Align(q={self.query_idx}[{self.query_start}:{self.query_end}] "
            f"t={self.target_idx}[{self.target_start}:{self.target_end}] "
            f"score={self.score:.1f})"
        )


class SmithWatermanGPU:
    """
    Batch GPU Smith-Waterman local alignment.

    Strategy: use PyTorch vectorization to compute score matrices for all
    query-target pairs in parallel and track the highest-scoring positions.
    Suitable for bulk screening of medium/short reads (<= 1000 bp).

    Args:
        device        : GPU device
        match_score   : Base match score
        mismatch_score: Base mismatch penalty (negative)
        gap_open      : Gap open penalty (negative)
        gap_extend    : Gap extension penalty (negative)
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        match_score: float = 2.0,
        mismatch_score: float = -1.0,
        gap_open: float = -2.0,
        gap_extend: float = -0.5,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.match = match_score
        self.mismatch = mismatch_score
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        logger.info(
            f"SmithWatermanGPU on {device}: "
            f"match={match_score}, mismatch={mismatch_score}"
        )

    def score_matrix(
        self,
        queries: List[str],
        targets: List[str],
    ) -> torch.Tensor:
        """
        Compute optimal local alignment scores for all query-target pairs.

        Args:
            queries : Query sequence list (length <= 1000)
            targets : Target sequence list

        Returns:
            [Q, T] float32 score matrix
        """
        q_enc = encode_sequences(queries, device=self.device)   # [Q, Lq]
        t_enc = encode_sequences(targets, device=self.device)   # [T, Lt]
        Q, Lq = q_enc.shape
        T, Lt = t_enc.shape

        logger.info(f"Smith-Waterman: {Q} queries × {T} targets, Lq={Lq}, Lt={Lt}")

        with Timer("sw_score") as timer:
            scores = self._batch_sw_score(q_enc, t_enc)

        logger.info(f"SW scoring done in {timer.elapsed_ms:.1f} ms")
        return scores

    def _batch_sw_score(
        self,
        q_enc: torch.Tensor,   # [Q, Lq]
        t_enc: torch.Tensor,   # [T, Lt]
    ) -> torch.Tensor:
        """
        Vectorized Smith-Waterman scoring (linear gap penalty, no traceback).
        Uses anti-diagonal parallelism.
        """
        Q, Lq = q_enc.shape
        T, Lt = t_enc.shape

        # Substitution matrix: match for identical bases, mismatch otherwise, 0 for N
        # Precompute [Q, T, Lq, Lt] substitution scores (may be large; chunk if needed)
        # Simplified version: column-wise scan with [Q*T, 1] state tensor

        # Avoid Q*T*Lq*Lt memory blowup: column-wise iteration with GPU parallelism
        # H[i,j] = max(0, H[i-1,j-1]+s(q[i],t[j]), H[i-1,j]+gap, H[i,j-1]+gap)

        # Broadcast would be [Q, 1, Lq] vs [1, T, Lt] → [Q, T, Lq, Lt] (too large)
        # Column-wise iteration: each step processes one column of the full [Q, T] matrix
        H = torch.zeros(Q, T, Lq + 1, device=self.device)   # Sliding window only
        best = torch.zeros(Q, T, device=self.device)

        for j in range(Lt):
            target_col = t_enc[:, j].long()           # [T]
            # Substitution score sub[q, t, i] = match if q_enc[q,i]==target_col[t] else mismatch
            # q_enc[:, :] [Q, Lq]; target_col [T]
            q_bases = q_enc.long()                    # [Q, Lq]
            t_base_j = target_col.unsqueeze(0).unsqueeze(2)  # [1, T, 1]
            q_bases_exp = q_bases.unsqueeze(1)        # [Q, 1, Lq]
            match_mask = (q_bases_exp == t_base_j)    # [Q, T, Lq]
            n_mask = (q_bases_exp == 4) | (t_base_j == 4)
            sub = torch.where(
                n_mask, torch.zeros_like(match_mask, dtype=torch.float32),
                torch.where(match_mask,
                            torch.full_like(match_mask, self.match, dtype=torch.float32),
                            torch.full_like(match_mask, self.mismatch, dtype=torch.float32))
            )  # [Q, T, Lq]

            # DP recurrence (simplified linear gap)
            H_prev = H[:, :, :-1]   # [Q, T, Lq]
            H_new = (H_prev + sub).clamp(min=0)
            # Gap penalty (uniform penalty; open/extend not distinguished)
            H_new = torch.max(H_new, (H[:, :, 1:] + self.gap_open).clamp(min=0))

            H[:, :, 1:] = H_new
            best = torch.max(best, H_new.max(dim=2).values)

        return best

    def find_top_hits(
        self,
        scores: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[int, int, float]]:
        """
        Find top-k alignment hits in the score matrix.

        Args:
            scores   : [Q, T] score matrix
            top_k    : Best targets returned per query
            threshold: Minimum score threshold

        Returns:
            List of (query_idx, target_idx, score) triples
        """
        Q = scores.shape[0]
        top_scores, top_idx = torch.topk(scores, min(top_k, scores.shape[1]), dim=1)
        hits = []
        for q in range(Q):
            for k in range(top_scores.shape[1]):
                s = float(top_scores[q, k].item())
                if s >= threshold:
                    hits.append((q, int(top_idx[q, k].item()), s))
        return hits
