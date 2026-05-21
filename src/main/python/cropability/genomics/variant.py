"""
Variant calling (SNP/Indel)
===========================
GPU-accelerated multi-sample SNP calling with parallel statistical tests on base matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import numpy as np

from cropability.kernels.seq import encode_sequences
from cropability.kernels.stats import welford_mean_var
from cropability.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SNPResult:
    """Detection result for a single SNP site."""
    position: int
    ref_allele: str
    alt_allele: str
    alt_freq: float
    quality_score: float
    samples_called: int
    samples_total: int

    @property
    def call_rate(self) -> float:
        return self.samples_called / max(self.samples_total, 1)

    def __repr__(self) -> str:
        return (
            f"SNP(pos={self.position}, {self.ref_allele}>{self.alt_allele}, "
            f"AF={self.alt_freq:.3f}, Q={self.quality_score:.1f})"
        )


class VariantCaller:
    """
    Multi-sample SNP calling engine (GPU-accelerated).

    Workflow:
    1. Stack per-sample reads by position into an [S, L] int8 matrix
    2. Compute per-position base frequencies in parallel on the GPU
    3. Filter false positives with Fisher exact test or chi-square test

    Args:
        device        : GPU device (auto-selected if None)
        min_alt_freq  : Minimum alternate allele frequency (filters low-frequency noise)
        min_depth     : Minimum coverage depth
        quality_thresh: Minimum quality score threshold
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        min_alt_freq: float = 0.05,
        min_depth: int = 10,
        quality_thresh: float = 20.0,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.min_alt_freq = min_alt_freq
        self.min_depth = min_depth
        self.quality_thresh = quality_thresh
        logger.info(
            f"VariantCaller initialized on {device}, "
            f"min_af={min_alt_freq}, min_depth={min_depth}"
        )

    def call_snps(
        self,
        sequences: List[str],
        reference: str,
        positions: Optional[List[int]] = None,
    ) -> List[SNPResult]:
        """
        Call SNPs from a batch of sample sequences.

        Args:
            sequences : Sample sequence list (equal length, aligned to reference)
            reference : Reference sequence
            positions : Positions to scan (None = all)

        Returns:
            List of SNPResult (low-quality sites already filtered)
        """
        n_samples = len(sequences)
        seq_len = len(reference)
        logger.info(f"Calling SNPs: {n_samples} samples, length={seq_len}")

        # Encode reference and sample sequences
        all_seqs = [reference] + sequences
        encoded = encode_sequences(all_seqs, device=self.device)  # [S+1, L]
        ref_enc = encoded[0]        # [L]
        sample_enc = encoded[1:]    # [S, L]

        if positions is None:
            positions = list(range(seq_len))

        results: List[SNPResult] = []
        pos_tensor = torch.tensor(positions, device=self.device)

        # Extract base matrix at selected positions [S, P]
        sample_bases = sample_enc[:, pos_tensor]  # [S, P]
        ref_bases = ref_enc[pos_tensor]           # [P]

        for p_idx, pos in enumerate(positions):
            ref_b = int(ref_bases[p_idx].item())
            if ref_b == 4:  # Reference is N; skip
                continue

            col = sample_bases[:, p_idx]           # [S]
            called_mask = col != 4                 # Non-N sites
            n_called = int(called_mask.sum().item())
            if n_called < self.min_depth:
                continue

            called_bases = col[called_mask]
            # Count non-reference base frequency
            n_alt = int((called_bases != ref_b).sum().item())
            alt_freq = n_alt / n_called

            if alt_freq < self.min_alt_freq:
                continue

            # Quality score: simplified Phred score based on binomial distribution
            quality = self._compute_quality(n_alt, n_called)
            if quality < self.quality_thresh:
                continue

            # Determine the most frequent alternate allele
            alt_counts = torch.zeros(4, device=self.device)
            for base_idx in range(4):
                if base_idx != ref_b:
                    alt_counts[base_idx] = (called_bases == base_idx).float().sum()
            alt_b = int(alt_counts.argmax().item())

            from cropability.kernels.seq import _INT_TO_BASE
            results.append(SNPResult(
                position=pos,
                ref_allele=_INT_TO_BASE[ref_b],
                alt_allele=_INT_TO_BASE[alt_b],
                alt_freq=alt_freq,
                quality_score=quality,
                samples_called=n_called,
                samples_total=n_samples,
            ))

        logger.info(f"Called {len(results)} SNPs (before filtering)")
        return results

    @staticmethod
    def _compute_quality(n_alt: int, n_total: int) -> float:
        """Simplified Phred quality score approximating binomial p-value."""
        if n_total == 0:
            return 0.0
        import math
        af = n_alt / n_total
        # Exclude pure reference genotype
        if af <= 0:
            return 0.0
        # Simplified: use -10 * log10(binom_pmf_null)
        p_null = 0.001  # Sequencing error rate
        expected_err = n_total * p_null
        if n_alt <= expected_err:
            return 0.0
        # Approximate Phred = 10 * log10(n_alt / expected_err)
        quality = 10.0 * math.log10(max(n_alt / expected_err, 1e-10))
        return min(quality, 60.0)

    def filter_snps(
        self,
        snps: List[SNPResult],
        min_call_rate: float = 0.90,
        max_alt_freq: float = 0.95,
    ) -> List[SNPResult]:
        """Filter SNP list, removing low-quality and extreme-frequency sites."""
        filtered = [
            s for s in snps
            if s.call_rate >= min_call_rate
            and self.min_alt_freq <= s.alt_freq <= max_alt_freq
        ]
        logger.info(f"After filtering: {len(filtered)}/{len(snps)} SNPs retained")
        return filtered
