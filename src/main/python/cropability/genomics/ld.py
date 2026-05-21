"""
Linkage disequilibrium (LD) analysis
====================================
GPU-accelerated genome-wide LD matrix computation (r² and D' statistics).
Supports chunked parallel computation for large SNP matrices.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

from cropability.kernels.stats import pearson_correlation
from cropability.utils.logging import get_logger
from cropability.utils.timer import Timer

logger = get_logger(__name__)


@dataclass
class LDResult:
    """Container for LD computation results."""
    r2_matrix: torch.Tensor       # [P, P] r² matrix
    snp_positions: List[int]
    n_samples: int

    @property
    def n_snps(self) -> int:
        return len(self.snp_positions)

    def get_r2(self, pos_i: int, pos_j: int) -> float:
        """Return r² between two positions."""
        i = self.snp_positions.index(pos_i)
        j = self.snp_positions.index(pos_j)
        return float(self.r2_matrix[i, j].item())

    def high_ld_pairs(self, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """Return high-LD pairs (r² > threshold)."""
        pairs = []
        triu = torch.triu(self.r2_matrix, diagonal=1)
        idx = (triu > threshold).nonzero(as_tuple=False)
        for ij in idx:
            i, j = int(ij[0]), int(ij[1])
            pairs.append((
                self.snp_positions[i],
                self.snp_positions[j],
                float(self.r2_matrix[i, j].item()),
            ))
        return sorted(pairs, key=lambda x: -x[2])


class LDCalculator:
    """
    Genome-wide LD computation engine.

    Uses squared Pearson correlation (r²) as the LD statistic.
    Applies a chunking strategy for large SNP matrices to avoid OOM.

    Args:
        device    : GPU device
        chunk_size: Chunk size (SNPs per block)
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        chunk_size: int = 2000,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.chunk_size = chunk_size
        logger.info(f"LDCalculator on {device}, chunk_size={chunk_size}")

    def compute_ld_matrix(
        self,
        genotype_matrix: torch.Tensor,
        positions: Optional[List[int]] = None,
    ) -> LDResult:
        """
        Compute the LD matrix.

        Args:
            genotype_matrix: [N_samples, N_snps] float32,
                             each element is genotype (0/1/2 = hom-ref/het/hom-alt)
            positions      : SNP position list (None uses indices)

        Returns:
            LDResult object
        """
        n_samples, n_snps = genotype_matrix.shape
        if positions is None:
            positions = list(range(n_snps))

        logger.info(f"Computing LD matrix: {n_snps} SNPs × {n_samples} samples")

        geno = genotype_matrix.float().to(self.device)

        with Timer("ld_computation") as t:
            if n_snps <= self.chunk_size:
                r2 = self._compute_r2_block(geno)
            else:
                r2 = self._compute_r2_chunked(geno)

        logger.info(f"LD matrix computed in {t.elapsed_ms:.1f} ms")
        return LDResult(r2_matrix=r2, snp_positions=positions, n_samples=n_samples)

    def _compute_r2_block(self, geno: torch.Tensor) -> torch.Tensor:
        """Small scale: compute the full r² matrix in one pass."""
        # geno: [N, P] → [P, P] correlation matrix
        r = pearson_correlation(geno.T, geno.T)  # [P, P]
        r2 = r.pow(2).clamp(0, 1)
        return r2

    def _compute_r2_chunked(self, geno: torch.Tensor) -> torch.Tensor:
        """Large scale: chunked computation to avoid OOM."""
        n_snps = geno.shape[1]
        r2 = torch.zeros(n_snps, n_snps, device=self.device)

        chunks = list(range(0, n_snps, self.chunk_size))
        for i, start_i in enumerate(chunks):
            end_i = min(start_i + self.chunk_size, n_snps)
            block_i = geno[:, start_i:end_i].T  # [chunk, N]
            for start_j in chunks[i:]:
                end_j = min(start_j + self.chunk_size, n_snps)
                block_j = geno[:, start_j:end_j].T
                r_block = pearson_correlation(block_i, block_j)
                r2_block = r_block.pow(2).clamp(0, 1)
                r2[start_i:end_i, start_j:end_j] = r2_block
                if start_i != start_j:
                    r2[start_j:end_j, start_i:end_i] = r2_block.T

            logger.debug(f"LD chunk {i+1}/{len(chunks)} done")

        return r2

    def prune_by_ld(
        self,
        r2_matrix: torch.Tensor,
        positions: List[int],
        threshold: float = 0.5,
    ) -> List[int]:
        """
        LD pruning: greedily remove high-LD SNPs, keeping an independent set.

        Args:
            r2_matrix : [P, P] r² matrix
            positions : SNP position list
            threshold : r² cutoff (one of a pair is pruned if exceeded)

        Returns:
            List of retained SNP positions
        """
        p = len(positions)
        kept = list(range(p))
        removed = set()

        for i in range(p):
            if i in removed:
                continue
            for j in range(i + 1, p):
                if j in removed:
                    continue
                if float(r2_matrix[i, j].item()) > threshold:
                    removed.add(j)

        retained = [positions[i] for i in kept if i not in removed]
        logger.info(
            f"LD pruning (r²>{threshold}): {p} → {len(retained)} SNPs retained"
        )
        return retained
