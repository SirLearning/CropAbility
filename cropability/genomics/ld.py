"""
连锁不平衡（LD）分析
====================
GPU 加速的全基因组 LD 矩阵计算（r² 和 D' 统计量）。
支持大规模 SNP 矩阵的分块并行计算。
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
    """LD 计算结果容器。"""
    r2_matrix: torch.Tensor       # [P, P] r² 矩阵
    snp_positions: List[int]
    n_samples: int

    @property
    def n_snps(self) -> int:
        return len(self.snp_positions)

    def get_r2(self, pos_i: int, pos_j: int) -> float:
        """获取两个位点之间的 r² 值。"""
        i = self.snp_positions.index(pos_i)
        j = self.snp_positions.index(pos_j)
        return float(self.r2_matrix[i, j].item())

    def high_ld_pairs(self, threshold: float = 0.8) -> List[Tuple[int, int, float]]:
        """返回高 LD 对（r² > threshold）。"""
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
    全基因组 LD 计算引擎。

    使用 Pearson 相关系数的平方（r²）作为 LD 统计量。
    对大规模 SNP 矩阵采用分块策略避免显存溢出。

    Args:
        device    : GPU 设备
        chunk_size: 分块大小（每块 SNP 数）
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
        计算 LD 矩阵。

        Args:
            genotype_matrix: [N_samples, N_snps] float32，
                             每个元素为基因型（0/1/2 = 纯合参考/杂合/纯合替代）
            positions      : SNP 位置列表（None 则用序号代替）

        Returns:
            LDResult 对象
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
        """小规模：一次性计算完整 r² 矩阵。"""
        # geno: [N, P]，计算 [P, P] 相关矩阵
        r = pearson_correlation(geno.T, geno.T)  # [P, P]
        r2 = r.pow(2).clamp(0, 1)
        return r2

    def _compute_r2_chunked(self, geno: torch.Tensor) -> torch.Tensor:
        """大规模：分块计算，避免显存爆炸。"""
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
        LD pruning：贪心移除高 LD SNP，保留独立位点集合。

        Args:
            r2_matrix : [P, P] r² 矩阵
            positions : SNP 位置列表
            threshold : r² 上限（超过则剪除其中一个）

        Returns:
            保留的 SNP 位置列表
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
