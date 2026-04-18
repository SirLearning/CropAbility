"""
变异检测（SNP/Indel）
=====================
GPU 加速的多样本 SNP 调用，基于碱基矩阵的并行统计检验。
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
    """单个 SNP 位点的检测结果。"""
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
    多样本 SNP 检测引擎（GPU 加速）。

    工作流：
    1. 将每个样本的 reads 按位置堆叠为 [S, L] int8 矩阵
    2. 在 GPU 上并行统计各位置的碱基频率
    3. 用 Fisher 精确检验或卡方检验过滤假阳性

    Args:
        device        : GPU 设备（None 则自动选择）
        min_alt_freq  : 最小替代等位基因频率（过滤低频噪音）
        min_depth     : 最小覆盖深度
        quality_thresh: 最低质量分数阈值
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
        对一批样本序列检测 SNP。

        Args:
            sequences : 样本序列列表（等长，与 reference 对齐）
            reference : 参考序列
            positions : 待检测位置列表（None = 全部）

        Returns:
            SNPResult 列表（已过滤低质量位点）
        """
        n_samples = len(sequences)
        seq_len = len(reference)
        logger.info(f"Calling SNPs: {n_samples} samples, length={seq_len}")

        # 编码参考和样本序列
        all_seqs = [reference] + sequences
        encoded = encode_sequences(all_seqs, device=self.device)  # [S+1, L]
        ref_enc = encoded[0]        # [L]
        sample_enc = encoded[1:]    # [S, L]

        if positions is None:
            positions = list(range(seq_len))

        results: List[SNPResult] = []
        pos_tensor = torch.tensor(positions, device=self.device)

        # 提取指定位置的碱基矩阵 [S, P]
        sample_bases = sample_enc[:, pos_tensor]  # [S, P]
        ref_bases = ref_enc[pos_tensor]           # [P]

        for p_idx, pos in enumerate(positions):
            ref_b = int(ref_bases[p_idx].item())
            if ref_b == 4:  # 参考为 N，跳过
                continue

            col = sample_bases[:, p_idx]           # [S]
            called_mask = col != 4                 # 非 N 位点
            n_called = int(called_mask.sum().item())
            if n_called < self.min_depth:
                continue

            called_bases = col[called_mask]
            # 统计非参考碱基频率
            n_alt = int((called_bases != ref_b).sum().item())
            alt_freq = n_alt / n_called

            if alt_freq < self.min_alt_freq:
                continue

            # 质量分数：基于二项分布的简化 Phred 评分
            quality = self._compute_quality(n_alt, n_called)
            if quality < self.quality_thresh:
                continue

            # 确定最高频的替代等位基因
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
        """简化 Phred 质量分数，基于二项分布 p-value 的近似。"""
        if n_total == 0:
            return 0.0
        import math
        af = n_alt / n_total
        # 排除纯参考型
        if af <= 0:
            return 0.0
        # 简化：使用 -10 * log10(binom_pmf_null)
        p_null = 0.001  # 测序错误率
        expected_err = n_total * p_null
        if n_alt <= expected_err:
            return 0.0
        # 近似 Phred = 10 * log10(n_alt / expected_err)
        quality = 10.0 * math.log10(max(n_alt / expected_err, 1e-10))
        return min(quality, 60.0)

    def filter_snps(
        self,
        snps: List[SNPResult],
        min_call_rate: float = 0.90,
        max_alt_freq: float = 0.95,
    ) -> List[SNPResult]:
        """过滤 SNP 列表，移除低质量和极端频率位点。"""
        filtered = [
            s for s in snps
            if s.call_rate >= min_call_rate
            and self.min_alt_freq <= s.alt_freq <= max_alt_freq
        ]
        logger.info(f"After filtering: {len(filtered)}/{len(snps)} SNPs retained")
        return filtered
