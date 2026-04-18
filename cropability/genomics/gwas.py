"""
全基因组关联分析（GWAS）辅助工具
==================================
GPU 加速的线性/逻辑回归关联检验（百万 SNP 级别）。
提供线性模型、混合模型协方差校正的基础框架。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import math
import torch
import numpy as np

from cropability.utils.logging import get_logger
from cropability.utils.timer import Timer

logger = get_logger(__name__)


@dataclass
class GWASResult:
    """GWAS 单位点关联结果。"""
    position: int
    beta: float              # 效应量
    se: float                # 标准误
    t_stat: float            # t 统计量
    p_value: float           # p 值
    neg_log10_p: float = field(init=False)

    def __post_init__(self) -> None:
        self.neg_log10_p = -math.log10(max(self.p_value, 1e-300))

    def is_significant(self, threshold: float = 5e-8) -> bool:
        """是否达到全基因组显著性水平（默认 5×10⁻⁸）。"""
        return self.p_value < threshold

    def __repr__(self) -> str:
        return (
            f"GWAS(pos={self.position}, β={self.beta:.4f}, "
            f"p={self.p_value:.2e}, -log10p={self.neg_log10_p:.2f})"
        )


class GWASEngine:
    """
    GPU 加速 GWAS 引擎，支持：
    - 简单线性回归（OLS）
    - 逻辑回归（二元性状）
    - 主成分协变量校正（PC回归）

    Args:
        device    : GPU 设备
        n_pcs     : 用于校正的前 n 个主成分数
        chunk_size: 批量 SNP 块大小（避免显存溢出）
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        n_pcs: int = 10,
        chunk_size: int = 10_000,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_pcs = n_pcs
        self.chunk_size = chunk_size
        logger.info(f"GWASEngine on {device}, n_pcs={n_pcs}")

    def run_linear_gwas(
        self,
        genotypes: torch.Tensor,
        phenotype: torch.Tensor,
        covariates: Optional[torch.Tensor] = None,
        positions: Optional[List[int]] = None,
    ) -> List[GWASResult]:
        """
        执行线性回归 GWAS。

        Args:
            genotypes  : [N, P] float32，样本×SNP 基因型矩阵（0/1/2）
            phenotype  : [N] float32 数量性状
            covariates : [N, C] float32 协变量矩阵（包括 PC、性别等）
            positions  : SNP 位置列表

        Returns:
            GWASResult 列表（按位置排序）
        """
        n_samples, n_snps = genotypes.shape
        if positions is None:
            positions = list(range(n_snps))

        logger.info(
            f"Linear GWAS: {n_snps} SNPs, {n_samples} samples, "
            f"covariates={'yes' if covariates is not None else 'no'}"
        )

        geno = genotypes.float().to(self.device)
        pheno = phenotype.float().to(self.device).unsqueeze(1)  # [N, 1]

        # 构建协变量矩阵（截距 + 用户提供的协变量）
        intercept = torch.ones(n_samples, 1, device=self.device)
        if covariates is not None:
            cov = covariates.float().to(self.device)
            Z = torch.cat([intercept, cov], dim=1)  # [N, C+1]
        else:
            Z = intercept

        # 残差化表型（去除协变量效应）
        pheno_resid = self._residualize(pheno, Z)   # [N, 1]
        # 残差化基因型
        with Timer("geno_residualize"):
            geno_resid = self._residualize_batch(geno, Z, self.chunk_size)  # [N, P]

        # 批量线性回归
        with Timer("regression") as t:
            results = self._batch_linear_regression(
                geno_resid, pheno_resid.squeeze(1), positions
            )

        logger.info(
            f"GWAS complete in {t.elapsed_ms:.1f} ms, "
            f"{sum(1 for r in results if r.is_significant())} genome-wide significant hits"
        )
        return results

    def _residualize(
        self, Y: torch.Tensor, Z: torch.Tensor
    ) -> torch.Tensor:
        """Y = Y - Z(Z^TZ)^-1 Z^T Y（投影残差）。"""
        ZtZ = Z.T @ Z
        ZtZ_inv = torch.linalg.pinv(ZtZ)
        proj = Z @ ZtZ_inv @ Z.T
        return Y - proj @ Y

    def _residualize_batch(
        self, X: torch.Tensor, Z: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        """对基因型矩阵各列批量残差化。"""
        n, p = X.shape
        out = torch.empty_like(X)
        ZtZ_inv = torch.linalg.pinv(Z.T @ Z)
        proj = Z @ ZtZ_inv @ Z.T  # [N, N]
        for start in range(0, p, chunk_size):
            end = min(start + chunk_size, p)
            chunk = X[:, start:end]
            out[:, start:end] = chunk - proj @ chunk
        return out

    def _batch_linear_regression(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        positions: List[int],
    ) -> List[GWASResult]:
        """
        对 [N, P] 矩阵的每列对 y 做简单线性回归，
        利用 batched least squares 向量化实现。
        """
        n, p = X.shape
        y = y.float()
        X = X.float()

        # beta = (X^T y) / (X^T X)（已残差化，截距为0）
        Xty = (X * y.unsqueeze(1)).sum(dim=0)  # [P]
        XtX = (X ** 2).sum(dim=0)              # [P]
        XtX_safe = XtX.clamp(min=1e-12)

        beta = Xty / XtX_safe                  # [P]

        # 计算残差和标准误
        y_hat = X * beta.unsqueeze(0)          # [N, P]
        resid = y.unsqueeze(1) - y_hat         # [N, P]
        rss = (resid ** 2).sum(dim=0)          # [P]
        df = max(n - 2, 1)
        mse = rss / df
        se = (mse / XtX_safe).clamp(min=1e-20).sqrt()  # [P]

        t_stat = beta / se.clamp(min=1e-20)    # [P]

        # t 分布 p 值近似（使用正态近似，样本量通常足够大）
        # p = 2 * Φ(-|t|)，用 erfc 近似
        p_values = self._t_to_p(t_stat, df=df)  # [P]

        results = []
        beta_cpu = beta.cpu().numpy()
        se_cpu = se.cpu().numpy()
        t_cpu = t_stat.cpu().numpy()
        p_cpu = p_values.cpu().numpy()

        for i, pos in enumerate(positions):
            results.append(GWASResult(
                position=pos,
                beta=float(beta_cpu[i]),
                se=float(se_cpu[i]),
                t_stat=float(t_cpu[i]),
                p_value=float(p_cpu[i]),
            ))

        return results

    @staticmethod
    def _t_to_p(t: torch.Tensor, df: int) -> torch.Tensor:
        """t 统计量转双尾 p 值（正态近似）。"""
        # 使用正态分布近似（大样本 n>30 时误差很小）
        abs_t = t.abs()
        # erf 近似: p ≈ erfc(|t| / sqrt(2))
        p = torch.erfc(abs_t / math.sqrt(2))
        return p.clamp(min=1e-300, max=1.0)

    def compute_pca(
        self,
        genotypes: torch.Tensor,
        n_components: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        对基因型矩阵进行 PCA（用于 stratification 校正）。

        Args:
            genotypes   : [N, P] 基因型矩阵
            n_components: 保留的主成分数

        Returns:
            (scores, loadings) — scores: [N, n_components],
                                  loadings: [P, n_components]
        """
        geno = genotypes.float().to(self.device)
        # 标准化
        mean = geno.mean(dim=0, keepdim=True)
        std = geno.std(dim=0, keepdim=True).clamp(min=1e-8)
        geno_norm = (geno - mean) / std

        logger.info(f"PCA: {geno.shape[1]} SNPs → {n_components} PCs")
        with Timer("svd") as t:
            # 使用截断 SVD
            U, S, Vt = torch.linalg.svd(geno_norm, full_matrices=False)

        logger.info(f"SVD completed in {t.elapsed_ms:.1f} ms")
        scores = U[:, :n_components] * S[:n_components].unsqueeze(0)
        loadings = Vt[:n_components].T
        return scores, loadings
