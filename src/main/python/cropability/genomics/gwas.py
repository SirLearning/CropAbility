"""
Genome-wide association study (GWAS) helpers
==========================================
GPU-accelerated linear/logistic regression association tests (million-SNP scale).
Provides a foundation for linear models and mixed-model covariance correction.
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
    """Single-locus GWAS association result."""
    position: int
    beta: float              # Effect size
    se: float                # Standard error
    t_stat: float            # t statistic
    p_value: float           # p-value
    neg_log10_p: float = field(init=False)

    def __post_init__(self) -> None:
        self.neg_log10_p = -math.log10(max(self.p_value, 1e-300))

    def is_significant(self, threshold: float = 5e-8) -> bool:
        """Return whether genome-wide significance is reached (default 5×10⁻⁸)."""
        return self.p_value < threshold

    def __repr__(self) -> str:
        return (
            f"GWAS(pos={self.position}, β={self.beta:.4f}, "
            f"p={self.p_value:.2e}, -log10p={self.neg_log10_p:.2f})"
        )


class GWASEngine:
    """
    GPU-accelerated GWAS engine supporting:
    - Simple linear regression (OLS)
    - Logistic regression (binary traits)
    - Principal component covariate correction (PC regression)

    Args:
        device    : GPU device
        n_pcs     : Number of leading PCs used for correction
        chunk_size: SNP batch size (avoids OOM)
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
        Run linear regression GWAS.

        Args:
            genotypes  : [N, P] float32 sample×SNP genotype matrix (0/1/2)
            phenotype  : [N] float32 quantitative trait
            covariates : [N, C] float32 covariate matrix (PCs, sex, etc.)
            positions  : SNP position list

        Returns:
            List of GWASResult (sorted by position)
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

        # Build covariate matrix (intercept + user-provided covariates)
        intercept = torch.ones(n_samples, 1, device=self.device)
        if covariates is not None:
            cov = covariates.float().to(self.device)
            Z = torch.cat([intercept, cov], dim=1)  # [N, C+1]
        else:
            Z = intercept

        # Residualize phenotype (remove covariate effects)
        pheno_resid = self._residualize(pheno, Z)   # [N, 1]
        # Residualize genotypes
        with Timer("geno_residualize"):
            geno_resid = self._residualize_batch(geno, Z, self.chunk_size)  # [N, P]

        # Batch linear regression
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
        """Y = Y - Z(Z^TZ)^-1 Z^T Y (projection residual)."""
        ZtZ = Z.T @ Z
        ZtZ_inv = torch.linalg.pinv(ZtZ)
        proj = Z @ ZtZ_inv @ Z.T
        return Y - proj @ Y

    def _residualize_batch(
        self, X: torch.Tensor, Z: torch.Tensor, chunk_size: int
    ) -> torch.Tensor:
        """Batch-residualize each column of the genotype matrix."""
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
        Run simple linear regression of y against each column of [N, P],
        using vectorized batched least squares.
        """
        n, p = X.shape
        y = y.float()
        X = X.float()

        # beta = (X^T y) / (X^T X) (already residualized; intercept is 0)
        Xty = (X * y.unsqueeze(1)).sum(dim=0)  # [P]
        XtX = (X ** 2).sum(dim=0)              # [P]
        XtX_safe = XtX.clamp(min=1e-12)

        beta = Xty / XtX_safe                  # [P]

        # Compute residuals and standard errors
        y_hat = X * beta.unsqueeze(0)          # [N, P]
        resid = y.unsqueeze(1) - y_hat         # [N, P]
        rss = (resid ** 2).sum(dim=0)          # [P]
        df = max(n - 2, 1)
        mse = rss / df
        se = (mse / XtX_safe).clamp(min=1e-20).sqrt()  # [P]

        t_stat = beta / se.clamp(min=1e-20)    # [P]

        # t-distribution p-value approximation (normal approx; sample size usually large)
        # p = 2 * Φ(-|t|), approximated via erfc
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
        """Convert t statistic to two-tailed p-value (normal approximation)."""
        # Normal approximation (error is small for large n > 30)
        abs_t = t.abs()
        # erf approximation: p ≈ erfc(|t| / sqrt(2))
        p = torch.erfc(abs_t / math.sqrt(2))
        return p.clamp(min=1e-300, max=1.0)

    def compute_pca(
        self,
        genotypes: torch.Tensor,
        n_components: int = 10,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run PCA on the genotype matrix (for stratification correction).

        Args:
            genotypes   : [N, P] genotype matrix
            n_components: Number of principal components to retain

        Returns:
            (scores, loadings) — scores: [N, n_components],
                                  loadings: [P, n_components]
        """
        geno = genotypes.float().to(self.device)
        # Standardize
        mean = geno.mean(dim=0, keepdim=True)
        std = geno.std(dim=0, keepdim=True).clamp(min=1e-8)
        geno_norm = (geno - mean) / std

        logger.info(f"PCA: {geno.shape[1]} SNPs → {n_components} PCs")
        with Timer("svd") as t:
            # Truncated SVD
            U, S, Vt = torch.linalg.svd(geno_norm, full_matrices=False)

        logger.info(f"SVD completed in {t.elapsed_ms:.1f} ms")
        scores = U[:, :n_components] * S[:n_components].unsqueeze(0)
        loadings = Vt[:n_components].T
        return scores, loadings
