"""
Likelihood computation for genotype selection (vectorized PyTorch with optional Triton kernels).

Inputs: X of shape (N, 3) containing non-negative counts per locus for three categories.
Outputs: Y of shape (N, 6) with the following column order:
  [Homo(0), Hetero(0,1), Hetero(0,2), Homo(1), Hetero(1,2), Homo(2)]
Each entry is the negative log10-likelihood: log10(multinomial_coeff) + per-model term.
"""

import logging
import math
from typing import Optional

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    logging.getLogger("triton").setLevel(logging.WARNING)
except Exception:  # pragma: no cover - optional dependency
    triton = None  # type: ignore
    tl = None  # type: ignore
    TRITON_AVAILABLE = False


# Error model parameters
errorRate = 0.05
same_1 = -math.log10(1 - 0.75 * errorRate)
same_2 = -math.log10(0.25 * errorRate)
diff_1 = -math.log10(0.5 - 0.25 * errorRate)
diff_2 = -math.log10(0.25 * errorRate)


def _ensure_float_tensor(x: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    if not isinstance(x, torch.Tensor):
        raise TypeError("X must be a torch.Tensor")
    if x.dim() != 2 or x.size(-1) != 3:
        raise AssertionError("X must have shape (N, 3)")
    if device is not None and x.device != device:
        x = x.to(device)
    return x.to(dtype=torch.float32)


def select_genotype_from_likelihoods(Y: torch.Tensor):
    """Select the most likely genotype per row from negative log10-likelihoods.

    Args:
        Y: Tensor of shape (N,6), smaller values mean higher likelihood.
    Returns:
        best_idx: LongTensor (N,), index in [0..5]
        best_vals: Tensor (N,), the minimal negative log10-likelihood per row
        margin: Tensor (N,), difference between second-best and best (>=0)
    """
    if not isinstance(Y, torch.Tensor):
        raise TypeError("Y must be a torch.Tensor")
    if Y.dim() != 2 or Y.size(-1) != 6:
        raise AssertionError("Y must have shape (N, 6)")
    # Best index and value
    best_vals, best_idx = torch.min(Y, dim=1)
    # Margin between best and 2nd best for confidence estimation
    sorted_vals, _ = torch.sort(Y, dim=1)
    margin = sorted_vals[:, 1] - sorted_vals[:, 0]
    return best_idx, best_vals, margin


def log10_multinomial_coeff(X: torch.Tensor) -> torch.Tensor:
    """Compute log10 multinomial coefficient for each row: log10(n!/(a! b! c!)).

    Args:
        X: (N, 3) non-negative counts
    Returns:
        (N,) tensor of log10 coefficients
    """
    X = _ensure_float_tensor(X)
    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    n = a + b + c
    # log(n!) = lgamma(n+1)
    ln_coeff = torch.lgamma(n + 1.0) - (torch.lgamma(a + 1.0) + torch.lgamma(b + 1.0) + torch.lgamma(c + 1.0))
    return ln_coeff / math.log(10.0)


def same_likelihood_terms(X: torch.Tensor) -> torch.Tensor:
    """Per-row same-allele terms (N, 3):
      s0 = same_1*a + same_2*(b+c)
      s1 = same_1*b + same_2*(a+c)
      s2 = same_1*c + same_2*(a+b)
    """
    X = _ensure_float_tensor(X)
    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    s0 = same_1 * a + same_2 * (b + c)
    s1 = same_1 * b + same_2 * (a + c)
    s2 = same_1 * c + same_2 * (a + b)
    return torch.stack([s0, s1, s2], dim=1)  # (N,3)


def diff_likelihood_terms(X: torch.Tensor) -> torch.Tensor:
    """Per-row different-allele (hetero) terms (N, 3) in order:
      d01 = diff_1*(a+b) + diff_2*c
      d02 = diff_1*(a+c) + diff_2*b
      d12 = diff_1*(b+c) + diff_2*a
    """
    X = _ensure_float_tensor(X)
    a, b, c = X[:, 0], X[:, 1], X[:, 2]
    d01 = diff_1 * (a + b) + diff_2 * c
    d02 = diff_1 * (a + c) + diff_2 * b
    d12 = diff_1 * (b + c) + diff_2 * a
    return torch.stack([d01, d02, d12], dim=1)  # (N,3)


# Note: If selecting the best genotype is desired, use select_genotype_from_likelihoods(Y)


def gtp_gpu(
    X1: torch.Tensor,
    X2: torch.Tensor,
    X3: torch.Tensor,
    prefer_cuda: bool = True,
    out_device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Compute negative log10-likelihoods for 6 genotype models per row on GPU when available.

    Args:
        X: Input counts, shape (N,3), non-negative.
        prefer_cuda: If True and CUDA is available, compute on CUDA regardless of X's current device.
        out_device: Force output to this device (e.g., torch.device('cuda')); when None, uses CUDA if available and
                    `prefer_cuda` is True, else uses X.device.

    Returns:
        Tensor of shape (N,6) with column order:
        [Homo(0), Hetero(0,1), Hetero(0,2), Homo(1), Hetero(1,2), Homo(2)].
    """
    # Select device
    X = torch.stack([X1, X2, X3], dim=1)
    
    if out_device is not None:
        target_dev = out_device
    elif prefer_cuda and torch.cuda.is_available():
        target_dev = torch.device('cuda')
    else:
        target_dev = X.device

    X = _ensure_float_tensor(X, device=target_dev)
    # Basic validation: non-negative counts
    if torch.any(X < 0):
        raise AssertionError("X must be non-negative counts")
    coeff = log10_multinomial_coeff(X)            # (N,)
    same_terms = same_likelihood_terms(X)         # (N,3)
    diff_terms = diff_likelihood_terms(X)         # (N,3)

    # Broadcast coeff to (N,1) and add
    Y_same = coeff.unsqueeze(1) + same_terms      # (N,3)
    Y_diff = coeff.unsqueeze(1) + diff_terms      # (N,3)

    # Assemble in required order
    Y0 = Y_same[:, 0]
    Y1 = Y_diff[:, 0]
    Y2 = Y_diff[:, 1]
    Y3 = Y_same[:, 1]
    Y4 = Y_diff[:, 2]
    Y5 = Y_same[:, 2]
    Y = torch.stack([Y0, Y1, Y2, Y3, Y4, Y5], dim=1)  # (N,6) on target_dev

    # Add best genotype index as last column
    best_idx, best_vals, margin = select_genotype_from_likelihoods(Y)
    Y = torch.round(Y)
    
    Y_out = torch.cat([Y, best_idx.unsqueeze(1)], dim=1)  # (N,7)
    return Y_out


# Note: A Triton implementation can be added later if needed; current version uses PyTorch vectorization.


__all__ = [
    "gtp_gpu",
    "log10_multinomial_coeff",
    "same_likelihood_terms",
    "diff_likelihood_terms",
    "select_genotype_from_likelihoods",
]



