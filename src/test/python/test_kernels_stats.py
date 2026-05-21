"""Tests for statistical compute kernels."""

import pytest
import torch
from cropability.kernels.stats import welford_mean_var, zscore_normalize, pearson_correlation


class TestWelfordMeanVar:
    def test_basic(self):
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
        mean, var = welford_mean_var(x)
        assert abs(float(mean[0]) - 3.0) < 1e-4
        assert abs(float(var[0]) - 2.5) < 1e-4  # sample variance

    def test_batch(self):
        x = torch.randn(10, 100)
        mean, var = welford_mean_var(x)
        assert mean.shape == (10,)
        assert var.shape == (10,)
        # Compare with torch built-ins
        ref_mean = x.mean(dim=1)
        ref_var = x.var(dim=1)
        assert torch.allclose(mean, ref_mean, atol=1e-4)
        assert torch.allclose(var, ref_var, atol=1e-4)

    def test_single_element(self):
        x = torch.tensor([[5.0]])
        mean, var = welford_mean_var(x)
        assert abs(float(mean[0]) - 5.0) < 1e-5
        # Single-element variance is 0 (unbiased estimate N/A; should not crash)

    def test_constant_series(self):
        x = torch.ones(3, 50)
        mean, var = welford_mean_var(x)
        assert torch.allclose(mean, torch.ones(3), atol=1e-5)
        assert torch.allclose(var, torch.zeros(3), atol=1e-5)


class TestZscoreNormalize:
    def test_basic(self):
        x = torch.tensor([[1.0, 2.0, 3.0]])
        z = zscore_normalize(x)
        assert abs(float(z.mean()) < 1e-5)
        assert abs(float(z.std()) - 1.0) < 0.1  # approximately 1

    def test_already_normalized(self):
        x = torch.randn(5, 100)
        z = zscore_normalize(x)
        assert z.shape == x.shape
        row_means = z.mean(dim=1)
        row_stds = z.std(dim=1)
        assert torch.allclose(row_means, torch.zeros(5), atol=1e-4)

    def test_constant_row(self):
        x = torch.tensor([[3.0, 3.0, 3.0, 3.0]])
        z = zscore_normalize(x)
        # Constant row: std ≈ 0, z-score should be 0
        assert torch.allclose(z, torch.zeros_like(z), atol=1e-4)

    def test_shape_preserved(self):
        x = torch.randn(8, 200)
        z = zscore_normalize(x)
        assert z.shape == x.shape


class TestPearsonCorrelation:
    def test_perfect_correlation(self):
        x = torch.arange(10, dtype=torch.float32).unsqueeze(0)  # [1, 10]
        y = x * 2.0
        r = pearson_correlation(x, y)
        assert r.shape == (1, 1)
        assert abs(float(r[0, 0]) - 1.0) < 1e-3

    def test_perfect_anticorrelation(self):
        x = torch.arange(10, dtype=torch.float32).unsqueeze(0)
        y = -x
        r = pearson_correlation(x, y)
        assert abs(float(r[0, 0]) + 1.0) < 1e-3

    def test_matrix_shape(self):
        x = torch.randn(5, 50)
        y = torch.randn(8, 50)
        r = pearson_correlation(x, y)
        assert r.shape == (5, 8)

    def test_range(self):
        x = torch.randn(10, 100)
        y = torch.randn(10, 100)
        r = pearson_correlation(x, y)
        assert (r >= -1.01).all() and (r <= 1.01).all()
