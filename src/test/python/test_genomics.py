"""Tests for genomics analysis modules."""

import pytest
import torch
from cropability.genomics.variant import VariantCaller, SNPResult
from cropability.genomics.ld import LDCalculator, LDResult
from cropability.genomics.gwas import GWASEngine, GWASResult
from cropability.genomics.alignment import SmithWatermanGPU


class TestVariantCaller:
    @pytest.fixture
    def caller(self):
        return VariantCaller(
            device=torch.device("cpu"),
            min_alt_freq=0.05,
            min_depth=2,
            quality_thresh=5.0,
        )

    def test_call_snps_basic(self, caller):
        reference = "ATCGATCG"
        samples = [
            "ATCGATCG",  # same as reference
            "ATCGATCG",
            "TTCGATCG",  # position 0: A → T
            "TTCGATCG",
        ]
        snps = caller.call_snps(samples, reference)
        assert isinstance(snps, list)
        if snps:
            for snp in snps:
                assert isinstance(snp, SNPResult)
                assert 0 <= snp.alt_freq <= 1

    def test_no_variants(self, caller):
        reference = "ACGTACGT"
        samples = ["ACGTACGT"] * 5
        snps = caller.call_snps(samples, reference)
        assert len(snps) == 0

    def test_filter_snps(self, caller):
        snps = [
            SNPResult(0, "A", "T", 0.5, 30.0, 10, 10),
            SNPResult(1, "C", "G", 0.01, 5.0, 5, 10),  # frequency too low
            SNPResult(2, "G", "A", 0.3, 15.0, 9, 10),
        ]
        filtered = caller.filter_snps(snps, min_call_rate=0.8)
        assert all(s.alt_freq >= caller.min_alt_freq for s in filtered)
        assert all(s.call_rate >= 0.8 for s in filtered)

    def test_snp_result_properties(self):
        snp = SNPResult(100, "A", "T", 0.3, 25.0, 90, 100)
        assert snp.call_rate == 0.9
        assert "SNP" in repr(snp)
        assert snp.position == 100


class TestLDCalculator:
    @pytest.fixture
    def calc(self):
        return LDCalculator(device=torch.device("cpu"), chunk_size=50)

    def test_basic_ld(self, calc):
        n_samples, n_snps = 50, 10
        genotypes = torch.randint(0, 3, (n_samples, n_snps)).float()
        result = calc.compute_ld_matrix(genotypes)
        assert isinstance(result, LDResult)
        assert result.r2_matrix.shape == (n_snps, n_snps)
        assert result.n_snps == n_snps
        assert result.n_samples == n_samples

    def test_r2_diagonal_is_one(self, calc):
        genotypes = torch.randn(100, 5).abs()
        result = calc.compute_ld_matrix(genotypes)
        # Diagonal (self-correlation) should be 1
        diag = result.r2_matrix.diagonal()
        assert torch.allclose(diag, torch.ones(5), atol=0.01)

    def test_r2_range(self, calc):
        genotypes = torch.randint(0, 3, (100, 20)).float()
        result = calc.compute_ld_matrix(genotypes)
        assert (result.r2_matrix >= -0.01).all()
        assert (result.r2_matrix <= 1.01).all()

    def test_high_ld_pairs(self, calc):
        genotypes = torch.randint(0, 3, (100, 10)).float()
        result = calc.compute_ld_matrix(genotypes)
        pairs = result.high_ld_pairs(threshold=0.5)
        assert isinstance(pairs, list)
        for i, j, r2 in pairs:
            assert r2 > 0.5

    def test_ld_pruning(self, calc):
        genotypes = torch.randint(0, 3, (100, 20)).float()
        result = calc.compute_ld_matrix(genotypes)
        retained = calc.prune_by_ld(result.r2_matrix, list(range(20)), threshold=0.8)
        assert len(retained) <= 20
        assert len(retained) > 0

    def test_chunked_matches_direct(self, calc):
        genotypes = torch.randint(0, 3, (100, 5)).float()
        result_direct = calc.compute_ld_matrix(genotypes)
        calc_chunked = LDCalculator(device=torch.device("cpu"), chunk_size=2)
        result_chunked = calc_chunked.compute_ld_matrix(genotypes)
        assert torch.allclose(result_direct.r2_matrix, result_chunked.r2_matrix, atol=1e-4)


class TestGWASEngine:
    @pytest.fixture
    def engine(self):
        return GWASEngine(device=torch.device("cpu"), n_pcs=5, chunk_size=50)

    def test_linear_gwas(self, engine):
        n_samples, n_snps = 100, 20
        genotypes = torch.randint(0, 3, (n_samples, n_snps)).float()
        phenotype = torch.randn(n_samples)
        results = engine.run_linear_gwas(genotypes, phenotype)
        assert len(results) == n_snps
        for r in results:
            assert isinstance(r, GWASResult)
            assert 0 < r.p_value <= 1
            assert r.se >= 0
            assert r.neg_log10_p >= 0

    def test_gwas_with_covariates(self, engine):
        n_samples, n_snps = 100, 10
        genotypes = torch.randint(0, 3, (n_samples, n_snps)).float()
        phenotype = torch.randn(n_samples)
        covariates = torch.randn(n_samples, 3)
        results = engine.run_linear_gwas(genotypes, phenotype, covariates=covariates)
        assert len(results) == n_snps

    def test_pca(self, engine):
        genotypes = torch.randint(0, 3, (200, 100)).float()
        scores, loadings = engine.compute_pca(genotypes, n_components=5)
        assert scores.shape == (200, 5)
        assert loadings.shape == (100, 5)

    def test_significant_detection(self):
        r = GWASResult(0, 1.5, 0.1, 15.0, 1e-10)
        assert r.is_significant()
        r2 = GWASResult(1, 0.1, 0.1, 1.0, 0.3)
        assert not r2.is_significant()

    def test_neg_log10_p(self):
        r = GWASResult(0, 1.0, 0.1, 10.0, 1e-5)
        assert abs(r.neg_log10_p - 5.0) < 0.01


class TestSmithWaterman:
    @pytest.fixture
    def sw(self):
        return SmithWatermanGPU(device=torch.device("cpu"))

    def test_identical_sequences(self, sw):
        queries = ["ACGT"]
        targets = ["ACGT"]
        scores = sw.score_matrix(queries, targets)
        assert scores.shape == (1, 1)
        assert float(scores[0, 0]) > 0

    def test_different_sequences(self, sw):
        queries = ["AAAA"]
        targets = ["TTTT"]
        scores = sw.score_matrix(queries, targets)
        # No match; score should be 0 (Smith-Waterman floor)
        assert float(scores[0, 0]) >= 0

    def test_batch_shape(self, sw):
        queries = ["ACGT", "GCTA", "TTTT"]
        targets = ["ACGT", "NNGCTA", "AAAA", "CCCC"]
        scores = sw.score_matrix(queries, targets)
        assert scores.shape == (3, 4)

    def test_find_top_hits(self, sw):
        queries = ["ACGTACGT"]
        targets = ["ACGTACGT", "TTTTTTTT", "ACGT", "GGGGGGGG"]
        scores = sw.score_matrix(queries, targets)
        hits = sw.find_top_hits(scores, top_k=2, threshold=1.0)
        assert len(hits) <= 2
        if hits:
            q, t, s = hits[0]
            assert s >= 1.0
