"""Tests for sequence processing kernels."""

import pytest
import torch
from cropability.kernels.seq import (
    encode_sequences,
    gc_content_kernel,
    reverse_complement_kernel,
    kmer_count_kernel,
)


class TestEncodeSequences:
    def test_basic_encoding(self):
        seqs = ["ACGT", "NNNN"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        assert enc.shape == (2, 4)
        assert enc.dtype == torch.int8
        # A=0, C=1, G=2, T=3
        assert list(enc[0].numpy()) == [0, 1, 2, 3]
        assert list(enc[1].numpy()) == [4, 4, 4, 4]

    def test_lowercase(self):
        seqs = ["acgt"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        assert list(enc[0].numpy()) == [0, 1, 2, 3]

    def test_padding(self):
        seqs = ["ACG", "ACGT"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        assert enc.shape == (2, 4)
        assert enc[0, 3].item() == 4  # padded with N

    def test_empty_list(self):
        enc = encode_sequences([], device=torch.device("cpu"))
        assert enc.shape[0] == 0

    def test_rna_u_maps_to_t(self):
        seqs = ["AUCG"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        assert enc[0, 1].item() == 3  # U → T(3)


class TestGCContent:
    def test_pure_gc(self):
        seqs = ["GCGCGCGC"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        gc = gc_content_kernel(enc)
        assert abs(float(gc[0]) - 1.0) < 1e-5

    def test_pure_at(self):
        seqs = ["ATATATATAT"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        gc = gc_content_kernel(enc)
        assert abs(float(gc[0]) - 0.0) < 1e-5

    def test_mixed(self):
        seqs = ["ACGT"]  # 2 GC / 4 total
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        gc = gc_content_kernel(enc)
        assert abs(float(gc[0]) - 0.5) < 1e-5

    def test_with_n(self):
        seqs = ["ACGN"]  # 2 GC / 3 valid
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        gc = gc_content_kernel(enc)
        assert abs(float(gc[0]) - 2 / 3) < 1e-4

    def test_batch(self):
        seqs = ["GGGG", "AAAA", "ATAT"]  # GGGG=100% GC, AAAA=0% GC, ATAT=0% GC
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        gc = gc_content_kernel(enc)
        assert gc.shape == (3,)
        assert abs(float(gc[0]) - 1.0) < 1e-5
        assert abs(float(gc[1]) - 0.0) < 1e-5
        assert abs(float(gc[2]) - 0.0) < 1e-5

    def test_half_gc(self):
        seqs = ["ATGC"]  # A=0, T=0, G=GC, C=GC → 50%
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        gc = gc_content_kernel(enc)
        assert abs(float(gc[0]) - 0.5) < 1e-5


class TestReverseComplement:
    def test_basic(self):
        seqs = ["ATCG"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        rc = reverse_complement_kernel(enc)
        # ATCG → rev_comp = CGAT
        # A(0)→T(3), T(3)→A(0), C(1)→G(2), G(2)→C(1)
        # original: A T C G → comp: T A G C → rev: C G A T
        decoded = [int(x) for x in rc[0].tolist()]
        assert decoded == [1, 2, 0, 3]  # C G A T

    def test_palindrome(self):
        seqs = ["ATAT"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        rc = reverse_complement_kernel(enc)
        # A T A T → comp T A T A → rev A T A T
        assert list(rc[0].numpy()) == list(enc[0].numpy())

    def test_shape_preserved(self):
        seqs = ["ACGTACGT", "GGCCGGCC"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        rc = reverse_complement_kernel(enc)
        assert rc.shape == enc.shape


class TestKmerCount:
    def test_basic_4mer(self):
        seqs = ["ACGT"]  # 1 unique 4-mer: ACGT
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        counts = kmer_count_kernel(enc, k=4)
        assert counts.shape == (1, 4 ** 4)
        assert abs(counts.sum().item() - 1.0) < 1e-4  # normalized sum is 1

    def test_n_excluded(self):
        seqs = ["ACGN"]  # N makes 4-mer invalid
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        counts = kmer_count_kernel(enc, k=4)
        assert counts.sum().item() == 0.0

    def test_shape(self):
        seqs = ["ACGTACGT", "GCTAGCTA"]
        enc = encode_sequences(seqs, device=torch.device("cpu"))
        counts = kmer_count_kernel(enc, k=3)
        assert counts.shape == (2, 4 ** 3)
