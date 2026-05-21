"""Tests for ``cropability.ngs.io`` (Rust FASTA; pure-Python VCF helpers)."""

from __future__ import annotations

import pytest

from cropability.ngs.io import VCFReader, VCFRecord

# ---------------------------------------------------------------------------
# FASTA — requires maturin-built _core
# ---------------------------------------------------------------------------

@pytest.fixture
def fasta_file(tmp_path):
    p = tmp_path / "test.fa"
    p.write_text(
        ">seq1 description\nACGTACGT\nACGT\n"
        ">seq2\nGCTAGCTA\n"
        ">seq3\nNNNNNNNN\n",
        encoding="utf-8",
    )
    return p


@pytest.mark.native
def test_fasta_read_all(fasta_file, native_core):
    from cropability.ngs.io import FastaReader

    reader = FastaReader(fasta_file)
    data = reader.read_all()
    assert "seq1" in data
    assert data["seq3"] == "NNNNNNNN"


@pytest.mark.native
def test_fasta_file_not_found(tmp_path, native_core):
    from cropability.ngs.io import FastaReader

    with pytest.raises((FileNotFoundError, OSError, ValueError)):
        FastaReader(tmp_path / "nonexistent.fa")


# ---------------------------------------------------------------------------
# VCF — pure Python (no _core)
# ---------------------------------------------------------------------------


class TestVCFRecord:
    def test_is_snp(self):
        rec = VCFRecord("chr1", 100, ".", "A", ["T"], 30.0, [], {})
        assert rec.is_snp is True
        assert rec.is_indel is False

    def test_is_indel(self):
        rec = VCFRecord("chr1", 100, ".", "A", ["ATG"], 20.0, [], {})
        assert rec.is_indel is True

    def test_get_genotypes(self):
        rec = VCFRecord(
            "chr1", 100, ".", "A", ["T"], 30.0, [],
            {}, ["GT"], ["0/1", "1/1", "./."]
        )
        gts = rec.get_genotypes()
        assert gts[0] == (0, 1)
        assert gts[1] == (1, 1)
        assert gts[2] is None

    def test_repr(self):
        rec = VCFRecord("chr1", 100, ".", "A", ["T"], 30.0, [], {})
        assert "chr1" in repr(rec)
        assert "100" in repr(rec)


class TestVCFReader:
    @pytest.fixture
    def vcf_file(self, tmp_path):
        p = tmp_path / "test.vcf"
        p.write_text(
            "##fileformat=VCFv4.2\n"
            "##INFO=<ID=AF,Number=A,Type=Float,Description=\"Allele Freq\">\n"
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample1\tsample2\n"
            "chr1\t100\trs1\tA\tT\t30.0\tPASS\tAF=0.3\tGT\t0/1\t1/1\n"
            "chr1\t200\trs2\tC\tG\t25.0\tPASS\tAF=0.1\tGT\t0/0\t0/1\n"
            "chr1\t300\t.\tAT\tA\t15.0\tPASS\t.\tGT\t0/1\t0/0\n",
            encoding="utf-8",
        )
        return p

    def test_read_vcf(self, vcf_file):
        reader = VCFReader(vcf_file)
        records = list(reader)
        assert len(records) == 3
        assert records[0].chrom == "chr1"
        assert records[0].pos == 100
        assert records[0].is_snp

    def test_sample_names(self, vcf_file):
        reader = VCFReader(vcf_file)
        list(reader)
        assert reader.sample_names == ["sample1", "sample2"]

    def test_snp_filter(self, vcf_file):
        reader = VCFReader(vcf_file)
        snps = [r for r in reader if r.is_snp]
        assert len(snps) == 2

    def test_info_parsing(self, vcf_file):
        reader = VCFReader(vcf_file)
        records = list(reader)
        assert records[0].info.get("AF") == "0.3"

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            VCFReader(tmp_path / "nonexistent.vcf")
