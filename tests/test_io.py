"""测试 I/O 模块。"""

import gzip
import tempfile
from pathlib import Path

import pytest
from cropability.io.fasta import FastaReader, FastaWriter
from cropability.io.vcf import VCFReader, VCFWriter, VCFRecord


class TestFastaReader:
    @pytest.fixture
    def fasta_file(self, tmp_path):
        p = tmp_path / "test.fa"
        p.write_text(
            ">seq1 description\nACGTACGT\nACGT\n"
            ">seq2\nGCTAGCTA\n"
            ">seq3\nNNNNNNNN\n",
            encoding="utf-8",
        )
        return p

    @pytest.fixture
    def fasta_gz_file(self, tmp_path):
        p = tmp_path / "test.fa.gz"
        with gzip.open(p, "wt") as f:
            f.write(">seq1\nACGT\n>seq2\nGCTA\n")
        return p

    def test_read_fasta(self, fasta_file):
        reader = FastaReader(fasta_file)
        seqs = list(reader)
        assert len(seqs) == 3
        assert seqs[0][0] == "seq1"
        assert seqs[0][1] == "ACGTACGTACGT"
        assert seqs[1][0] == "seq2"

    def test_read_all(self, fasta_file):
        reader = FastaReader(fasta_file)
        d = reader.read_all()
        assert "seq1" in d
        assert "seq3" in d
        assert d["seq3"] == "NNNNNNNN"

    def test_gzip_support(self, fasta_gz_file):
        reader = FastaReader(fasta_gz_file)
        seqs = list(reader)
        assert len(seqs) == 2
        assert seqs[0][1] == "ACGT"

    def test_chunked_sequences(self, fasta_file):
        reader = FastaReader(fasta_file, chunk_size=4)
        for name, chunks in reader.chunked_sequences(chunk_size=4):
            for c in chunks:
                assert len(c) <= 4

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            FastaReader(tmp_path / "nonexistent.fa")


class TestFastaWriter:
    def test_write_and_read(self, tmp_path):
        path = tmp_path / "out.fa"
        with FastaWriter(path, line_width=4) as writer:
            writer.write("seq1", "ACGTACGT")
            writer.write("seq2", "GCTA")

        reader = FastaReader(path)
        seqs = reader.read_all()
        assert seqs["seq1"] == "ACGTACGT"
        assert seqs["seq2"] == "GCTA"

    def test_line_wrap(self, tmp_path):
        path = tmp_path / "wrapped.fa"
        with FastaWriter(path, line_width=4) as writer:
            writer.write("s", "ACGTACGT")
        lines = path.read_text().splitlines()
        assert len(lines) == 3  # header + 2 lines of 4


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
            "chr1\t300\t.\tAT\tA\t15.0\tPASS\t.\tGT\t0/1\t0/0\n",  # indel
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
