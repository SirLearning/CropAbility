"""测试 NGS pipeline 新增模块。"""


import pytest

from cropability.cli.main import build_parser
from cropability.genomics.fastcall3 import FastCall3Config, FastCall3Runner
from cropability.genomics.pileup import MpileupParser, PileupRecord, PileupSample
from cropability.genomics.pipeline import QCThresholds, VariantPipeline
from cropability.io.bam import AlignmentInputManager
from cropability.io.vcf import VCFReader


class TestAlignmentInputManager:
    def test_collect_bam_with_index(self, tmp_path):
        bam = tmp_path / "sample1.bam"
        bai = tmp_path / "sample1.bam.bai"
        bam.write_bytes(b"")
        bai.write_bytes(b"")
        mgr = AlignmentInputManager([bam])
        files = mgr.collect(require_index=True)
        assert len(files) == 1
        assert files[0].is_bam
        assert files[0].has_index

    def test_missing_index_raises(self, tmp_path):
        bam = tmp_path / "sample1.bam"
        bam.write_bytes(b"")
        mgr = AlignmentInputManager([bam])
        with pytest.raises(FileNotFoundError):
            mgr.collect(require_index=True)


class TestMpileupParser:
    def test_parse_line_and_summary(self):
        parser = MpileupParser(sample_names=["s1"])
        # chr1  10  A  depth=5  bases=".,TtA" quals dummy
        rec = parser.parse_line("chr1\t10\tA\t5\t.,TtA\tFFFFF")
        assert rec is not None
        assert rec.chrom == "chr1"
        assert rec.pos == 10
        assert rec.samples["s1"].depth == 5
        summaries = parser.summarize_sites(iter([rec]), min_depth=1, min_alt_freq=0.1)
        assert len(summaries) == 1
        assert summaries[0].alt_base in {"T", "C", "G", "A"}


class TestFastCall3Runner:
    def test_build_and_dry_run(self, tmp_path):
        out = tmp_path / "out.vcf"
        runner = FastCall3Runner(FastCall3Config())
        result = runner.run(
            reference="ref.fa",
            bam_files=["a.bam", "b.bam"],
            output_vcf=out,
            regions="chr1:1-100",
            min_base_quality=20,
            min_mapping_quality=30,
            min_depth=10,
            dry_run=True,
        )
        assert result.returncode == 0
        assert result.backend == "dry-run"
        assert "--reference" in result.command
        assert "--output" in result.command

    def test_native_run_writes_vcf(self, monkeypatch, tmp_path):
        out = tmp_path / "out.vcf"
        runner = FastCall3Runner(FastCall3Config(prefer_rust_backend=False))

        fake_records = [
            PileupRecord(
                chrom="chr1",
                pos=42,
                ref_base="A",
                samples={
                    "a": PileupSample(depth=10, base_counts={"A": 7, "C": 3, "G": 0, "T": 0, "N": 0}),
                    "b": PileupSample(depth=8, base_counts={"A": 3, "C": 5, "G": 0, "T": 0, "N": 0}),
                },
            )
        ]
        monkeypatch.setattr(runner.pileup_engine, "generate_records", lambda **_: iter(fake_records))

        res = runner.run(
            reference="ref.fa",
            bam_files=["a.bam", "b.bam"],
            output_vcf=out,
            min_depth=1,
            min_alt_freq=0.1,
            dry_run=False,
        )
        assert res.ok
        assert res.n_records == 1

        records = list(VCFReader(out))
        assert len(records) == 1
        assert records[0].chrom == "chr1"
        assert records[0].pos == 42
        assert records[0].ref == "A"
        assert records[0].alt == ["C"]


class TestVariantPipeline:
    def test_run_hybrid_dry_run(self, tmp_path):
        bam = tmp_path / "cohort.bam"
        bai = tmp_path / "cohort.bam.bai"
        bam.write_bytes(b"")
        bai.write_bytes(b"")

        pipeline = VariantPipeline()
        report = pipeline.run(
            mode="hybrid",
            reference="ref.fa",
            bam_files=[bam],
            output=tmp_path / "cohort.vcf",
            qc=QCThresholds(),
            dry_run=True,
        )
        assert report["mode"] == "hybrid"
        assert "mpileup" in report
        assert "fastcall3" in report
        assert report["engine"] == "native"
        assert report["fastcall3"]["engine"] == "dry-run"


class TestCliExtensions:
    def test_new_subcommands_exist(self):
        parser = build_parser()
        ns = parser.parse_args(
            [
                "call-variants",
                "-r",
                "ref.fa",
                "-b",
                "a.bam",
                "-o",
                "out.vcf",
                "--mode",
                "hybrid",
                "--dry-run",
            ]
        )
        assert ns.command == "call-variants"
        assert ns.mode == "hybrid"
