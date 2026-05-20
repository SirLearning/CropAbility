"""测试 NGS pipeline 新增模块。"""

from pathlib import Path

import pytest

from cropability.genomics.fastcall3 import FastCall3Config, FastCall3Runner
from cropability.genomics.pileup import MpileupParser
from cropability.genomics.pipeline import QCThresholds, VariantPipeline
from cropability.io.bam import AlignmentInputManager


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
    def test_build_and_dry_run(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda x: f"/usr/bin/{x}")
        out = tmp_path / "out.vcf"
        runner = FastCall3Runner(FastCall3Config(executable="FastCall3"))
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
        assert "-r" in result.command
        assert "-o" in result.command
        assert "--min-base-qual" in result.command


class TestVariantPipeline:
    def test_run_hybrid_dry_run(self, monkeypatch, tmp_path):
        monkeypatch.setattr("shutil.which", lambda x: f"/usr/bin/{x}")
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
