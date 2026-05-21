"""Tests for ``cropability.ngs`` pipeline and BAM input (Rust ``_core``)."""

import pytest

pytest.importorskip("cropability.native._core")
pytestmark = pytest.mark.native

from cropability.ngs.io import AlignmentInputManager
from cropability.ngs.pipeline import QCThresholds, VariantPipeline


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


class TestVariantPipeline:
    def test_run_hybrid_dry_run(self, tmp_path, native_core):
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
        assert report["engine"] == "rust"
        assert report["mpileup"]["engine"] == "rust"
        assert report["fastcall3"]["engine"] == "rust"
