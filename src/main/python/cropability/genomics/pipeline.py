"""
NGS variant detection pipeline orchestration
===========================================
Native in-process pipeline for mpileup-like summaries and FastCall3-style VCF
calling without shelling out to external binaries.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cropability.genomics.fastcall3 import FastCall3Config, FastCall3Runner
from cropability.genomics.pileup import MpileupParser, NativePileupEngine
from cropability.io.bam import AlignmentInputManager
from cropability.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class QCThresholds:
    min_depth: int = 10
    min_base_quality: int = 20
    min_mapping_quality: int = 20
    min_alt_freq: float = 0.05


@dataclass
class PipelineConfig:
    fastcall3: FastCall3Config = field(default_factory=FastCall3Config)
    prefer_rust_backend: bool = True
    timeout_seconds: int = 3600


class VariantPipeline:
    """Native NGS variant pipeline."""

    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self.fastcall3_runner = FastCall3Runner(self.config.fastcall3)
        self.pileup_engine = NativePileupEngine(prefer_rust_backend=self.config.prefer_rust_backend)

    def _write_pileup_summary(
        self,
        reference: str | Path,
        bam_files: Sequence[str | Path],
        output_path: str | Path,
        qc: QCThresholds,
        regions: str | None = None,
    ) -> dict[str, object]:
        out = Path(output_path)
        records = self.pileup_engine.generate_records(
            reference=reference,
            bam_files=bam_files,
            regions=regions,
            min_base_quality=qc.min_base_quality,
            min_mapping_quality=qc.min_mapping_quality,
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        n_sites = 0
        parser = MpileupParser()
        with out.open("w", encoding="utf-8") as f:
            f.write("#CHROM\tPOS\tREF\tDP\tALT\tAC\tAF\n")
            summaries = parser.summarize_sites(
                records=records,
                min_depth=qc.min_depth,
                min_alt_freq=qc.min_alt_freq,
            )
            for s in summaries:
                f.write(
                    f"{s.chrom}\t{s.pos}\t{s.ref_base}\t{s.depth}\t"
                    f"{s.alt_base or '.'}\t{s.alt_count}\t{s.alt_freq:.6f}\n"
                )
                n_sites += 1
        return {
            "engine": "native",
            "output": str(out),
            "returncode": 0,
            "n_sites": n_sites,
        }

    def run_mpileup(
        self,
        reference: str | Path,
        bam_files: Sequence[str | Path],
        output_path: str | Path,
        qc: QCThresholds,
        regions: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        if dry_run:
            return {
                "engine": "native",
                "command": [
                    "cropability-native-pileup",
                    "--reference",
                    str(reference),
                    "--output",
                    str(output_path),
                ],
                "output": str(output_path),
                "returncode": 0,
            }
        return self._write_pileup_summary(
            reference=reference,
            bam_files=bam_files,
            output_path=output_path,
            qc=qc,
            regions=regions,
        )

    def run_fastcall3(
        self,
        reference: str | Path,
        bam_files: Sequence[str | Path],
        output_vcf: str | Path,
        qc: QCThresholds,
        regions: str | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        result = self.fastcall3_runner.run(
            reference=reference,
            bam_files=bam_files,
            output_vcf=output_vcf,
            regions=regions,
            min_base_quality=qc.min_base_quality,
            min_mapping_quality=qc.min_mapping_quality,
            min_depth=qc.min_depth,
            min_alt_freq=qc.min_alt_freq,
            dry_run=dry_run,
        )
        return {
            "engine": result.backend,
            "command": result.command,
            "output": str(result.output_vcf),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "n_records": result.n_records,
            "elapsed_seconds": result.elapsed_seconds,
        }

    def run(
        self,
        mode: str,
        reference: str | Path,
        bam_files: Sequence[str | Path],
        output: str | Path,
        qc: QCThresholds | None = None,
        regions: str | None = None,
        mpileup_output: str | Path | None = None,
        dry_run: bool = False,
    ) -> dict[str, object]:
        mode = mode.lower()
        if mode not in {"mpileup", "fastcall3", "hybrid"}:
            raise ValueError("mode must be one of: mpileup, fastcall3, hybrid")

        qc = qc or QCThresholds()
        input_manager = AlignmentInputManager(bam_files)
        input_manager.validate_tools(require_samtools=False, require_pysam=not dry_run)
        alignment_files = input_manager.collect(require_index=True)
        bam_paths = [f.path for f in alignment_files]

        start = time.time()
        report: dict[str, object] = {"mode": mode, "reference": str(reference), "engine": "native"}
        if mode == "mpileup":
            report["mpileup"] = self.run_mpileup(
                reference=reference,
                bam_files=bam_paths,
                output_path=output,
                qc=qc,
                regions=regions,
                dry_run=dry_run,
            )
        elif mode == "fastcall3":
            report["fastcall3"] = self.run_fastcall3(
                reference=reference,
                bam_files=bam_paths,
                output_vcf=output,
                qc=qc,
                regions=regions,
                dry_run=dry_run,
            )
        else:
            mpileup_out = (
                Path(mpileup_output)
                if mpileup_output is not None
                else Path(f"{output}.pileup.summary.tsv")
            )
            report["mpileup"] = self.run_mpileup(
                reference=reference,
                bam_files=bam_paths,
                output_path=mpileup_out,
                qc=qc,
                regions=regions,
                dry_run=dry_run,
            )
            report["fastcall3"] = self.run_fastcall3(
                reference=reference,
                bam_files=bam_paths,
                output_vcf=output,
                qc=qc,
                regions=regions,
                dry_run=dry_run,
            )
        report["elapsed_seconds"] = time.time() - start
        return report

