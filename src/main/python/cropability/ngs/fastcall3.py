"""FastCall3 — CPU variant calling in Rust."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from cropability.ngs.pipeline import QCThresholds, VariantPipeline


@dataclass
class FastCall3Config:
    timeout_seconds: int = 3600
    prefer_rust_backend: bool = True
    extra_args: list[str] = field(default_factory=list)


@dataclass
class FastCall3RunResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    output_vcf: Path
    backend: str = "rust"
    n_records: int = 0
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output_vcf.exists()


class FastCall3Runner:
    def __init__(self, config: FastCall3Config | None = None) -> None:
        self.config = config or FastCall3Config()
        self._pipeline = VariantPipeline()

    def run(self, reference, bam_files, output_vcf, **kwargs) -> FastCall3RunResult:
        qc = QCThresholds(
            min_depth=kwargs.get("min_depth", 10),
            min_base_quality=kwargs.get("min_base_quality", 20),
            min_mapping_quality=kwargs.get("min_mapping_quality", 20),
            min_alt_freq=kwargs.get("min_alt_freq", 0.05),
        )
        report = self._pipeline.run(
            mode="fastcall3",
            reference=str(reference),
            bam_files=[str(x) for x in bam_files],
            output=str(output_vcf),
            qc=qc,
            regions=kwargs.get("regions"),
            dry_run=kwargs.get("dry_run", False),
        )
        fc = report.get("fastcall3", report)
        return FastCall3RunResult(
            command=[],
            returncode=fc.get("returncode", 0),
            stdout="",
            stderr="",
            output_vcf=Path(fc.get("output_vcf", output_vcf)),
            backend=fc.get("backend", "rust"),
            n_records=int(fc.get("n_records", 0)),
            elapsed_seconds=float(fc.get("elapsed_seconds", 0.0)),
        )
