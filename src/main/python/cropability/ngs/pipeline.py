"""NGS pipeline — orchestration in Rust (``cropability.native._core``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from cropability.native._core import QCThresholds as _RustQC
from cropability.native._core import VariantPipeline as _RustPipeline


@dataclass
class QCThresholds:
    min_depth: int = 10
    min_base_quality: int = 20
    min_mapping_quality: int = 20
    min_alt_freq: float = 0.05

    def _to_rust(self) -> _RustQC:
        return _RustQC(
            min_depth=self.min_depth,
            min_base_quality=self.min_base_quality,
            min_mapping_quality=self.min_mapping_quality,
            min_alt_freq=self.min_alt_freq,
        )


@dataclass
class PipelineConfig:
    prefer_rust_backend: bool = True
    timeout_seconds: int = 3600


class VariantPipeline:
    def __init__(self, config: PipelineConfig | None = None) -> None:
        self.config = config or PipelineConfig()
        self._inner = _RustPipeline()

    def run(
        self,
        mode: str,
        reference,
        bam_files: Sequence[str],
        output,
        qc: QCThresholds | None = None,
        regions: str | None = None,
        mpileup_output=None,
        dry_run: bool = False,
    ) -> dict:
        qc = qc or QCThresholds()
        return self._inner.run(
            mode=mode,
            reference=str(reference),
            bam_files=[str(x) for x in bam_files],
            output=str(output),
            qc=qc._to_rust(),
            regions=regions,
            mpileup_output=str(mpileup_output) if mpileup_output else None,
            dry_run=dry_run,
        )
