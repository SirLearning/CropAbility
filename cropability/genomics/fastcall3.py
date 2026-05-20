"""
FastCall3 native adapter
========================
Implements in-process variant calling in CropAbility without invoking external
FastCall3 or samtools binaries, with an optional Rust acceleration hook.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cropability.genomics.pileup import NativePileupEngine, PileupRecord
from cropability.io.vcf import VCFRecord, VCFWriter
from cropability.utils.logging import get_logger

logger = get_logger(__name__)


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
    backend: str = "python"
    n_records: int = 0
    elapsed_seconds: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output_vcf.exists()


def _choose_genotype(ref_count: int, alt_count: int, depth: int, min_depth: int) -> str:
    if depth < min_depth:
        return "./."
    if depth == 0:
        return "./."
    af = alt_count / depth
    if af >= 0.8:
        return "1/1"
    if af >= 0.2:
        return "0/1"
    return "0/0"


class FastCall3Runner:
    """Native FastCall3-style variant caller executed fully in CropAbility."""

    def __init__(self, config: FastCall3Config | None = None) -> None:
        self.config = config or FastCall3Config()
        self.pileup_engine = NativePileupEngine(prefer_rust_backend=self.config.prefer_rust_backend)

    def _run_rust_backend(
        self,
        reference: str | Path,
        bam_files: Sequence[str | Path],
        output_vcf: str | Path,
        regions: str | None,
        min_base_quality: int,
        min_mapping_quality: int,
        min_depth: int,
        min_alt_freq: float,
    ) -> FastCall3RunResult | None:
        if not self.config.prefer_rust_backend:
            return None
        backend = self.pileup_engine.try_get_rust_backend()
        if backend is None:
            return None
        logger.info("Using optional Rust backend for native FastCall3 call")
        return backend.run_fastcall3(
            reference=str(reference),
            bam_files=[str(x) for x in bam_files],
            output_vcf=str(output_vcf),
            regions=regions,
            min_base_quality=min_base_quality,
            min_mapping_quality=min_mapping_quality,
            min_depth=min_depth,
            min_alt_freq=min_alt_freq,
        )

    def run(
        self,
        reference: str | Path,
        bam_files: Sequence[str | Path],
        output_vcf: str | Path,
        regions: str | None = None,
        min_base_quality: int | None = None,
        min_mapping_quality: int | None = None,
        min_depth: int | None = None,
        min_alt_freq: float = 0.05,
        extra_args: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> FastCall3RunResult:
        output_path = Path(output_vcf)
        mbq = 20 if min_base_quality is None else min_base_quality
        mmq = 20 if min_mapping_quality is None else min_mapping_quality
        mdp = 10 if min_depth is None else min_depth

        command = [
            "cropability-fastcall3-native",
            "--reference",
            str(reference),
            "--output",
            str(output_path),
            "--min-base-quality",
            str(mbq),
            "--min-mapping-quality",
            str(mmq),
            "--min-depth",
            str(mdp),
            "--min-alt-freq",
            str(min_alt_freq),
        ]
        for bam in bam_files:
            command.extend(["--bam", str(bam)])
        if regions:
            command.extend(["--regions", regions])
        if self.config.extra_args:
            command.extend(self.config.extra_args)
        if extra_args:
            command.extend([str(x) for x in extra_args])

        if dry_run:
            return FastCall3RunResult(
                command=command,
                returncode=0,
                stdout="dry-run",
                stderr="",
                output_vcf=output_path,
                backend="dry-run",
                n_records=0,
                elapsed_seconds=0.0,
            )

        rust_result = self._run_rust_backend(
            reference=reference,
            bam_files=bam_files,
            output_vcf=output_vcf,
            regions=regions,
            min_base_quality=mbq,
            min_mapping_quality=mmq,
            min_depth=mdp,
            min_alt_freq=min_alt_freq,
        )
        if rust_result is not None:
            return rust_result

        start = time.time()
        pileup_records = self.pileup_engine.generate_records(
            reference=reference,
            bam_files=bam_files,
            regions=regions,
            min_base_quality=mbq,
            min_mapping_quality=mmq,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        n_records = 0
        sample_names = [Path(x).stem for x in bam_files]
        meta = [
            '##INFO=<ID=DP,Number=1,Type=Integer,Description="Total Depth">',
            '##INFO=<ID=AF,Number=1,Type=Float,Description="Alt Allele Frequency">',
            '##INFO=<ID=AC,Number=1,Type=Integer,Description="Alt Allele Count">',
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
            '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Read Depth">',
            '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allele depths (ref,alt)">',
        ]

        with VCFWriter(output_path, sample_names=sample_names) as writer:
            writer.write_header(source="CropAbility.NativeFastCall3", extra_meta=meta)
            for rec in pileup_records:
                out = self._call_record(
                    rec=rec,
                    sample_names=sample_names,
                    min_depth=mdp,
                    min_alt_freq=min_alt_freq,
                )
                if out is None:
                    continue
                writer.write_record(out)
                n_records += 1

        elapsed = time.time() - start
        return FastCall3RunResult(
            command=command,
            returncode=0,
            stdout=f"native fastcall3 completed with {n_records} records",
            stderr="",
            output_vcf=output_path,
            backend="python",
            n_records=n_records,
            elapsed_seconds=elapsed,
        )

    def _call_record(
        self,
        rec: PileupRecord,
        sample_names: Sequence[str],
        min_depth: int,
        min_alt_freq: float,
    ) -> VCFRecord | None:
        total_depth = 0
        merged: dict[str, int] = {k: 0 for k in ("A", "C", "G", "T", "N")}
        for sample in rec.samples.values():
            total_depth += sample.depth
            for base, count in sample.base_counts.items():
                merged[base] = merged.get(base, 0) + count

        if total_depth < min_depth:
            return None

        ref = rec.ref_base.upper()
        candidates = {b: c for b, c in merged.items() if b in {"A", "C", "G", "T"} and b != ref}
        if not candidates:
            return None
        alt, alt_count = max(candidates.items(), key=lambda kv: kv[1])
        alt_freq = alt_count / max(1, total_depth)
        if alt_freq < min_alt_freq:
            return None

        sample_values: list[str] = []
        for sample_name in sample_names:
            s = rec.samples[sample_name]
            depth = s.depth
            ref_count = s.base_counts.get(ref, 0)
            sample_alt_count = s.base_counts.get(alt, 0)
            gt = _choose_genotype(ref_count, sample_alt_count, depth, min_depth)
            sample_values.append(f"{gt}:{depth}:{ref_count},{sample_alt_count}")

        return VCFRecord(
            chrom=rec.chrom,
            pos=rec.pos,
            id=".",
            ref=ref,
            alt=[alt],
            qual=None,
            filter=[],
            info={
                "DP": str(total_depth),
                "AF": f"{alt_freq:.6f}",
                "AC": str(alt_count),
            },
            format=["GT", "DP", "AD"],
            samples=sample_values,
        )

