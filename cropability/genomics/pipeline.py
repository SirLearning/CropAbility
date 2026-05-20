"""
NGS 变异检测流程编排
====================
统一管理 mpileup / FastCall3 / hybrid 三种调用模式。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union
import shutil
import subprocess
import time

from cropability.genomics.fastcall3 import FastCall3Config, FastCall3Runner
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
    samtools_executable: str = "samtools"
    bcftools_executable: str = "bcftools"
    fastcall3: FastCall3Config = field(default_factory=FastCall3Config)
    timeout_seconds: int = 3600


class VariantPipeline:
    """NGS 变异检测流程。"""

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self.fastcall3_runner = FastCall3Runner(self.config.fastcall3)

    def _resolve_executable(self, exe: str) -> str:
        resolved = shutil.which(exe)
        if resolved is None:
            raise RuntimeError(f"Executable not found in PATH: {exe}")
        return resolved

    def run_mpileup(
        self,
        reference: Union[str, Path],
        bam_files: Sequence[Union[str, Path]],
        output_path: Union[str, Path],
        qc: QCThresholds,
        regions: Optional[str] = None,
        extra_args: Optional[Sequence[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, object]:
        samtools = self._resolve_executable(self.config.samtools_executable)
        out = Path(output_path)

        cmd: List[str] = [
            samtools,
            "mpileup",
            "-f",
            str(reference),
            "-q",
            str(qc.min_mapping_quality),
            "-Q",
            str(qc.min_base_quality),
        ]
        if regions:
            cmd.extend(["-r", regions])
        if extra_args:
            cmd.extend([str(x) for x in extra_args])
        cmd.extend([str(b) for b in bam_files])

        logger.info("Running mpileup: %s", " ".join(cmd))
        if dry_run:
            return {"command": cmd, "output": str(out), "returncode": 0}

        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            proc = subprocess.run(
                cmd,
                stdout=f,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.config.timeout_seconds,
                check=False,
            )
        if proc.returncode != 0:
            raise RuntimeError(
                f"samtools mpileup failed with exit code {proc.returncode}: {proc.stderr.strip()}"
            )
        return {"command": cmd, "output": str(out), "returncode": proc.returncode}

    def run_fastcall3(
        self,
        reference: Union[str, Path],
        bam_files: Sequence[Union[str, Path]],
        output_vcf: Union[str, Path],
        qc: QCThresholds,
        regions: Optional[str] = None,
        extra_args: Optional[Sequence[str]] = None,
        dry_run: bool = False,
    ) -> Dict[str, object]:
        result = self.fastcall3_runner.run(
            reference=reference,
            bam_files=bam_files,
            output_vcf=output_vcf,
            regions=regions,
            min_base_quality=qc.min_base_quality,
            min_mapping_quality=qc.min_mapping_quality,
            min_depth=qc.min_depth,
            extra_args=extra_args,
            dry_run=dry_run,
        )
        return {
            "command": result.command,
            "output": str(result.output_vcf),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    def run(
        self,
        mode: str,
        reference: Union[str, Path],
        bam_files: Sequence[Union[str, Path]],
        output: Union[str, Path],
        qc: Optional[QCThresholds] = None,
        regions: Optional[str] = None,
        mpileup_output: Optional[Union[str, Path]] = None,
        dry_run: bool = False,
    ) -> Dict[str, object]:
        mode = mode.lower()
        if mode not in {"mpileup", "fastcall3", "hybrid"}:
            raise ValueError("mode must be one of: mpileup, fastcall3, hybrid")

        qc = qc or QCThresholds()
        AlignmentInputManager(bam_files).validate_tools(require_samtools=(mode in {"mpileup", "hybrid"}))
        alignment_files = AlignmentInputManager(bam_files).collect(require_index=True)
        bam_paths = [f.path for f in alignment_files]

        start = time.time()
        report: Dict[str, object] = {"mode": mode, "reference": str(reference)}
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
            mpileup_out = Path(mpileup_output) if mpileup_output is not None else Path(f"{output}.mpileup")
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
