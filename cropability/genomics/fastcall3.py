"""
FastCall3 外部调用适配层
========================
将 FastCall3 作为外部程序纳入 CropAbility 流水线，负责参数映射与执行。
"""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cropability.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FastCall3Config:
    executable: str = "FastCall3"
    timeout_seconds: int = 3600
    extra_args: list[str] = field(default_factory=list)


@dataclass
class FastCall3RunResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    output_vcf: Path

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.output_vcf.exists()


class FastCall3Runner:
    """FastCall3 适配执行器。"""

    def __init__(self, config: FastCall3Config | None = None) -> None:
        self.config = config or FastCall3Config()

    def validate_executable(self) -> str:
        resolved = shutil.which(self.config.executable)
        if resolved is None:
            p = Path(self.config.executable)
            if not p.exists():
                raise RuntimeError(
                    f"FastCall3 executable not found: {self.config.executable}"
                )
            resolved = str(p)
        return resolved

    def build_command(
        self,
        reference: str | Path,
        bam_files: Sequence[str | Path],
        output_vcf: str | Path,
        regions: str | None = None,
        min_base_quality: int | None = None,
        min_mapping_quality: int | None = None,
        min_depth: int | None = None,
        extra_args: Sequence[str] | None = None,
    ) -> list[str]:
        exe = self.validate_executable()
        cmd: list[str] = [exe, "-r", str(reference), "-o", str(output_vcf)]
        for bam in bam_files:
            cmd.extend(["-i", str(bam)])
        if regions:
            cmd.extend(["-R", regions])
        if min_base_quality is not None:
            cmd.extend(["--min-base-qual", str(min_base_quality)])
        if min_mapping_quality is not None:
            cmd.extend(["--min-mapq", str(min_mapping_quality)])
        if min_depth is not None:
            cmd.extend(["--min-depth", str(min_depth)])
        cmd.extend(self.config.extra_args)
        if extra_args:
            cmd.extend([str(x) for x in extra_args])
        return cmd

    def run(
        self,
        reference: str | Path,
        bam_files: Sequence[str | Path],
        output_vcf: str | Path,
        regions: str | None = None,
        min_base_quality: int | None = None,
        min_mapping_quality: int | None = None,
        min_depth: int | None = None,
        extra_args: Sequence[str] | None = None,
        dry_run: bool = False,
    ) -> FastCall3RunResult:
        output_path = Path(output_vcf)
        cmd = self.build_command(
            reference=reference,
            bam_files=bam_files,
            output_vcf=output_path,
            regions=regions,
            min_base_quality=min_base_quality,
            min_mapping_quality=min_mapping_quality,
            min_depth=min_depth,
            extra_args=extra_args,
        )
        logger.info("Running FastCall3: %s", " ".join(cmd))

        if dry_run:
            return FastCall3RunResult(
                command=cmd,
                returncode=0,
                stdout="",
                stderr="",
                output_vcf=output_path,
            )

        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=self.config.timeout_seconds,
            check=False,
        )

        result = FastCall3RunResult(
            command=cmd,
            returncode=proc.returncode,
            stdout=proc.stdout,
            stderr=proc.stderr,
            output_vcf=output_path,
        )

        if proc.returncode != 0:
            raise RuntimeError(
                f"FastCall3 failed with exit code {proc.returncode}: {proc.stderr.strip()}"
            )
        if not output_path.exists():
            raise RuntimeError(
                f"FastCall3 completed but output VCF not found: {output_path}"
            )
        return result
