"""
BAM/CRAM input abstraction
============================
Provides BAM/CRAM file validation, sample naming, and index checks for the NGS
variant detection pipeline.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from cropability.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class AlignmentFile:
    """Metadata for a single alignment input file."""

    path: Path
    sample_name: str
    fmt: str  # bam | cram
    has_index: bool
    index_path: Path | None

    @property
    def is_bam(self) -> bool:
        return self.fmt == "bam"

    @property
    def is_cram(self) -> bool:
        return self.fmt == "cram"


def _detect_format(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".bam":
        return "bam"
    if suffix == ".cram":
        return "cram"
    raise ValueError(f"Unsupported alignment format for: {path} (expected .bam/.cram)")


def _resolve_index(path: Path, fmt: str) -> Path | None:
    if fmt == "bam":
        candidates = [Path(f"{path}.bai"), path.with_suffix(".bai")]
    else:
        candidates = [Path(f"{path}.crai"), path.with_suffix(".crai")]
    for idx in candidates:
        if idx.exists():
            return idx
    return None


class AlignmentInputManager:
    """
    BAM/CRAM input manager.

    Supports:
    - Basic file existence checks
    - BAM/CRAM format detection
    - Index file existence checks
    - samtools / optional pysam dependency checks
    """

    def __init__(self, paths: Sequence[str | Path]) -> None:
        if not paths:
            raise ValueError("At least one alignment file is required")
        self.paths = [Path(p) for p in paths]

    @staticmethod
    def check_samtools() -> bool:
        return shutil.which("samtools") is not None

    @staticmethod
    def check_pysam() -> bool:
        try:
            import pysam  # noqa: F401

            return True
        except Exception:
            return False

    def validate_tools(
        self,
        require_samtools: bool = True,
        require_pysam: bool = False,
    ) -> None:
        if require_samtools and not self.check_samtools():
            raise RuntimeError("samtools is required but not found in PATH")
        if require_pysam and not self.check_pysam():
            raise RuntimeError("pysam is required but not installed")

    def collect(self, require_index: bool = True) -> list[AlignmentFile]:
        files: list[AlignmentFile] = []
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(f"Alignment file not found: {p}")
            fmt = _detect_format(p)
            idx = _resolve_index(p, fmt)
            if require_index and idx is None:
                ext = ".bai" if fmt == "bam" else ".crai"
                raise FileNotFoundError(
                    f"Index not found for {p}. Expected {p}{ext} or sibling index file."
                )
            files.append(
                AlignmentFile(
                    path=p,
                    sample_name=p.stem,
                    fmt=fmt,
                    has_index=idx is not None,
                    index_path=idx,
                )
            )
        logger.info(f"Collected {len(files)} alignment files")
        return files
