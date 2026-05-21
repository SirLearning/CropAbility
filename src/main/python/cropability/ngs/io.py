"""Genomic file I/O — Rust via ``cropability.native._core``; VCF types for parsing."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union

from cropability.native._core import FastaReader as _RustFastaReader


# ---------------------------------------------------------------------------
# FASTA (Rust)
# ---------------------------------------------------------------------------

class FastaReader:
    """FASTA reader backed by Rust ``FastaReader``."""

    def __init__(self, path: Union[str, Path]) -> None:
        self._inner = _RustFastaReader(str(path))

    def read_all(self) -> Dict[str, str]:
        return dict(self._inner.read_all())


# ---------------------------------------------------------------------------
# BAM / CRAM (validation; pipeline runs in Rust)
# ---------------------------------------------------------------------------

@dataclass
class AlignmentFile:
    path: Path
    sample_name: str
    fmt: str
    has_index: bool

    @property
    def is_bam(self) -> bool:
        return self.fmt == "bam"


class AlignmentInputManager:
    def __init__(self, paths: Sequence[str | Path]) -> None:
        if not paths:
            raise ValueError("At least one alignment file is required")
        self.paths = [Path(p) for p in paths]

    def collect(self, require_index: bool = True) -> list[AlignmentFile]:
        for p in self.paths:
            if not p.exists():
                raise FileNotFoundError(p)
            fmt = "bam" if p.suffix.lower() == ".bam" else "cram" if p.suffix.lower() == ".cram" else None
            if fmt is None:
                raise ValueError(f"Unsupported alignment format: {p}")
            idx = p.with_suffix(".bai" if fmt == "bam" else ".crai")
            has_index = idx.exists()
            if require_index and not has_index:
                raise FileNotFoundError(f"Missing index: {idx}")
        return [
            AlignmentFile(
                path=p,
                sample_name=p.stem,
                fmt="bam" if p.suffix.lower() == ".bam" else "cram",
                has_index=True,
            )
            for p in self.paths
        ]


# ---------------------------------------------------------------------------
# VCF (lightweight Python types; production output from Rust pipeline)
# ---------------------------------------------------------------------------

@dataclass
class VCFRecord:
    chrom: str
    pos: int
    id: str = "."
    ref: str = ""
    alt: List[str] = field(default_factory=list)
    qual: Optional[float] = None
    filter: List[str] = field(default_factory=list)
    info: Dict[str, str] = field(default_factory=dict)
    format: List[str] = field(default_factory=list)
    samples: List[str] = field(default_factory=list)

    @property
    def is_snp(self) -> bool:
        return len(self.ref) == 1 and len(self.alt) == 1 and len(self.alt[0]) == 1

    @property
    def is_indel(self) -> bool:
        return not self.is_snp

    def get_genotypes(self) -> List[Optional[Tuple[int, int]]]:
        out: List[Optional[Tuple[int, int]]] = []
        for gt in self.samples:
            if gt in (".", "./."):
                out.append(None)
                continue
            parts = gt.replace("|", "/").split("/")
            if len(parts) != 2:
                out.append(None)
            else:
                try:
                    out.append((int(parts[0]), int(parts[1])))
                except ValueError:
                    out.append(None)
        return out

    def __repr__(self) -> str:
        return f"VCFRecord({self.chrom}:{self.pos} {self.ref}>{self.alt})"


class VCFReader:
    """Minimal VCF reader for tests and small files."""

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.sample_names: List[str] = []

    def __iter__(self) -> Iterator[VCFRecord]:
        with self.path.open(encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("##"):
                    continue
                if line.startswith("#CHROM"):
                    cols = line.strip().split("\t")
                    self.sample_names = cols[9:] if len(cols) > 9 else []
                    continue
                if not line.strip() or line.startswith("#"):
                    continue
                cols = line.strip().split("\t")
                if len(cols) < 8:
                    continue
                fmt_samples = cols[9:] if len(cols) > 9 else []
                yield VCFRecord(
                    chrom=cols[0],
                    pos=int(cols[1]),
                    id=cols[2],
                    ref=cols[3],
                    alt=cols[4].split(",") if cols[4] != "." else [],
                    qual=float(cols[5]) if cols[5] != "." else None,
                    filter=cols[6].split(";") if cols[6] != "." else [],
                    info=_parse_info(cols[7]),
                    format=cols[8].split(":") if len(cols) > 8 else [],
                    samples=fmt_samples,
                )


def _parse_info(field: str) -> Dict[str, str]:
    if field in (".", ""):
        return {}
    out: Dict[str, str] = {}
    for part in field.split(";"):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k] = v
        else:
            out[part] = "1"
    return out


class VCFWriter:
    """Placeholder — production VCF output uses Rust ``VariantPipeline``."""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError("Use VariantPipeline (Rust) for VCF output")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def write_header(self, *args, **kwargs) -> None:
        pass

    def write_record(self, *args, **kwargs) -> None:
        pass
