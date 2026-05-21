"""mpileup — CPU logic in Rust (``VariantPipeline`` / ``NativePileupEngine``)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterator, Optional


@dataclass
class PileupSample:
    depth: int
    base_counts: dict[str, int]
    insertions: int = 0
    deletions: int = 0


@dataclass
class PileupRecord:
    chrom: str
    pos: int
    ref_base: str
    samples: dict[str, PileupSample] = field(default_factory=dict)


@dataclass
class PileupSiteSummary:
    chrom: str
    pos: int
    ref_base: str
    depth: int
    alt_base: Optional[str]
    alt_count: int
    alt_freq: float


class MpileupParser:
    """Text mpileup parsing runs in Rust during pipeline export."""

    def __init__(self, sample_names=None) -> None:
        self.sample_names = sample_names

    def summarize_sites(self, records, min_depth: int = 10, min_alt_freq: float = 0.05):
        raise NotImplementedError("Use VariantPipeline.run (Rust) for pileup")


class NativePileupEngine:
    """Delegated to Rust via ``VariantPipeline``."""

    def __init__(self, prefer_rust_backend: bool = True) -> None:
        self.prefer_rust_backend = prefer_rust_backend

    def generate_records(self, *args, **kwargs) -> Iterator[PileupRecord]:
        raise NotImplementedError("Use VariantPipeline (Rust) for pileup generation")
