"""
mpileup parsing and native generation
====================================
Provides parser helpers for text mpileup and in-process pileup generation from
alignment files using pysam, with an optional Rust acceleration hook.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from cropability.utils.logging import get_logger

logger = get_logger(__name__)

_BASES = ("A", "C", "G", "T", "N")


@dataclass
class PileupSample:
    """Single-sample per-site pileup summary."""

    depth: int
    base_counts: dict[str, int]
    insertions: int = 0
    deletions: int = 0

    @property
    def alt_count(self) -> int:
        return sum(v for k, v in self.base_counts.items() if k in {"A", "C", "G", "T"})


@dataclass
class PileupRecord:
    """Single genomic site pileup summary across one or more samples."""

    chrom: str
    pos: int
    ref_base: str
    samples: dict[str, PileupSample] = field(default_factory=dict)

    def total_depth(self) -> int:
        return sum(s.depth for s in self.samples.values())


@dataclass
class PileupSiteSummary:
    """Merged cross-sample site summary for threshold-based filtering."""

    chrom: str
    pos: int
    ref_base: str
    depth: int
    alt_base: str | None
    alt_count: int
    alt_freq: float


def _parse_pileup_bases(bases: str, ref_base: str) -> tuple[dict[str, int], int, int]:
    counts = {b: 0 for b in _BASES}
    insertions = 0
    deletions = 0
    i = 0
    ref_u = ref_base.upper()

    while i < len(bases):
        c = bases[i]
        if c == "^":
            i += 2
            continue
        if c == "$":
            i += 1
            continue
        if c in "+-":
            sign = c
            i += 1
            nbuf = []
            while i < len(bases) and bases[i].isdigit():
                nbuf.append(bases[i])
                i += 1
            length = int("".join(nbuf)) if nbuf else 0
            if sign == "+":
                insertions += 1
            else:
                deletions += 1
            i += length
            continue
        if c == "*":
            deletions += 1
            i += 1
            continue
        if c in ".,":  # matches reference
            if ref_u in counts:
                counts[ref_u] += 1
            else:
                counts["N"] += 1
            i += 1
            continue

        b = c.upper()
        if b in counts:
            counts[b] += 1
        else:
            counts["N"] += 1
        i += 1

    return counts, insertions, deletions


class MpileupParser:
    """
    Parser for text-based mpileup output lines.

    Input format:
      CHROM POS REF [DP BASES QUAL]...
    where each sample occupies three columns.
    """

    def __init__(self, sample_names: Sequence[str] | None = None) -> None:
        self.sample_names = list(sample_names) if sample_names is not None else None

    def parse_line(self, line: str) -> PileupRecord | None:
        line = line.rstrip("\n")
        if not line:
            return None
        cols = line.split("\t")
        if len(cols) < 6:
            return None
        if (len(cols) - 3) % 3 != 0:
            return None

        chrom, pos_s, ref_base = cols[0], cols[1], cols[2]
        per_sample = cols[3:]
        n_samples = len(per_sample) // 3
        if self.sample_names is None:
            names = [f"sample{i + 1}" for i in range(n_samples)]
        else:
            if len(self.sample_names) != n_samples:
                raise ValueError(
                    f"sample_names count ({len(self.sample_names)}) != mpileup samples ({n_samples})"
                )
            names = list(self.sample_names)

        samples: dict[str, PileupSample] = {}
        for i, name in enumerate(names):
            depth = int(per_sample[i * 3])
            bases = per_sample[i * 3 + 1]
            counts, ins, dels = _parse_pileup_bases(bases, ref_base)
            samples[name] = PileupSample(
                depth=depth,
                base_counts=counts,
                insertions=ins,
                deletions=dels,
            )

        return PileupRecord(
            chrom=chrom,
            pos=int(pos_s),
            ref_base=ref_base.upper(),
            samples=samples,
        )

    def parse_file(self, path: str | Path) -> Iterator[PileupRecord]:
        p = Path(path)
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                rec = self.parse_line(line)
                if rec is not None:
                    yield rec

    def summarize_sites(
        self,
        records: Iterator[PileupRecord],
        min_depth: int = 10,
        min_alt_freq: float = 0.05,
    ) -> list[PileupSiteSummary]:
        summaries: list[PileupSiteSummary] = []
        for rec in records:
            merged_counts = {b: 0 for b in _BASES}
            depth = 0
            for sample in rec.samples.values():
                depth += sample.depth
                for b, c in sample.base_counts.items():
                    merged_counts[b] = merged_counts.get(b, 0) + c

            if depth < min_depth:
                continue
            ref = rec.ref_base
            alt_candidates = {
                b: c
                for b, c in merged_counts.items()
                if b in {"A", "C", "G", "T"} and b != ref
            }
            if not alt_candidates:
                continue
            alt_base, alt_count = max(alt_candidates.items(), key=lambda kv: kv[1])
            alt_freq = alt_count / max(depth, 1)
            if alt_freq < min_alt_freq:
                continue
            summaries.append(
                PileupSiteSummary(
                    chrom=rec.chrom,
                    pos=rec.pos,
                    ref_base=ref,
                    depth=depth,
                    alt_base=alt_base,
                    alt_count=alt_count,
                    alt_freq=alt_freq,
                )
            )

        logger.info("Generated %d pileup site summaries", len(summaries))
        return summaries


class NativePileupEngine:
    """In-process pileup engine replacing external `samtools mpileup` calls."""

    def __init__(self, prefer_rust_backend: bool = True) -> None:
        self.prefer_rust_backend = prefer_rust_backend

    def try_get_rust_backend(self):
        if not self.prefer_rust_backend:
            return None
        try:
            # Optional extension point. This module is intentionally optional.
            from cropability.native import ngs_core  # type: ignore

            return ngs_core
        except Exception:
            return None

    def _require_pysam(self):
        try:
            import pysam  # noqa: PLC0415

            return pysam
        except Exception as e:  # pragma: no cover - import path dependent
            raise RuntimeError(
                "pysam is required for native pileup. Install optional dependency: cropability[io]"
            ) from e

    def _parse_region(self, region: str | None) -> tuple[str | None, int | None, int | None]:
        if not region:
            return None, None, None
        if ":" not in region:
            return region, None, None
        chrom, span = region.split(":", 1)
        if "-" not in span:
            return chrom, None, None
        start_s, end_s = span.split("-", 1)
        start = max(0, int(start_s.replace(",", "")) - 1)
        end = int(end_s.replace(",", ""))
        return chrom, start, end

    def _count_sample_column(
        self,
        column,
        ref_base: str,
        min_base_quality: int,
        min_mapping_quality: int,
    ) -> PileupSample:
        counts = {b: 0 for b in _BASES}
        insertions = 0
        deletions = 0
        depth = 0
        ref_base_u = ref_base.upper()
        for pr in column.pileups:
            aln = pr.alignment
            if aln.mapping_quality < min_mapping_quality:
                continue
            if pr.is_refskip:
                continue
            if pr.is_del:
                deletions += 1
                continue
            qpos = pr.query_position
            if qpos is None:
                continue
            if qpos >= len(aln.query_qualities):
                continue
            if aln.query_qualities[qpos] < min_base_quality:
                continue
            depth += 1
            if pr.indel > 0:
                insertions += 1
            elif pr.indel < 0:
                deletions += 1

            base = aln.query_sequence[qpos].upper()
            if base not in counts:
                base = "N"
            if base == ref_base_u:
                counts[ref_base_u] += 1
            else:
                counts[base] += 1
        return PileupSample(depth=depth, base_counts=counts, insertions=insertions, deletions=deletions)

    def generate_records(
        self,
        reference: str | Path,
        bam_files: Sequence[str | Path],
        regions: str | None = None,
        min_base_quality: int = 20,
        min_mapping_quality: int = 20,
    ) -> Iterator[PileupRecord]:
        rust_backend = self.try_get_rust_backend()
        if rust_backend is not None and hasattr(rust_backend, "generate_pileup_records"):
            logger.info("Using optional Rust backend for pileup generation")
            yield from rust_backend.generate_pileup_records(
                reference=str(reference),
                bam_files=[str(x) for x in bam_files],
                regions=regions,
                min_base_quality=min_base_quality,
                min_mapping_quality=min_mapping_quality,
            )
            return

        pysam = self._require_pysam()
        sample_names = [Path(x).stem for x in bam_files]
        chrom_filter, start, end = self._parse_region(regions)
        ref = pysam.FastaFile(str(reference))
        bam_handles = [pysam.AlignmentFile(str(p), "rc" if str(p).endswith(".cram") else "rb") for p in bam_files]
        try:
            site_table: dict[tuple[str, int], dict[str, PileupSample]] = {}
            ref_table: dict[tuple[str, int], str] = {}
            for sample_name, bam in zip(sample_names, bam_handles):
                pileup_iter = bam.pileup(
                    contig=chrom_filter,
                    start=start,
                    stop=end,
                    truncate=chrom_filter is not None,
                    min_base_quality=0,  # handled by custom thresholds
                    stepper="all",
                )
                for col in pileup_iter:
                    chrom = col.reference_name
                    pos1 = col.pos + 1
                    key = (chrom, pos1)
                    if key not in ref_table:
                        ref_base = ref.fetch(chrom, col.pos, col.pos + 1).upper()
                        ref_table[key] = ref_base if ref_base else "N"
                    sample_pileup = self._count_sample_column(
                        column=col,
                        ref_base=ref_table[key],
                        min_base_quality=min_base_quality,
                        min_mapping_quality=min_mapping_quality,
                    )
                    site_table.setdefault(key, {})[sample_name] = sample_pileup

            for (chrom, pos1), sample_data in sorted(site_table.items()):
                for sample_name in sample_names:
                    if sample_name not in sample_data:
                        sample_data[sample_name] = PileupSample(
                            depth=0,
                            base_counts={b: 0 for b in _BASES},
                            insertions=0,
                            deletions=0,
                        )
                yield PileupRecord(
                    chrom=chrom,
                    pos=pos1,
                    ref_base=ref_table[(chrom, pos1)],
                    samples=sample_data,
                )
        finally:
            ref.close()
            for h in bam_handles:
                h.close()
