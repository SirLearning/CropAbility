"""
mpileup 解析与标准化
===================
提供对 samtools mpileup 文本输出的解析、计数统计与位点摘要能力。
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
    """单样本位点统计。"""

    depth: int
    base_counts: dict[str, int]
    insertions: int = 0
    deletions: int = 0

    @property
    def alt_count(self) -> int:
        return sum(v for k, v in self.base_counts.items() if k in {"A", "C", "G", "T"})


@dataclass
class PileupRecord:
    """单个位点记录。"""

    chrom: str
    pos: int
    ref_base: str
    samples: dict[str, PileupSample] = field(default_factory=dict)

    def total_depth(self) -> int:
        return sum(s.depth for s in self.samples.values())


@dataclass
class PileupSiteSummary:
    """跨样本位点汇总。"""

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
        if c in ".,":  # 与参考一致
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
    mpileup 文本解析器。

    mpileup 格式：
      CHROM POS REF [DP BASES QUAL]...
    每个样本占 3 列。
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
            alt_candidates = {b: c for b, c in merged_counts.items() if b in {"A", "C", "G", "T"} and b != ref}
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

        logger.info(f"Generated {len(summaries)} pileup site summaries")
        return summaries
