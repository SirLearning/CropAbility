"""
VCF 文件读写
============
轻量级 VCF v4.x 解析器，无 htslib 依赖，纯 Python 实现。
"""

from __future__ import annotations

import gzip
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple, Union

from cropability.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class VCFRecord:
    """单条 VCF 变异记录。"""
    chrom: str
    pos: int         # 1-based
    id: str
    ref: str
    alt: List[str]
    qual: Optional[float]
    filter: List[str]
    info: Dict[str, str]
    format: List[str] = field(default_factory=list)
    samples: List[str] = field(default_factory=list)

    @property
    def is_snp(self) -> bool:
        return len(self.ref) == 1 and all(len(a) == 1 for a in self.alt)

    @property
    def is_indel(self) -> bool:
        return not self.is_snp

    @property
    def alt_str(self) -> str:
        return ",".join(self.alt)

    def get_genotypes(self) -> List[Optional[Tuple[int, int]]]:
        """
        解析所有样本的基因型，返回 (allele1, allele2) 或 None（缺失）。
        """
        if not self.format or "GT" not in self.format:
            return []
        gt_idx = self.format.index("GT")
        genotypes = []
        for sample in self.samples:
            fields = sample.split(":")
            if gt_idx >= len(fields) or fields[gt_idx] in (".", "./."):
                genotypes.append(None)
                continue
            gt_str = fields[gt_idx].replace("|", "/")
            parts = gt_str.split("/")
            try:
                genotypes.append((int(parts[0]), int(parts[1])))
            except (ValueError, IndexError):
                genotypes.append(None)
        return genotypes

    def __repr__(self) -> str:
        return f"VCF({self.chrom}:{self.pos} {self.ref}>{self.alt_str})"


class VCFReader:
    """
    VCF 文件流式读取器，支持 gzip 压缩。

    使用方式::

        reader = VCFReader("variants.vcf.gz")
        for record in reader:
            if record.is_snp:
                process(record)
    """

    def __init__(self, path: Union[str, Path]) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"VCF file not found: {self.path}")
        self.header_lines: List[str] = []
        self.sample_names: List[str] = []
        logger.info(f"VCFReader: {self.path}")

    def _open(self):
        p = str(self.path)
        if p.endswith(".gz") or p.endswith(".bgz"):
            return gzip.open(p, "rt", encoding="utf-8")
        return open(p, "r", encoding="utf-8")

    def __iter__(self) -> Iterator[VCFRecord]:
        with self._open() as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith("##"):
                    self.header_lines.append(line)
                    continue
                if line.startswith("#CHROM"):
                    cols = line.split("\t")
                    self.sample_names = cols[9:] if len(cols) > 9 else []
                    continue
                record = self._parse_record(line)
                if record is not None:
                    yield record

    def _parse_record(self, line: str) -> Optional[VCFRecord]:
        cols = line.split("\t")
        if len(cols) < 8:
            return None
        chrom, pos_s, vid, ref, alt_s, qual_s, filt_s, info_s = cols[:8]
        fmt = cols[8].split(":") if len(cols) > 8 else []
        samples = cols[9:] if len(cols) > 9 else []

        try:
            qual = float(qual_s) if qual_s not in (".", "") else None
        except ValueError:
            qual = None

        info: Dict[str, str] = {}
        for token in info_s.split(";"):
            if "=" in token:
                k, v = token.split("=", 1)
                info[k] = v
            else:
                info[token] = "True"

        return VCFRecord(
            chrom=chrom,
            pos=int(pos_s),
            id=vid,
            ref=ref,
            alt=alt_s.split(","),
            qual=qual,
            filter=[] if filt_s in (".", "PASS") else filt_s.split(";"),
            info=info,
            format=fmt,
            samples=samples,
        )

    def to_genotype_matrix(
        self, max_snps: Optional[int] = None
    ) -> Tuple[List[int], List[str]]:
        """
        将 VCF 中的 SNP 提取为整数编码的等位基因计数列表。

        Returns:
            (positions, [encoded_genotype_strings, ...])  —
            适合后续转为 numpy / torch 矩阵。
        """
        positions = []
        rows = []
        for rec in self:
            if not rec.is_snp:
                continue
            gts = rec.get_genotypes()
            if not gts:
                continue
            positions.append(rec.pos)
            # 每个样本的备择等位基因计数 (0/1/2/miss=-1)
            encoded = []
            for gt in gts:
                if gt is None:
                    encoded.append(-1)
                else:
                    encoded.append(gt[0] + gt[1])  # 0=hom-ref, 1=het, 2=hom-alt
            rows.append(encoded)
            if max_snps and len(positions) >= max_snps:
                break
        return positions, rows


class VCFWriter:
    """VCF 文件写入器。"""

    def __init__(self, path: Union[str, Path], sample_names: Optional[List[str]] = None) -> None:
        self.path = Path(path)
        self.sample_names = sample_names or []
        self._f = self.path.open("w", encoding="utf-8")
        self._wrote_header = False
        self._meta_lines: List[str] = []

    def write_header(
        self,
        extra_meta: Optional[List[str]] = None,
        source: Optional[str] = None,
        fileformat: str = "VCFv4.2",
    ) -> None:
        self._f.write(f"##fileformat={fileformat}\n")
        if source:
            meta = f"##source={source}"
            self._f.write(meta + "\n")
            self._meta_lines.append(meta)
        for line in (extra_meta or []):
            self._f.write(line + "\n")
            self._meta_lines.append(line)
        cols = ["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
        if self.sample_names:
            cols += ["FORMAT"] + self.sample_names
        self._f.write("\t".join(cols) + "\n")
        self._wrote_header = True

    def write_record(self, rec: VCFRecord) -> None:
        if not self._wrote_header:
            self.write_header()
        qual_s = f"{rec.qual:.1f}" if rec.qual is not None else "."
        filt_s = "PASS" if not rec.filter else ";".join(rec.filter)
        info_s = ";".join(
            f"{k}={v}" if v != "True" else k for k, v in rec.info.items()
        ) or "."
        cols = [
            rec.chrom, str(rec.pos), rec.id, rec.ref, rec.alt_str,
            qual_s, filt_s, info_s,
        ]
        if rec.format:
            cols.append(":".join(rec.format))
            cols.extend(rec.samples)
        elif self.sample_names:
            cols.append("GT")
            if rec.samples:
                cols.extend(rec.samples)
            else:
                cols.extend(["./."] * len(self.sample_names))
        self._f.write("\t".join(cols) + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "VCFWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
