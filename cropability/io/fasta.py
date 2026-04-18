"""
FASTA/FASTQ 文件读写
====================
提供流式（生成器）读取，支持 gzip 压缩文件。
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple, Union

from cropability.utils.logging import get_logger

logger = get_logger(__name__)


def _open_file(path: Union[str, Path]):
    """自动识别 gzip 压缩格式。"""
    path = Path(path)
    if path.suffix in (".gz", ".gzip"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


class FastaReader:
    """
    FASTA/FASTQ 流式读取器。

    支持格式：
    - `.fa`, `.fasta`, `.fna` — FASTA
    - `.fq`, `.fastq`         — FASTQ（自动忽略质量行）
    - `.gz` 后缀              — gzip 压缩

    使用方式::

        reader = FastaReader("genome.fa.gz")
        for name, seq in reader:
            process(name, seq)
    """

    def __init__(
        self,
        path: Union[str, Path],
        chunk_size: int = 100_000,
    ) -> None:
        self.path = Path(path)
        self.chunk_size = chunk_size
        self._is_fastq = self.path.stem.split(".")[-1] in ("fq", "fastq") or \
                         str(self.path).endswith((".fq", ".fastq",
                                                   ".fq.gz", ".fastq.gz"))
        if not self.path.exists():
            raise FileNotFoundError(f"Sequence file not found: {self.path}")
        logger.info(f"FastaReader: {self.path} ({'FASTQ' if self._is_fastq else 'FASTA'})")

    def __iter__(self) -> Iterator[Tuple[str, str]]:
        if self._is_fastq:
            yield from self._read_fastq()
        else:
            yield from self._read_fasta()

    def _read_fasta(self) -> Generator[Tuple[str, str], None, None]:
        """读取 FASTA，逐序列 yield (name, sequence)。"""
        current_name: Optional[str] = None
        buf: List[str] = []
        with _open_file(self.path) as f:
            for line in f:
                line = line.rstrip("\n")
                if line.startswith(">"):
                    if current_name is not None:
                        yield current_name, "".join(buf)
                    current_name = line[1:].split()[0]
                    buf = []
                elif line and current_name is not None:
                    buf.append(line)
            if current_name is not None and buf:
                yield current_name, "".join(buf)

    def _read_fastq(self) -> Generator[Tuple[str, str], None, None]:
        """读取 FASTQ，忽略质量行。"""
        with _open_file(self.path) as f:
            while True:
                header = f.readline().rstrip()
                if not header:
                    break
                seq = f.readline().rstrip()
                f.readline()     # '+'
                f.readline()     # quality
                if header.startswith("@"):
                    yield header[1:].split()[0], seq

    def read_all(self) -> Dict[str, str]:
        """将全部序列读入内存字典 {name: sequence}。"""
        return {name: seq for name, seq in self}

    def chunked_sequences(
        self, chunk_size: Optional[int] = None
    ) -> Generator[Tuple[str, List[str]], None, None]:
        """
        将每条序列切分为固定长度的片段，适合大基因组的流式处理。

        Yields:
            (original_name, [chunk1, chunk2, ...])
        """
        cs = chunk_size or self.chunk_size
        for name, seq in self:
            chunks = [seq[i : i + cs] for i in range(0, len(seq), cs)]
            yield name, chunks


class FastaWriter:
    """FASTA 文件写入器。"""

    def __init__(
        self,
        path: Union[str, Path],
        line_width: int = 60,
        compress: bool = False,
    ) -> None:
        self.path = Path(path)
        self.line_width = line_width
        self._f = gzip.open(self.path, "wt") if compress else self.path.open("w")

    def write(self, name: str, sequence: str) -> None:
        self._f.write(f">{name}\n")
        for i in range(0, len(sequence), self.line_width):
            self._f.write(sequence[i : i + self.line_width] + "\n")

    def close(self) -> None:
        self._f.close()

    def __enter__(self) -> "FastaWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()
