"""
FASTA/FASTQ read/write
======================
Streaming (generator-based) reading with gzip compression support.
"""

from __future__ import annotations

import gzip
import io
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Tuple, Union

from cropability.utils.logging import get_logger

logger = get_logger(__name__)


def _open_file(path: Union[str, Path]):
    """Detect and open gzip-compressed files automatically."""
    path = Path(path)
    if path.suffix in (".gz", ".gzip"):
        return gzip.open(path, "rt", encoding="utf-8")
    return path.open("r", encoding="utf-8")


class FastaReader:
    """
    Streaming FASTA/FASTQ reader.

    Supported formats:
    - `.fa`, `.fasta`, `.fna` — FASTA
    - `.fq`, `.fastq`         — FASTQ (quality lines are skipped)
    - `.gz` suffix            — gzip compression

    Usage::

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
        """Read FASTA and yield (name, sequence) per record."""
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
        """Read FASTQ and skip quality lines."""
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
        """Load all sequences into an in-memory {name: sequence} dict."""
        return {name: seq for name, seq in self}

    def chunked_sequences(
        self, chunk_size: Optional[int] = None
    ) -> Generator[Tuple[str, List[str]], None, None]:
        """
        Split each sequence into fixed-length chunks for streaming large genomes.

        Yields:
            (original_name, [chunk1, chunk2, ...])
        """
        cs = chunk_size or self.chunk_size
        for name, seq in self:
            chunks = [seq[i : i + cs] for i in range(0, len(seq), cs)]
            yield name, chunks


class FastaWriter:
    """FASTA file writer."""

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
