"""Deprecated: use ``cropability.ngs``."""

from cropability.ngs.io import (
    AlignmentFile,
    AlignmentInputManager,
    FastaReader,
    VCFReader,
    VCFRecord,
    VCFWriter,
)

__all__ = [
    "FastaReader",
    "AlignmentFile",
    "AlignmentInputManager",
    "VCFReader",
    "VCFRecord",
    "VCFWriter",
]
