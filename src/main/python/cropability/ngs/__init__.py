"""
NGS / CPU layer — thin Python facade over ``cropability.native._core`` (Rust).

Use this package for I/O, pileup, and variant pipelines. GPU genomics lives in
``cropability.genomics``.
"""

from cropability.ngs.fastcall3 import FastCall3Config, FastCall3RunResult, FastCall3Runner
from cropability.ngs.io import (
    AlignmentFile,
    AlignmentInputManager,
    FastaReader,
    VCFReader,
    VCFRecord,
    VCFWriter,
)
from cropability.ngs.pipeline import PipelineConfig, QCThresholds, VariantPipeline
from cropability.ngs.pileup import (
    MpileupParser,
    NativePileupEngine,
    PileupRecord,
    PileupSample,
    PileupSiteSummary,
)

__all__ = [
    "AlignmentFile",
    "AlignmentInputManager",
    "FastaReader",
    "VCFReader",
    "VCFRecord",
    "VCFWriter",
    "PipelineConfig",
    "QCThresholds",
    "VariantPipeline",
    "MpileupParser",
    "NativePileupEngine",
    "PileupRecord",
    "PileupSample",
    "PileupSiteSummary",
    "FastCall3Config",
    "FastCall3RunResult",
    "FastCall3Runner",
]
