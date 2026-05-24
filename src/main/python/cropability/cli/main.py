"""
CropAbility command-line tool
==============================
Entry point: ``cropability <subcommand> [options]``

Subcommands:
  info      — Print system/GPU information
  snp       — Detect SNPs from a FASTA file
  ld        — Compute linkage disequilibrium matrix
  gwas      — Genome-wide association analysis
  align     — Batch sequence alignment
  pileup    — Run native mpileup (built into CropAbility)
  call-variants — Run NGS variant detection pipeline
  benchmark — GPU performance benchmark
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional

from cropability.utils.logging import setup_logging, get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------

def cmd_info(args: argparse.Namespace) -> int:
    """Print system, GPU, and library version information."""
    import torch
    from cropability.gpu import get_device_manager
    import cropability

    print(f"\n{'=' * 50}")
    print(f"  CropAbility v{cropability.__version__}")
    print(f"{'=' * 50}")
    print(f"  Python  : {sys.version.split()[0]}")
    print(f"  PyTorch : {torch.__version__}")
    print(f"  CUDA    : {torch.version.cuda or 'N/A'}")

    try:
        import triton
        print(f"  Triton  : {triton.__version__}")
    except ImportError:
        print("  Triton  : not installed")

    dm = get_device_manager()
    print(f"\n  GPUs ({dm.num_gpus} available):")
    for dev in dm.devices:
        print(f"    cuda:{dev.index}  {dev.name}  "
              f"{dev.total_memory_gb:.1f} GB  "
              f"CC {dev.compute_capability[0]}.{dev.compute_capability[1]}  "
              f"{'BF16' if dev.supports_bf16 else 'FP16'}")

    if dm.has_gpu:
        print("\n  Memory status:")
        for name, s in dm.memory_stats().items():
            if "error" not in s:
                print(f"    {name}: {s['used_gb']:.2f}/{s['total_gb']:.2f} GB "
                      f"({s['utilization'] * 100:.1f}% used)")
    print()
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """GPU performance benchmark."""
    import torch
    from cropability.gpu import get_device_manager
    from cropability.utils.timer import Timer
    from cropability.kernels.seq import encode_sequences, gc_content_kernel
    from cropability.kernels.stats import zscore_normalize

    dm = get_device_manager()
    device = dm.get_primary_device()
    print(f"\nBenchmark on {device}")

    n, l = args.n_seqs, args.seq_len
    print(f"  Sequences: {n} × {l} bp")

    # Generate random sequences
    import random
    bases = "ACGT"
    seqs = ["".join(random.choices(bases, k=l)) for _ in range(n)]

    with Timer("encode_sequences", use_cuda=True) as t:
        encoded = encode_sequences(seqs, device=device)
    print(f"  encode_sequences   : {t.elapsed_ms:.2f} ms")

    with Timer("gc_content", use_cuda=True) as t:
        _ = gc_content_kernel(encoded)
    print(f"  gc_content_kernel  : {t.elapsed_ms:.2f} ms")

    # Matrix benchmark
    d = args.matrix_size
    print(f"\n  Matrix: {d}×{d} float32")
    A = torch.randn(d, d, device=device)
    B = torch.randn(d, d, device=device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    with Timer("matmul", use_cuda=True) as t:
        _ = torch.mm(A, B)
    tflops = 2 * d ** 3 / (t.elapsed_ms / 1000) / 1e12
    print(f"  torch.mm {d}×{d}   : {t.elapsed_ms:.2f} ms  ({tflops:.2f} TFLOPS)")

    # zscore
    X = torch.randn(n, l, device=device)
    with Timer("zscore", use_cuda=True) as t:
        _ = zscore_normalize(X)
    print(f"  zscore_normalize   : {t.elapsed_ms:.2f} ms  ({n}×{l})")

    print()
    return 0


def cmd_snp(args: argparse.Namespace) -> int:
    """Detect SNPs from aligned FASTA files."""
    from cropability.ngs.io import FastaReader
    from cropability.genomics.variant import VariantCaller
    from cropability.gpu import get_device_manager

    dm = get_device_manager()
    device = dm.get_primary_device()

    reader = FastaReader(args.input)
    seqs = reader.read_all()
    names = list(seqs.keys())
    sequences = list(seqs.values())

    if len(sequences) < 2:
        print("ERROR: Need at least 2 sequences (reference + samples)")
        return 1

    reference = sequences[0]
    samples = sequences[1:]
    print(f"Reference: {names[0]} ({len(reference)} bp)")
    print(f"Samples: {len(samples)}")

    caller = VariantCaller(
        device=device,
        min_alt_freq=args.min_af,
        min_depth=args.min_depth,
    )
    snps = caller.call_snps(samples, reference)
    snps = caller.filter_snps(snps)

    print(f"\nDetected {len(snps)} SNPs:")
    for snp in snps[:20]:
        print(f"  {snp}")
    if len(snps) > 20:
        print(f"  ... and {len(snps) - 20} more")
    return 0


def cmd_ld(args: argparse.Namespace) -> int:
    """Compute LD matrix (example using a random genotype matrix)."""
    import torch
    from cropability.genomics.ld import LDCalculator
    from cropability.gpu import get_device_manager

    dm = get_device_manager()
    device = dm.get_primary_device()

    n_samples, n_snps = args.n_samples, args.n_snps
    print(f"Computing LD matrix: {n_samples} samples × {n_snps} SNPs on {device}")
    genotypes = torch.randint(0, 3, (n_samples, n_snps)).float()

    calc = LDCalculator(device=device)
    result = calc.compute_ld_matrix(genotypes)
    pairs = result.high_ld_pairs(threshold=0.5)
    print(f"High-LD pairs (r²>0.5): {len(pairs)}")
    for p in pairs[:10]:
        print(f"  pos {p[0]} ↔ {p[1]}: r²={p[2]:.3f}")
    return 0


def cmd_pileup(args: argparse.Namespace) -> int:
    """Run native mpileup and write site summary text."""
    from cropability.ngs.pipeline import QCThresholds, VariantPipeline

    qc = QCThresholds(
        min_depth=args.min_depth,
        min_base_quality=args.min_baseq,
        min_mapping_quality=args.min_mapq,
        min_alt_freq=args.min_af,
    )
    pipeline = VariantPipeline()
    report = pipeline.run(
        mode="mpileup",
        reference=args.reference,
        bam_files=args.bam,
        output=args.output,
        qc=qc,
        regions=args.region,
        dry_run=args.dry_run,
    )
    print("native pileup completed")
    print(f"  output: {report['mpileup']['output']}")
    return 0


def cmd_call_variants(args: argparse.Namespace) -> int:
    """Run native variant detection pipeline (mpileup / fastcall3 / hybrid)."""
    from cropability.ngs.pipeline import QCThresholds, VariantPipeline

    qc = QCThresholds(
        min_depth=args.min_depth,
        min_base_quality=args.min_baseq,
        min_mapping_quality=args.min_mapq,
        min_alt_freq=args.min_af,
    )
    pipeline = VariantPipeline()
    report = pipeline.run(
        mode=args.mode,
        reference=args.reference,
        bam_files=args.bam,
        output=args.output,
        qc=qc,
        regions=args.region,
        mpileup_output=args.mpileup_output,
        dry_run=args.dry_run,
    )
    print(f"pipeline mode={report['mode']} completed")
    if "mpileup" in report:
        print(f"  mpileup: {report['mpileup']['output']}")
    if "fastcall3" in report:
        fc = report["fastcall3"]
        print(f"  vcf: {fc.get('output_vcf', fc.get('output', ''))}")
        print(f"  records: {fc.get('n_records', 0)}")
        print(f"  elapsed: {fc.get('elapsed_seconds', 0):.1f}s")
    return 0


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cropability",
        description="CropAbility — high-performance GPU framework for plant genomics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Log level (default: INFO)")
    parser.add_argument("--log-file", default=None, help="Log output file (optional)")

    sub = parser.add_subparsers(dest="command", required=True)

    # info
    sub.add_parser("info", help="Print system/GPU information")

    # benchmark
    bench = sub.add_parser("benchmark", help="GPU performance benchmark")
    bench.add_argument("--n-seqs", type=int, default=1000)
    bench.add_argument("--seq-len", type=int, default=512)
    bench.add_argument("--matrix-size", type=int, default=4096)

    # snp
    snp = sub.add_parser("snp", help="Detect SNPs")
    snp.add_argument("-i", "--input", required=True, help="Multi-sample aligned FASTA file")
    snp.add_argument("--min-af", type=float, default=0.05, help="Minimum alternate allele frequency")
    snp.add_argument("--min-depth", type=int, default=10, help="Minimum coverage depth")

    # ld
    ld = sub.add_parser("ld", help="Compute linkage disequilibrium matrix")
    ld.add_argument("--n-samples", type=int, default=200)
    ld.add_argument("--n-snps", type=int, default=500)

    # pileup
    pileup = sub.add_parser("pileup", help="Run native mpileup")
    pileup.add_argument("-r", "--reference", required=True, help="Reference genome FASTA")
    pileup.add_argument("-b", "--bam", required=True, nargs="+", help="Input BAM/CRAM file list")
    pileup.add_argument("-o", "--output", required=True, help="Output site summary path (TSV)")
    pileup.add_argument("--region", default=None, help="Region filter, e.g. chr1:1-100000")
    pileup.add_argument("--min-depth", type=int, default=10, help="Minimum depth threshold")
    pileup.add_argument("--min-baseq", type=int, default=20, help="Minimum base quality")
    pileup.add_argument("--min-mapq", type=int, default=20, help="Minimum mapping quality")
    pileup.add_argument("--min-af", type=float, default=0.05, help="Minimum alternate allele frequency")
    pileup.add_argument("--dry-run", action="store_true", help="Print command only; do not execute")

    # call-variants
    call = sub.add_parser("call-variants", help="Run variant detection pipeline")
    call.add_argument("-r", "--reference", required=True, help="Reference genome FASTA")
    call.add_argument("-b", "--bam", required=True, nargs="+", help="Input BAM/CRAM file list")
    call.add_argument("-o", "--output", required=True, help="Output VCF (or mpileup) path")
    call.add_argument(
        "--mode",
        choices=["mpileup", "fastcall3", "hybrid"],
        default="hybrid",
        help="Pipeline mode",
    )
    call.add_argument("--mpileup-output", default=None, help="Intermediate mpileup output path for hybrid mode")
    call.add_argument("--region", default=None, help="Region filter, e.g. chr1:1-100000")
    call.add_argument("--min-depth", type=int, default=10, help="Minimum depth threshold")
    call.add_argument("--min-baseq", type=int, default=20, help="Minimum base quality")
    call.add_argument("--min-mapq", type=int, default=20, help="Minimum mapping quality")
    call.add_argument("--min-af", type=float, default=0.05, help="Minimum alternate allele frequency")
    call.add_argument("--dry-run", action="store_true", help="Print command only; do not execute")

    return parser


_COMMANDS = {
    "info": cmd_info,
    "benchmark": cmd_benchmark,
    "snp": cmd_snp,
    "ld": cmd_ld,
    "pileup": cmd_pileup,
    "call-variants": cmd_call_variants,
}


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(level=args.log_level, log_file=args.log_file)
    handler = _COMMANDS.get(args.command)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)


if __name__ == "__main__":
    sys.exit(main())
