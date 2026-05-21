# CropAbility

**High-performance GPU computing framework for plant genomics**

A GPU-accelerated plant genomics toolkit built on [Triton](https://github.com/openai/triton) and PyTorch,
optimized for **NVIDIA H100 PCIe** and **A2** dual-GPU setups while remaining CPU-compatible.

---

## Architecture overview

```
src/main/python/cropability/   # GPU kernels, viz, ngs facade, CLI (Python host)
src/main/rust/               # CPU: I/O, NGS pipeline (PyO3 extension)
src/main/resources/          # Config + private assets (gitignored private/)
src/test/python/             # Pytest (see doc/TESTING.md)
archive/legacy/pgl/          # Legacy reference only (not installed)
doc/                         # Development and testing docs
```

Single root `Cargo.toml` — no nested Rust test crates or TorchScript.

**Python**: GPU compute + `cropability.viz`. **Rust**: I/O, pileup, variant pipeline, config orchestration (see `doc/RUST_DEVELOPMENT.md`).

---

## Quick start

### Requirements

| Dependency | Version |
|------------|---------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 (CUDA 12.x) |
| Triton | ≥ 2.1 |
| CUDA Driver | ≥ 520 (H100 requires ≥ 525) |
| Rust (optional) | stable (PyO3 / maturin) |

### Install

**Recommended — Conda** (Python, Rust, bioinformatics tools, and pip extras):

```bash
git clone https://github.com/example/CropAbility.git
cd CropAbility
conda env create -f environment.yml
conda activate cropability

# CPU-only (no CUDA): environment_cpu.yml
# Then: maturin develop --release --features python,htslib && cargo test
```

**Pip only** (install Rust and samtools separately if using native layers):

```bash
pip install -e ".[gpu,dev,io,viz]"
```

### Basic usage

```bash
# System and GPU info
cropability info

# GPU benchmark
cropability benchmark --n-seqs 5000 --seq-len 512 --matrix-size 8192

# SNP calling from FASTA
cropability snp -i samples.fa --min-af 0.05 --min-depth 10

# LD matrix
cropability ld --n-samples 500 --n-snps 1000

# Native mpileup (real BAM/CRAM input)
cropability pileup -r ref.fa -b sample1.bam sample2.bam -o cohort.mpileup

# Variant calling pipeline (default hybrid: native mpileup + FastCall3 logic)
cropability call-variants -r ref.fa -b sample1.bam sample2.bam -o cohort.vcf --mode hybrid
```

> NGS pipelines run inside CropAbility by default; no external `samtools`/`FastCall3` binaries required.
> Runtime needs `pysam` (`pip install "cropability[io]"`). An optional Rust backend can accelerate I/O.

### Python API

```python
from cropability.gpu import get_device_manager
from cropability.kernels.seq import encode_sequences, gc_content_kernel
from cropability.kernels.stats import zscore_normalize, pearson_correlation
from cropability.genomics.variant import VariantCaller
from cropability.genomics.ld import LDCalculator
from cropability.genomics.gwas import GWASEngine

# GPU device manager (auto-detects H100/A2)
dm = get_device_manager(device_ids=[0, 1])
dm.print_memory_stats()

# Sequence encoding and GC content
sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "NNNATCGAAANN"]
encoded = encode_sequences(sequences, device=dm.get_primary_device())
gc = gc_content_kernel(encoded)

# SNP calling
caller = VariantCaller(device=dm.get_primary_device())
snps = caller.call_snps(sample_seqs, reference)

# GWAS (GPU-accelerated linear regression)
import torch
engine = GWASEngine(device=dm.get_primary_device())
results = engine.run_linear_gwas(
    genotypes=torch.randint(0, 3, (1000, 50000)).float(),
    phenotype=torch.randn(1000),
)
hits = [r for r in results if r.is_significant()]
```

### Multi-GPU distributed training

```python
from cropability.gpu.distributed import launch_ddp

def train_fn(rank: int, world_size: int, **kwargs):
    from cropability.gpu.distributed import wrap_ddp
    model = wrap_ddp(MyModel())
    # ... training loop ...

launch_ddp(train_fn, num_gpus=2)
```

### Rust native extension (PyO3)

```bash
maturin develop --release --features python,htslib
```

See [doc/RUST_DEVELOPMENT.md](doc/RUST_DEVELOPMENT.md) for build steps and Python API.

---

## Testing

See **[doc/TESTING.md](doc/TESTING.md)** for layout, markers, and CI-style commands.

```bash
pytest src/test/python
maturin develop --release --features python,htslib
pytest -m native
```

---

## Configuration

Create `cropability.yaml` at the project root to override defaults:

```yaml
gpu:
  device_ids: [0, 1]
  memory_fraction: 0.90
  mixed_precision: true

compute:
  batch_size: 256
  block_size: 2048

genomics:
  min_base_quality: 20
  kmer_size: 31

logging:
  level: INFO
  file: cropability.log
```

Environment variables use `CROPABILITY_<SECTION>__<KEY>`:

```bash
export CROPABILITY_GPU__DEVICE_IDS=0,1
export CROPABILITY_COMPUTE__BATCH_SIZE=512
```

---

## Modules

### `cropability.gpu`

- `DeviceManager`: enumerate GPUs (H100/A2), memory quotas, pick least-busy device
- `get_device_manager()`: global singleton
- `launch_ddp()` / `wrap_ddp()`: single-node multi-GPU DDP

### `cropability.kernels`

Triton GPU paths with PyTorch CPU fallback when CUDA is unavailable.

| Kernel | Purpose |
|--------|---------|
| `encode_sequences` | ASCII → int8 (A=0,C=1,G=2,T=3,N=4) |
| `gc_content_kernel` | Batch GC content (N filtered) |
| `reverse_complement_kernel` | Batch reverse complement on GPU |
| `kmer_count_kernel` | Normalized k-mer frequency vectors |
| `welford_mean_var` | Numerically stable mean/variance |
| `zscore_normalize` | Row-wise z-score |
| `pearson_correlation` | [M,D]×[N,D] → [M,N] correlation matrix |
| `hamming_distance_matrix` | Pairwise Hamming distances |
| `jaccard_similarity_matrix` | k-mer Jaccard similarity |

### `cropability.genomics`

| Class | Purpose |
|-------|---------|
| `VariantCaller` | Multi-sample SNP calling (AF/depth/Phred filters) |
| `LDCalculator` | r² LD matrix (blocked), LD pruning |
| `GWASEngine` | Linear GWAS (OLS + PC covariates + SVD-PCA) |
| `SmithWatermanGPU` | Batch Smith-Waterman scoring |

---

## Development

```bash
pip install -e ".[dev]"
pytest src/test/python
cargo test
ruff check src/main/python/cropability
```

---

## AI agent configuration

- **[AGENTS.md](AGENTS.md)** — instructions for Cursor, Codex, and other agents
- **[.cursor/rules/](.cursor/rules/)** — Cursor project rules (English-only policy, layout)

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Acknowledgments

Built with [OpenAI Triton](https://github.com/openai/triton), [PyTorch](https://pytorch.org), and Rust (PyO3 / maturin).
