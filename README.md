# CropAbility

**High-performance GPU computing framework for plant genomics**

A GPU-accelerated plant genomics toolkit built on [Triton](https://github.com/openai/triton) and PyTorch,
optimized for **NVIDIA H100 PCIe** and **A2** dual-GPU setups while remaining CPU-compatible.

---

## Architecture overview

```
src/main/python/cropability/   # GPU kernels, viz, ngs facade, CLI (Python host)
src/main/rust/                 # CPU: I/O, NGS pipeline (PyO3 → cropability.native._core)
src/main/resources/            # Config + private assets (private/ gitignored)
src/test/python/               # Pytest suite
archive/legacy/pgl/            # Legacy PGL reference only (not installed)
doc/                           # TODO checklist + engineering progress log only
```

Single root `Cargo.toml` — no nested Rust workspaces, TorchScript, or Java.

| Layer | Role |
|-------|------|
| **Python** | PyTorch GPU, Triton kernels, `cropability.viz`, CLI |
| **Rust** | FASTA/BAM/VCF I/O, pileup, FastCall3-style pipeline via PyO3 |
| **`cropability.ngs`** | Thin Python facade over `_core` |

Rust layout (`src/main/rust/`): `lib.rs`, `python.rs` (PyO3), `io/`, `genomics/`.  
Cargo features: `python` (extension), `htslib` (BAM via rust-htslib). No libtorch / `tch`.

---

## Requirements

| Dependency | Version |
|------------|---------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 (CUDA 12.x for GPU install) |
| Triton | ≥ 2.1 (GPU path) |
| CUDA driver | ≥ 520 (H100 ≥ 525) |
| Rust (optional) | stable — PyO3 / maturin |

CropAbility does **not** vendor native libraries under `lib/`. Use Conda + pip + maturin below.

---

## Install

### One command (recommended)

```bash
git clone https://github.com/example/CropAbility.git
cd CropAbility
python install.py              # GPU: CUDA PyTorch + Triton + Rust _core
# python install.py --cpu      # CPU-only PyTorch + Rust _core (no Triton)
conda activate cropability
```

`install.py`:

1. Creates or updates the **`cropability`** Conda env from `environment.yml` (Python, Rust, gcc, htslib, samtools, …).
2. Installs from `requirements-gpu.txt` or `requirements-cpu.txt`.
3. Runs `maturin develop --release --features python,htslib`.

Re-run with skips:

```bash
python install.py --skip-conda              # env already active
python install.py --skip-conda --skip-pip   # rebuild Rust only
python install.py --skip-rust               # refresh pip deps only
```

### Manual setup

**Conda base:**

```bash
conda env create -f environment.yml
conda activate cropability
```

**Python (pip):**

| Mode | Command |
|------|---------|
| GPU | `pip install -r requirements-gpu.txt` |
| CPU | `pip install -r requirements-cpu.txt` |

**Rust extension (maturin):**

```bash
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LIBCLANG_PATH=$CONDA_PREFIX/lib
export HTSLIB_DIR=$CONDA_PREFIX
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
maturin develop --release --features python,htslib
```

(`install.py` sets compiler and HTSlib paths automatically.)

### Pip only (no Conda orchestration)

```bash
pip install -e ".[gpu,dev,io,viz]"    # or CPU extras without [gpu]
```

Install Rust, samtools, and HTSlib separately if you use native BAM paths.

### Install artifacts

| File | Role |
|------|------|
| `environment.yml` | Conda base (Python 3.11, Rust, HTSlib, samtools, build tools) |
| `requirements-gpu.txt` | PyTorch CUDA 12.1 wheels + `cropability[gpu,dev,io,rust]` |
| `requirements-cpu.txt` | PyTorch CPU wheels + `cropability[dev,io,rust]` |
| `install.py` | Conda + pip + maturin orchestration |
| `pyproject.toml` | Extras: `gpu`, `dev`, `io`, `rust`, `viz` |

### What each layer provides

| Need | Source |
|------|--------|
| PyTorch (GPU/CPU) | pip + `requirements-gpu.txt` / `requirements-cpu.txt` |
| Triton | pip `[gpu]` extra |
| BAM/FASTA (Python) | `pysam`, `biopython` via `[io]` |
| Rust toolchain | Conda: `rust`, `cmake`, `gcc`, `libclang` |
| HTSlib (Rust link) | Conda `htslib`; `HTSLIB_DIR=$CONDA_PREFIX` |
| Bio CLI | Conda: `samtools`, `bcftools`, `bedtools` |
| PyO3 build | pip `maturin` via `[rust]` |
| Tests / lint | pip `[dev]` — pytest, ruff, mypy |

There is **no** separate CPU Conda env file; GPU vs CPU is chosen at pip/`install.py` time.

---

## Quick start (CLI)

```bash
cropability info
cropability benchmark --n-seqs 5000 --seq-len 512 --matrix-size 8192
cropability snp -i samples.fa --min-af 0.05 --min-depth 10
cropability ld --n-samples 500 --n-snps 1000
cropability pileup -r ref.fa -b sample1.bam sample2.bam -o cohort.mpileup
cropability call-variants -r ref.fa -b sample1.bam sample2.bam -o cohort.vcf --mode hybrid
```

NGS pipelines run in-process by default (no external `samtools`/`FastCall3` binaries).  
Use `pip install "cropability[io]"` for `pysam`. Optional Rust `_core` accelerates I/O and pileup.

---

## Python API

```python
from cropability.gpu import get_device_manager
from cropability.kernels.seq import encode_sequences, gc_content_kernel
from cropability.kernels.stats import zscore_normalize, pearson_correlation
from cropability.genomics.variant import VariantCaller
from cropability.genomics.ld import LDCalculator
from cropability.genomics.gwas import GWASEngine

dm = get_device_manager(device_ids=[0, 1])
dm.print_memory_stats()

sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "NNNATCGAAANN"]
encoded = encode_sequences(sequences, device=dm.get_primary_device())
gc = gc_content_kernel(encoded)

caller = VariantCaller(device=dm.get_primary_device())
snps = caller.call_snps(sample_seqs, reference)

import torch
engine = GWASEngine(device=dm.get_primary_device())
results = engine.run_linear_gwas(
    genotypes=torch.randint(0, 3, (1000, 50000)).float(),
    phenotype=torch.randn(1000),
)
```

### Multi-GPU (DDP)

```python
from cropability.gpu.distributed import launch_ddp, wrap_ddp

def train_fn(rank: int, world_size: int, **kwargs):
    model = wrap_ddp(MyModel())
    # ... training loop ...

launch_ddp(train_fn, num_gpus=2)
```

### Rust native extension (Python)

```python
from cropability.native._core import FastaReader, VariantPipeline, QCThresholds

reader = FastaReader("samples.fa")
data = reader.read_all()

pipe = VariantPipeline()
report = pipe.run(
    mode="hybrid",
    reference="ref.fa",
    bam_files=["s1.bam"],
    output="out.vcf",
    qc=QCThresholds(),
)
```

Rebuild extension only:

```bash
python install.py --skip-conda --skip-pip
# or: maturin develop --release --features python,htslib
```

Use **`cropability.ngs`** for NGS APIs (not deprecated `cropability.io` shims).  
Legacy PGL (TorchScript-era) lives under `archive/legacy/pgl/` — not installed.

---

## Configuration

`cropability.yaml` at project root:

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

Environment overrides: `CROPABILITY_<SECTION>__<KEY>` (e.g. `CROPABILITY_GPU__DEVICE_IDS=0,1`).

---

## Modules (summary)

### `cropability.gpu`

`DeviceManager`, `get_device_manager()`, `launch_ddp()` / `wrap_ddp()`.

### `cropability.kernels`

Triton GPU paths with PyTorch CPU fallback.

| Kernel | Purpose |
|--------|---------|
| `encode_sequences` | ASCII → int8 |
| `gc_content_kernel` | Batch GC content |
| `reverse_complement_kernel` | Batch reverse complement |
| `kmer_count_kernel` | k-mer frequency vectors |
| `welford_mean_var` / `zscore_normalize` | Stats |
| `pearson_correlation` | Correlation matrix |
| `hamming_distance_matrix` / `jaccard_similarity_matrix` | Similarity |

### `cropability.genomics`

`VariantCaller`, `LDCalculator`, `GWASEngine`, `SmithWatermanGPU`.

---

## Testing

```bash
pip install -e ".[dev]"
pytest src/test/python

maturin develop --release --features python,htslib
pytest src/test/python -m native

cargo test    # optional Rust unit tests
```

| Marker | Meaning |
|--------|---------|
| `native` | Needs `cropability.native._core` |
| `gpu` | Needs CUDA |
| `slow` | Long-running (reserved) |

```bash
pytest -m native
pytest -m "not native"
pytest -m gpu
pytest --cov=cropability
```

Tests live only under `src/test/python/` (no `scripts/`).  
Detailed layout, file map, and conventions for changing tests are recorded in
[`doc/TODO_PROGRESS_LOG.md`](doc/TODO_PROGRESS_LOG.md) (engineering log).

---

## Development

```bash
pip install -e ".[dev]"
pytest src/test/python
cargo test
ruff check src/main/python/cropability
```

**Project tracking:** [`doc/TODO.md`](doc/TODO.md) (checklist),
[`doc/TODO_PROGRESS_LOG.md`](doc/TODO_PROGRESS_LOG.md) (dated work log when you change code or close tasks).

---

## AI agent configuration

- **[AGENTS.md](AGENTS.md)** — agent instructions
- **[.cursor/rules/](.cursor/rules/)** — Cursor project rules

---

## License

MIT License — see [LICENSE](LICENSE)

---

## Acknowledgments

Built with [OpenAI Triton](https://github.com/openai/triton), [PyTorch](https://pytorch.org), and Rust (PyO3 / maturin).
