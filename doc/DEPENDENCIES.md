# Dependencies and environment

CropAbility does **not** vendor native libraries under `lib/`. Use the unified Conda base plus pip/maturin steps below.

## One-command setup (recommended)

From the repository root:

```bash
python install.py              # GPU: CUDA PyTorch + Triton + Rust _core
python install.py --cpu        # CPU-only PyTorch + Rust _core (no Triton)
```

`install.py` will:

1. Create or update the **`cropability`** Conda env from `environment.yml` (Python, Rust, gcc, htslib, samtools, …).
2. Install Python packages from `requirements-gpu.txt` or `requirements-cpu.txt` (PyTorch + editable `cropability` extras).
3. Run `maturin develop --release --features python,htslib` with the correct compiler and HTSlib paths.

Skip steps when re-running:

```bash
python install.py --skip-conda              # env already active
python install.py --skip-conda --skip-pip   # rebuild Rust only
python install.py --skip-rust               # refresh pip deps only
```

## Manual setup

### 1. Conda base

```bash
conda env create -f environment.yml
conda activate cropability
```

### 2. Python (pip)

| Mode | Command |
|------|---------|
| GPU | `pip install -r requirements-gpu.txt` |
| CPU | `pip install -r requirements-cpu.txt` |

These files install PyTorch from the official wheel index and the project in editable mode with extras (`[gpu,dev,io,rust]` or `[dev,io,rust]`).

### 3. Rust extension (maturin)

```bash
export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LIBCLANG_PATH=$CONDA_PREFIX/lib    # or /usr/lib/llvm-18/lib if needed
export HTSLIB_DIR=$CONDA_PREFIX
export PKG_CONFIG_PATH=$CONDA_PREFIX/lib/pkgconfig
maturin develop --release --features python,htslib
```

(`install.py` sets these automatically.)

## Files

| File | Role |
|------|------|
| `environment.yml` | Minimal Conda base (Python 3.11, Rust, bioinformatics CLI, HTSlib, build tools) |
| `requirements-gpu.txt` | PyTorch CUDA 12.1 wheels + `cropability[gpu,dev,io,rust]` |
| `requirements-cpu.txt` | PyTorch CPU wheels + `cropability[dev,io,rust]` |
| `install.py` | Orchestrates Conda + pip + maturin in one command |
| `pyproject.toml` | Package extras: `gpu`, `dev`, `io`, `rust`, `viz` |

There is **no separate CPU Conda env file**. GPU vs CPU is chosen at pip install time (`requirements-gpu.txt` vs `requirements-cpu.txt` or `install.py --cpu`).

## What each layer provides

| Need | Source | Notes |
|------|--------|-------|
| PyTorch (GPU) | pip + `requirements-gpu.txt` | `--extra-index-url` → PyTorch cu121 wheels |
| PyTorch (CPU) | pip + `requirements-cpu.txt` | CPU wheel index |
| Triton GPU kernels | pip via `[gpu]` extra | GPU path only |
| NumPy, PyYAML | `pyproject.toml` / pip | |
| BAM/FASTA (Python) | `pysam`, `biopython` via `[io]` | |
| Rust toolchain | Conda: `rust`, `cmake`, `gcc`, `libclang` | |
| HTSlib (Rust link) | Conda: `htslib` | `HTSLIB_DIR=$CONDA_PREFIX` for maturin |
| Bio CLI | Conda: `samtools`, `bcftools`, `bedtools` | |
| PyO3 extension build | pip: `maturin` via `[rust]` | |
| Tests / lint | pip via `[dev]` | pytest, ruff, mypy, … |

## Removed: repo `lib/` directory

The former `lib/` folder held vendored libtorch and Java JNI artifacts. None of that is used by the current **Python host → Rust PyO3** design.

## `archive/legacy/pgl/`

Not a dependency — historical reference only. See [archive/legacy/README.md](../archive/legacy/README.md).
