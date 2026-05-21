# Dependencies and environment

CropAbility does **not** vendor native libraries under `lib/`. Use Conda + pip as defined in `environment.yml` / `environment_cpu.yml`.

## Conda environments

| File | Use |
|------|-----|
| `environment.yml` | GPU: Python 3.11, PyTorch+CUDA 12.1, Rust, htslib, samtools, pytest, maturin, editable `cropability[gpu,dev,io,rust]` |
| `environment_cpu.yml` | CPU: same stack with `cpuonly`, editable `cropability[dev,io,rust]` (no Triton extra) |

```bash
conda env create -f environment.yml    # or environment_cpu.yml
conda activate cropability             # or cropability-cpu
maturin develop --release --features python,htslib
```

## What each layer provides

| Need | Source | Notes |
|------|--------|-------|
| PyTorch / CUDA | `conda`: `pytorch`, `pytorch-cuda` or `cpuonly` | Replaces old `lib/libtorch*.so` |
| Triton GPU kernels | `pip`: `triton` via `[gpu]` extra | GPU env only |
| NumPy, PyYAML | conda + `pyproject.toml` | |
| BAM/FASTA (Python) | `pysam`, `biopython` via `[io]` | |
| Rust toolchain | `conda`: `rust`, `cmake`, `pkg-config` | |
| HTSlib (Rust link) | `conda`: `htslib` | For `maturin` feature `htslib` |
| Bio CLI | `conda`: `samtools`, `bcftools`, `bedtools` | Pipeline / ops |
| PyO3 extension build | `pip`: `maturin` via `[rust]` | |
| Tests / lint | `pytest`, `ruff`, etc. via `[dev]` | |

## Removed: repo `lib/` directory

The former `lib/` folder held:

- **libtorch** `.so` files (often symlinks into a Conda env) — for Java/C++ TorchScript inference
- **`pytorch_java_libs/`** — `pytorch_java-*.jar`, `libjnitorch.so` for Java JNI

None of this is used by the current **Python host → Rust PyO3** design. Do not restore `lib/`; add `/lib/` to `.gitignore` if needed.

## `archive/legacy/pgl/`

Not a dependency — historical reference only. See [archive/legacy/README.md](../archive/legacy/README.md).
