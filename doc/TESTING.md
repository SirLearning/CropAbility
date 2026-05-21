# Testing

CropAbility tests live under **`src/test/python/`** (pytest). There is no separate `scripts/` tree for development checks—anything that validates behavior belongs in pytest.

## Quick start

```bash
pip install -e ".[dev]"
pytest src/test/python
```

NGS / FASTA tests need the Rust extension:

```bash
maturin develop --release --features python,htslib
pytest src/test/python -m native
```

Rust crate (optional):

```bash
cargo test
```

## Python test layout (`src/test/python/`)

| File | Layer | Needs `_core` | Needs GPU |
|------|--------|---------------|-----------|
| `test_config.py` | `utils.config` | No | No |
| `test_gpu_manager.py` | `gpu.device_manager` | No | Optional¹ |
| `test_kernels_seq.py` | `kernels.seq` | No | No (CPU) |
| `test_kernels_stats.py` | `kernels.stats` | No | No (CPU) |
| `test_genomics.py` | GPU `genomics/*` | No | No (CPU) |
| `test_ngs_io.py` | `ngs.io` (FASTA + VCF) | FASTA only | No |
| `test_ngs_pipeline.py` | `ngs.pipeline`, BAM input | Yes | No |
| `test_cli.py` | `cli.main` | No | No |

¹ `test_memory_stats_structure` is skipped when CUDA is unavailable.

Shared fixtures: `conftest.py` (`device`, `gpu_available`, `native_core`, `small_sequences`, …).

## Pytest markers

Defined in `pyproject.toml`:

| Marker | Meaning |
|--------|---------|
| `native` | Requires `cropability.native._core` (maturin build) |
| `gpu` | Requires CUDA |
| `slow` | Long-running (reserved) |

```bash
pytest -m native              # Rust-backed only
pytest -m "not native"        # Pure Python / CPU PyTorch
pytest -m gpu                 # GPU-only (when marked)
pytest --cov=cropability      # coverage (see pyproject omit rules)
```

## What is not in `src/test/python`

| Area | Location |
|------|----------|
| Rust unit / integration tests | `cargo test` (root `Cargo.toml`) |
| Legacy PGL reference code | `archive/legacy/pgl/` (not installed, not tested in CI by default) |

## Adding tests

- **GPU / kernels / genomics** → new `test_*.py` or cases in existing files under `src/test/python/`.
- **NGS / Rust facade** → mark with `@pytest.mark.native`; use `native_core` fixture when needed.
- **CLI** → extend `test_cli.py`.
- Do not add one-off runnable scripts under `scripts/`; use pytest with `@pytest.mark.slow` or `@pytest.mark.gpu` for heavy checks.

## Publishable package scope

Only `cropability` (under `src/main/python/cropability/`) is installed via `pyproject.toml`.  
`src/test/`, `archive/`, and `doc/` are development and documentation assets, not shipped in the wheel.
