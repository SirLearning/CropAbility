# Agent instructions (CropAbility)

CropAbility is a **publishable** plant-genomics software product: a Python package with an optional Rust native extension. Agents must preserve this architecture and avoid reintroducing removed patterns.

## Product model

| Concern | Decision |
|---------|----------|
| **Host runtime** | Python (PyTorch / Triton for GPU) |
| **Native extension** | Rust via maturin → `cropability.native._core` |
| **Shipped to users** | Only `cropability` from `src/main/python/cropability/` (wheel/sdist) |
| **User entry point** | `cropability` CLI (`pyproject.toml` `[project.scripts]`) |
| **Not shipped** | `src/test/`, `doc/`, `archive/`, `environment*.yml`, root `Cargo.toml` build artifacts |

## Architecture (binding)

**Python host → Rust extension.** No Java. No TorchScript. No libtorch in Rust. No Rust embedding Python for PyTorch.

```
src/main/python/cropability/     GPU: gpu/, kernels/, genomics/, viz/, cli/
cropability/ngs/                 CPU/NGS thin facade → native._core
src/main/rust/                   CPU: io/, genomics/, python.rs (PyO3)
src/main/resources/              Runtime config; private/ gitignored
```

- **GPU** inference/training: Python only (`torch`, optional `triton`).
- **CPU / NGS** (FASTA, BAM, pileup, variant pipeline): Rust implementation; Python `ngs/` is a thin wrapper.
- **Resources**: `src/main/resources/` only (not repo-root `resources/`). Model checkpoints are **not** vendored in the repo.

## Build

| File | Role |
|------|------|
| `pyproject.toml` | Python package, pytest, maturin config — **do not merge with Cargo** |
| `Cargo.toml` | **Single** crate `cropability-native` at repo root |
| `environment.yml` | Minimal Conda base (recommended) |
| `install.py` | One-shot Conda + pip + maturin |
| `requirements-gpu.txt` / `requirements-cpu.txt` | PyTorch + editable extras |

```bash
python install.py              # GPU
# python install.py --cpu      # CPU-only PyTorch
conda activate cropability
pytest src/test/python
cargo test   # optional
```

## Testing & docs (where things live)

| Content | Location | Never put here |
|---------|----------|----------------|
| Pytest suite | `src/test/python/` | `scripts/`, `src/test/python/README.md` |
| Install, dependencies, runtime | `README.md` | Extra guides under `doc/` |
| Progress index / daily log | `doc/PROGRESS_README.md`, `doc/progress/YYYY-MM-DD.md` | Other files under `doc/` |
| Legacy PGL reference | `archive/legacy/pgl/` | Install tree or `scripts/` |

- All validation → **pytest** in `src/test/python/` (`native`, `gpu`, `slow` markers).
- No `scripts/` folder; no one-off dev scripts in the repo root.

## Cursor rules (apply all)

| File | When |
|------|------|
| [`.cursor/rules/cropability-project.mdc`](.cursor/rules/cropability-project.mdc) | Always |
| [`.cursor/rules/english-only.mdc`](.cursor/rules/english-only.mdc) | Always |
| [`.cursor/rules/cropability-python.mdc`](.cursor/rules/cropability-python.mdc) | `src/main/python/cropability/**` |
| [`.cursor/rules/cropability-rust.mdc`](.cursor/rules/cropability-rust.mdc) | `src/main/rust/**` |
| [`.cursor/rules/cropability-tests.mdc`](.cursor/rules/cropability-tests.mdc) | `src/test/python/**` |
| [`.cursor/rules/cropability-docs.mdc`](.cursor/rules/cropability-docs.mdc) | `doc/**` |
| [`.cursor/rules/progress-logging.mdc`](.cursor/rules/progress-logging.mdc) | Always (logging + `doc/` layout) |

For non-Cursor agents: read every `.mdc` file; the markdown body below each YAML frontmatter block is binding.

## Do not reintroduce

- Java, `pom.xml`, `src/main/java`, `src/test/java`
- TorchScript export, `cropability/models/`, CLI `export` for `.pt`, `.pt` artifacts for Rust
- Rust `tch` / libtorch, `torch` Cargo feature, `integration_test`, `simulated.rs`
- Root `resources/` or repo `lib/` (vendored libtorch / Java JNI)
- `scripts/` or dev benchmark scripts outside pytest
- Nested Rust workspaces, `src/test/rust/Cargo.toml`, `src/main/rust/src/`, child `Cargo.toml` under `rust/`
- README or architecture docs under `src/test/` or `src/`
- Fat CPU/NGS logic duplicated in Python (belongs in Rust + `ngs/` facade)
- Rust calling Python to run PyTorch

## Agent behavior

- Follow **english-only** for docs, comments, CLI help, and rules.
- **Minimal diffs**; do not refactor `archive/legacy/` unless asked.
- Do not commit `src/main/resources/private/` or secrets.
- Do not `git commit` unless the user explicitly asks.
- New features: GPU → Python; CPU/NGS → Rust + thin `ngs/`; tests → `src/test/python/`; user docs → `README.md`; progress → append today's `doc/progress/YYYY-MM-DD.md` (`progress-logging.mdc`).
- Task planning is outside the repo; do not recreate `doc/TODO.md`.
