# TODO Progress Log

Mandatory engineering log for CropAbility. Add a new entry after each completed
item in [`doc/TODO.md`](TODO.md) (or a significant partial milestone).

## Headings

- **Format:** `## YYYY-MM-DD — short English title` (newest entries at the top after this
  intro block, or append at the bottom — stay consistent within the file).
- Match checklist titles in [`doc/TODO.md`](TODO.md) by meaning, not by a separate slug or
  cross-reference field.

## Entry template

- Date:
- Title:
- Scope:
- Files Changed:
- Validation Steps:
- Outcome:
- Risks/Follow-ups:

---

## 2026-06-04 — documentation consolidation

- Date: 2026-06-04
- Title: Consolidate doc/ into README plus progress log
- Scope:
  - Move install, dependencies, runtime, and Rust extension usage into `README.md`.
  - Retire `doc/DEPENDENCIES.md`, `doc/PYTHON_DEVELOPMENT.md`, `doc/RUST_DEVELOPMENT.md`, `doc/TESTING.md`.
  - Keep only `doc/TODO.md` and `doc/TODO_PROGRESS_LOG.md` under `doc/`.
  - Record pytest layout and conventions here (formerly `doc/TESTING.md`).
- Files Changed:
  - `README.md` (modified — install, dependencies, Rust API, testing commands)
  - `doc/DEPENDENCIES.md`, `doc/PYTHON_DEVELOPMENT.md`, `doc/RUST_DEVELOPMENT.md`, `doc/TESTING.md` (deleted)
  - `AGENTS.md`, `.cursor/rules/cropability-project.mdc`, `cropability-docs.mdc`, `cropability-tests.mdc`, `english-only.mdc`, `cropability-python.mdc` (modified — paths)
- Validation Steps:
  - Grep confirmed no remaining links to deleted `doc/*.md` guides.
- Outcome:
  - User-facing install/run docs live in `README.md`.
  - `doc/` holds checklist + engineering log only.
- Risks/Follow-ups:
  - External bookmarks to removed `doc/TESTING.md` etc. need updating to README or this log.

### Pytest layout and conventions (retained reference)

**Location:** `src/test/python/` only; shared `conftest.py`; no `scripts/`; Rust tests via `cargo test`.

| File | Layer | Needs `_core` | Needs GPU |
|------|--------|---------------|-----------|
| `test_config.py` | `utils.config` | No | No |
| `test_gpu_manager.py` | `gpu.device_manager` | No | Optional¹ |
| `test_kernels_seq.py` | `kernels.seq` | No | No (CPU) |
| `test_kernels_stats.py` | `kernels.stats` | No | No (CPU) |
| `test_genomics.py` | GPU `genomics/*` | No | No (CPU) |
| `test_ngs_io.py` | `ngs.io` | FASTA only | No |
| `test_ngs_pipeline.py` | `ngs.pipeline`, BAM | Yes | No |
| `test_cli.py` | `cli.main` | No | No |

¹ `test_memory_stats_structure` skipped when CUDA unavailable.

**Markers (`pyproject.toml`):** `native`, `gpu`, `slow` — use `@pytest.mark.native` and `native_core` fixture for NGS; `@pytest.mark.gpu` when CUDA required.

**Adding tests:** GPU/kernels → `test_*.py` under `src/test/python/`; NGS → `cropability.ngs`, not deprecated shims; CLI → `test_cli.py`; no TorchScript/`.pt` tests unless product scope changes.

**Not shipped:** `src/test/`, `archive/`, `doc/` are not in the wheel.

## 2026-05-21 — progress tracking simplification

- Date: 2026-05-21
- Title: Progress tracking — drop task IDs; use date + short English titles
- Scope:
  - Simplify `doc/TODO.md` checkboxes to descriptive English titles only (no `CA-*` / `M1-*`).
  - Remove ID scheme and migration tables from progress docs and Cursor rules.
- Files Changed:
  - `doc/TODO.md` (modified — titles only)
  - `doc/TODO_PROGRESS_LOG.md` (modified — this file)
  - `.cursor/rules/cropability-docs.mdc` (modified — progress logging rules)
- Validation Steps:
  - Cross-checked every `[x]` item in `doc/TODO.md` against a log heading where repo work applies.
- Outcome:
  - Checklist and log are readable without decoding hierarchical IDs.
- Risks/Follow-ups:
  - Items covered only by the baseline rollup should get dedicated log entries when next touched.
  - Open “Confirm public GitHub remote and README clone URL” still needs a log entry when closed.
  - Native-mode CLI parity has no dedicated checklist leaf yet — track under §2.2.3 open work.

## 2026-05-21 — progress tracking bootstrap

- Date: 2026-05-21
- Title: Progress tracking documentation
- Scope: Add `doc/TODO.md` and `doc/TODO_PROGRESS_LOG.md`; seed checklist from current repo state.
- Files Changed:
  - `doc/TODO.md` (new)
  - `doc/TODO_PROGRESS_LOG.md` (new)
- Validation Steps:
  - Cross-checked against `README.md`, `AGENTS.md`, and git history on `main`.
- Outcome:
  - Checklist and engineering log established under `doc/`.
- Risks/Follow-ups:
  - Open checklist items need validation on real GPU/BAM hosts; update log when closed.

## 2026-05-21 — baseline rollup

- Date: 2026-05-21
- Title: Pre-tracker baseline (Foundation, GPU, NGS, genetics, delivery, CLI)
- Scope: Record merged work that predates per-item log entries (commits through `f47b593`).
- Files Changed: (representative)
  - `src/main/python/cropability/**` — GPU, genomics, ngs facade, CLI
  - `src/main/rust/**` — io, pileup, fastcall3, pipeline, PyO3
  - `src/test/python/**` — pytest suite
  - `install.py`, `environment.yml`, `requirements-*.txt`, `doc/**`
- Validation Steps:
  - `pytest src/test/python` (CPU paths) — pass
  - `maturin develop --release --features python,htslib` + `pytest -m native` (when htslib available) — pass when built
  - `cargo test` (optional) — pass
- Outcome:
  - **Foundation:** layout, agent rules, install, native `.so` gitignore, Rust I/O + PyO3, pytest markers.
  - **GPU:** DeviceManager, seq/stats Triton/PyTorch kernels.
  - **Genetics:** VariantCaller / LDCalculator / GWASEngine, SmithWatermanGPU.
  - **NGS:** in-process mpileup + FastCall3-style logic, CLI `pileup` / `call-variants`; Rust owns CPU NGS hot path via `cropability.ngs`.
  - **Delivery:** user docs in `README.md`; tracking under `doc/`.
  - **CLI (rollup):** info, benchmark, snp, ld, gwas, align, pileup, call-variants.
- Risks/Follow-ups:
  - Historical Evo2 CLI experiments were removed during structure initiation; do not resurrect without explicit product decision.
  - README still shows placeholder clone URL — track under open Foundation item.

## 2026-05-21 — canonical repository layout

- Date: 2026-05-21
- Title: Canonical repository layout
- Scope: Single Python package `cropability`, flat Rust crate, no nested workspaces, tests only under `src/test/python/`.
- Files Changed: repo structure (`1532036 Structure initiation` and follow-ups)
- Validation Steps: `pyproject.toml` includes only `cropability*`; root `Cargo.toml` single crate — pass
- Outcome: Architecture matches `AGENTS.md` binding rules.
- Risks/Follow-ups: Do not reintroduce Java, TorchScript, or `scripts/`.

## 2026-05-21 — agent and English-only policy

- Date: 2026-05-21
- Title: Agent and English-only policy
- Scope: `AGENTS.md`, `.cursor/rules/*`, English-only docs and comments policy.
- Files Changed: `AGENTS.md`, `.cursor/rules/english-only.mdc`, `.cursor/rules/cropability-project.mdc`
- Validation Steps: Policy files present and referenced from README — pass
- Outcome: Agent workflow documented for Cursor/Codex.
- Risks/Follow-ups: Keep docs in `doc/` only (no README under `src/`).

## 2026-05-21 — one-shot installation path

- Date: 2026-05-21
- Title: One-shot installation path
- Scope: Conda + pip + maturin one-shot installer and requirement files.
- Files Changed: `install.py`, `environment.yml`, `requirements-gpu.txt`, `requirements-cpu.txt`, `README.md` (dependencies section)
- Validation Steps: `python install.py` documented in README — pass (documented)
- Outcome: Reproducible env bootstrap for GPU and CPU PyTorch variants.
- Risks/Follow-ups: htslib/samtools remain Conda/system deps for native BAM paths.

## 2026-05-21 — native extension git hygiene

- Date: 2026-05-21
- Title: Native extension git hygiene
- Scope: Stop tracking maturin-built `_core*.so`; rebuild via maturin after clone.
- Files Changed: `.gitignore`, removed tracked `.so` (`f47b593`)
- Validation Steps: `git status` clean after local maturin build — pass
- Outcome: Wheels/extensions built per platform, not committed.
- Risks/Follow-ups: Document rebuild in every NGS onboarding path.

## 2026-05-21 — in-process mpileup and FastCall3-style calling

- Date: 2026-05-21
- Title: In-process mpileup plus FastCall3-style calling
- Scope: In-process mpileup and variant detection without external samtools/FastCall3 binaries.
- Files Changed: `src/main/rust/genomics/pileup.rs`, `fastcall3.rs`, `pipeline.rs`, Python `ngs/*`
- Validation Steps: `pytest src/test/python/test_ngs_pipeline.py -m native` (with maturin build) — pass when built
- Outcome: Hybrid pipeline default in CLI; Rust owns CPU NGS hot path.
- Risks/Follow-ups: Native-mode CLI parity has no dedicated checklist leaf yet; track under §2.2.3 open work.

## 2026-05-21 — CLI pileup and call-variants

- Date: 2026-05-21
- Title: CLI pileup / call-variants
- Scope: User-facing `cropability` entry point and NGS/GPU commands (pileup, call-variants, and related subcommands).
- Files Changed: `src/main/python/cropability/cli/main.py`, `pyproject.toml` scripts
- Validation Steps: `pytest src/test/python/test_cli.py` — pass
- Outcome: Documented commands in README Quick start.
- Risks/Follow-ups: Structured output / viz entry points remain open (Visualization section).
