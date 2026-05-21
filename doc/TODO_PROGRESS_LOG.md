# TODO Progress Log

Mandatory engineering log for CropAbility. Add a new entry after each completed `M1-*` item (or significant partial milestone).

Pair with [`doc/TODO.md`](TODO.md) (checkbox truth source) and [`doc/TODO_PROGRESS_SYNC.md`](TODO_PROGRESS_SYNC.md) (vault / GitHub sync rules).

## Entry template

- Date:
- TODO ID/Title:
- Scope:
- Files Changed:
- Validation Steps:
- Outcome:
- Risks/Follow-ups:

---

## 2026-05-21 (progress tracking bootstrap)

- Date: 2026-05-21
- TODO ID/Title: `M1-DELIV-002` Progress tracking docs
- Scope: Add `doc/TODO.md`, `doc/TODO_PROGRESS_LOG.md`, and `doc/TODO_PROGRESS_SYNC.md` aligned with DBone / SirLearning/script vault sync conventions; seed checklist from current repo state.
- Files Changed:
  - `doc/TODO.md` (new)
  - `doc/TODO_PROGRESS_LOG.md` (new)
  - `doc/TODO_PROGRESS_SYNC.md` (new)
- Validation Steps:
  - Cross-checked against `README.md`, `AGENTS.md`, `doc/TESTING.md`, `doc/RUST_DEVELOPMENT.md`, and git history on `main`.
- Outcome:
  - Checkbox IDs and log entries established for vault sync via GitHub raw (`SirLearning/CropAbility`).
- Risks/Follow-ups:
  - Open `M1-*` items need validation on real GPU/BAM hosts; update log when closed.

## 2026-05-21 (baseline rollup)

- Date: 2026-05-21
- TODO ID/Title: Pre-tracker baseline → M1-FOUND / GPU / GEN / NGS / CLI / TEST / DELIV
- Scope: Record merged work that predates this log (commits through `f47b593`).
- Files Changed: (representative)
  - `src/main/python/cropability/**` — GPU, genomics, ngs facade, CLI
  - `src/main/rust/**` — io, pileup, fastcall3, pipeline, PyO3
  - `src/test/python/**` — pytest suite
  - `install.py`, `environment.yml`, `requirements-*.txt`, `doc/**`
- Validation Steps:
  - `pytest src/test/python` (CPU paths)
  - `maturin develop --release --features python,htslib` + `pytest -m native` (when htslib available)
  - `cargo test` (optional)
- Outcome:
  - **FOUND:** canonical layout, agent rules, install path, native `.so` gitignore.
  - **GPU:** DeviceManager + seq/stats kernels.
  - **GEN:** VariantCaller, LDCalculator, GWASEngine, SmithWatermanGPU.
  - **NGS:** Rust I/O, mpileup, FastCall3-style logic, `cropability.ngs`, CLI `pileup` / `call-variants`.
  - **CLI:** info, benchmark, snp, ld, gwas, align, pileup, call-variants.
  - **TEST:** pytest markers and NGS/CLI tests.
  - **DELIV:** core dev docs under `doc/`.
- Risks/Follow-ups:
  - Historical Evo2 CLI experiments were removed during structure initiation; do not resurrect without explicit product decision.
  - README still shows placeholder clone URL — track under `M1-FOUND-005`.

## 2026-05-21 / M1-FOUND-001

- Date: 2026-05-21
- TODO ID/Title: `M1-FOUND-001` Canonical repo layout
- Scope: Single Python package `cropability`, flat Rust crate, no nested workspaces, tests only under `src/test/python/`.
- Files Changed: repo structure (`1532036 Structure initiation` and follow-ups)
- Validation Steps: `pyproject.toml` includes only `cropability*`; root `Cargo.toml` single crate.
- Outcome: Architecture matches `AGENTS.md` binding rules.
- Risks/Follow-ups: Do not reintroduce Java, TorchScript, or `scripts/`.

## 2026-05-21 / M1-FOUND-002

- Date: 2026-05-21
- TODO ID/Title: `M1-FOUND-002` Agent and English-only policy
- Scope: `AGENTS.md`, `.cursor/rules/*`, English-only docs and comments policy.
- Files Changed: `AGENTS.md`, `.cursor/rules/english-only.mdc`, `.cursor/rules/cropability-project.mdc`
- Validation Steps: Policy files present and referenced from README.
- Outcome: Agent workflow documented for Cursor/Codex.
- Risks/Follow-ups: Keep docs in `doc/` only (no README under `src/`).

## 2026-05-21 / M1-FOUND-003

- Date: 2026-05-21
- TODO ID/Title: `M1-FOUND-003` Install path
- Scope: Conda + pip + maturin one-shot installer and requirement files.
- Files Changed: `install.py`, `environment.yml`, `requirements-gpu.txt`, `requirements-cpu.txt`, `doc/DEPENDENCIES.md`
- Validation Steps: `python install.py` documented in README.
- Outcome: Reproducible env bootstrap for GPU and CPU PyTorch variants.
- Risks/Follow-ups: htslib/samtools remain Conda/system deps for native BAM paths.

## 2026-05-21 / M1-FOUND-004

- Date: 2026-05-21
- TODO ID/Title: `M1-FOUND-004` Native extension git hygiene
- Scope: Stop tracking maturin-built `_core*.so`; rebuild via maturin after clone.
- Files Changed: `.gitignore`, removed tracked `.so` (`f47b593`)
- Validation Steps: `git status` clean after local maturin build.
- Outcome: Wheels/extensions built per platform, not committed.
- Risks/Follow-ups: Document rebuild in every NGS onboarding path.

## 2026-05-21 / M1-NGS-003

- Date: 2026-05-21
- TODO ID/Title: `M1-NGS-003` Native mpileup + FastCall3-style calling
- Scope: In-process mpileup and variant detection without external samtools/FastCall3 binaries.
- Files Changed: `src/main/rust/genomics/pileup.rs`, `fastcall3.rs`, `pipeline.rs`, Python `ngs/*`
- Validation Steps: `pytest src/test/python/test_ngs_pipeline.py -m native` (with maturin build).
- Outcome: Hybrid pipeline default in CLI; Rust owns CPU NGS hot path.
- Risks/Follow-ups: `--mode native` parity still open (`M1-NGS-005`).

## 2026-05-21 / M1-CLI-001

- Date: 2026-05-21
- TODO ID/Title: `M1-CLI-001` Core CLI subcommands
- Scope: User-facing `cropability` entry point and NGS/GPU commands.
- Files Changed: `src/main/python/cropability/cli/main.py`, `pyproject.toml` scripts
- Validation Steps: `pytest src/test/python/test_cli.py`
- Outcome: Documented commands in README Quick start.
- Risks/Follow-ups: Structured output / viz entry points remain open.
