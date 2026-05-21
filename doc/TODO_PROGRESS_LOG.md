# TODO Progress Log

Mandatory engineering log for CropAbility. Add a new entry after each completed
`CA-*` item in [`doc/TODO.md`](TODO.md) (or significant partial milestone).

Pair with [`doc/TODO_PROGRESS_SYNC.md`](TODO_PROGRESS_SYNC.md) (vault / GitHub sync
rules). Historical flat `M1-*` IDs remain in parentheses for entries written before
the vault-aligned checklist; see **ID scheme and historical migration** below.

## ID scheme and historical migration

**Current (2026-05-21):** Checkbox truth in `doc/TODO.md` uses `CA-{section}-{item}`
aligned with the dated vault TODO source `2026-05-21.md` → `## CropAbility`.
Parent items end in `-00`; leaf items end in `-01`, `-02`, and so on.

**Historical:** The first repository log used flat `M1-{AREA}-{nnn}` IDs (FOUND,
GPU, GEN, NGS, CLI, TEST, DELIV). Those IDs are **retired for new checkboxes** but
preserved here for traceability. Vault canonical mapping lives in
`CropAbility.md` §1.1.1 Historical ID migration.

### M1 → CA mapping (repository completions)

| Historical `M1-*` | Current `CA-*` | Title | Log heading |
|-------------------|----------------|-------|-------------|
| `M1-FOUND-001` | `CA-2211-01` | Canonical repository layout | 2026-05-21 / CA-2211-01 |
| `M1-FOUND-002` | `CA-2211-02` | Agent and English-only policy | 2026-05-21 / CA-2211-02 |
| `M1-FOUND-003` | `CA-2211-03` | One-shot installation path | 2026-05-21 / CA-2211-03 |
| `M1-FOUND-004` | `CA-2211-04` | Native `.so` git hygiene | 2026-05-21 / CA-2211-04 |
| `M1-FOUND-005` | `CA-2211-05` | Confirm public GitHub remote and README clone URL | *(open — no log yet)* |
| `M1-NGS-001` | `CA-2211-07` | Rust I/O plus PyO3 native extension | baseline rollup only |
| `M1-TEST-001` | `CA-2211-08` | Pytest layout and markers | baseline rollup only |
| `M1-GPU-001` | `CA-2212-01` | DeviceManager | baseline rollup only |
| `M1-GPU-002` | `CA-2212-02` | Triton/PyTorch kernels | baseline rollup only |
| `M1-DELIV-001` | `CA-2213-01` | Development documentation suite | baseline rollup only |
| `M1-DELIV-002` | `CA-2213-02` | Progress tracking documentation | 2026-05-21 / CA-2213-02 |
| `M1-CLI-001` | `CA-2223-21` | CLI pileup / call-variants | 2026-05-21 / CA-2223-21 |
| `M1-NGS-004` | `CA-2223-21` | CLI pileup / call-variants *(same leaf)* | 2026-05-21 / CA-2223-21 |
| `M1-NGS-003` | `CA-2223-22` | In-process mpileup plus FastCall3-style calling | 2026-05-21 / CA-2223-22 |
| `M1-GEN-001` | `CA-223-01` | Variant/LD/GWAS engine | baseline rollup only |
| `M1-GEN-002` | `CA-223-02` | SmithWatermanGPU | baseline rollup only |
| `M1-NGS-005` | *(none)* | Native-mode CLI parity *(retired ID)* | track under open `CA-2223-*` work |

Vault-only completions (no repository log body required; `LogRef: vault` in
`doc/TODO.md`): `CA-21-01`, `CA-2221-01`, `CA-2222-00` … `CA-2222-07`.

### Architecture realignment (`doc/TODO.md` sections)

| Old flat bucket | New checklist location | Example `CA-*` |
|-----------------|------------------------|----------------|
| M1-FOUND | §2.1.1 Foundation | `CA-2211-01` … `CA-2211-08` |
| M1-GPU | §2.1.2 GPU | `CA-2212-01` … `CA-2212-04` |
| M1-DELIV | §2.1.3 Delivery | `CA-2213-01` … `CA-2213-04` |
| M1-NGS, M1-CLI | §2.2.3 FastCall3 Rust rewrite | `CA-2223-21`, `CA-2223-22` |
| M1-GEN | §2.3 Personal genetics and breeding | `CA-223-01` … `CA-223-03` |
| *(new)* | §2.4 Visualization | `CA-224-01` |
| *(vault history)* | §1 FlagOS; §2.2.1–2.2.2 JNI/TorchScript explorations | `CA-21-01`, `CA-2221-*`, `CA-2222-*` |

**LogRef convention:** `doc/TODO.md` points to the **primary `CA-*` slug** in this
file (e.g. `LogRef: 2026-05-21 / CA-2211-01`). Include `(M1-…)` in the log heading
only when a historical ID exists.

## Entry template

- Date:
- TODO ID/Title: `CA-…` (historical `M1-…` if applicable)
- Scope:
- Files Changed:
- Validation Steps:
- Outcome:
- Risks/Follow-ups:

---

## 2026-05-21 / CA-2213-02 — checklist ID migration

- Date: 2026-05-21
- TODO ID/Title: `CA-2213-02` Progress tracking docs — ID scheme migration (historical `M1-DELIV-002`)
- Scope:
  - Realign `doc/TODO.md` with vault `2026-05-21.md` → `## CropAbility` using hierarchical `CA-*` IDs.
  - Retire flat `M1-*` as the checkbox ID scheme; document full `M1-*` → `CA-*` mapping in this log and in [`doc/TODO_PROGRESS_SYNC.md`](TODO_PROGRESS_SYNC.md).
  - Remap checklist sections: Foundation / GPU / Delivery under §2.1 architecture; NGS+CLI under §2.2.3 Rust FastCall rewrite; genomics under §2.3; visualization under §2.4.
- Files Changed:
  - `doc/TODO.md` (modified — `CA-*` IDs, vault traceability rules, LogRefs)
  - `doc/TODO_PROGRESS_LOG.md` (modified — migration table, dual-ID headings)
  - `doc/TODO_PROGRESS_SYNC.md` (modified — vault mapping, templates, agent steps)
  - `.cursor/rules/cropability-docs.mdc` (modified — progress logging rules)
- Validation Steps:
  - Cross-checked every `[x]` item in `doc/TODO.md` against a log heading or `LogRef: vault`.
  - Verified section numbers in `TODO_PROGRESS_SYNC.md` match `TODO.md` §2 hierarchy.
- Outcome:
  - `CA-*` is the authoritative checkbox ID; `M1-*` preserved in migration tables and historical log slugs.
  - Vault sync agents should copy `CA-*` IDs from `doc/TODO.md` and cite `CA-*` LogRef slugs from this file.
- Risks/Follow-ups:
  - Items covered only by the baseline rollup (`CA-2211-07`, `CA-2211-08`, `CA-2212-*`, `CA-2213-01`, `CA-223-*`) should get dedicated log entries when next touched.
  - Open `CA-2211-05` (README clone URL) replaces follow-up formerly noted as `M1-FOUND-005`.
  - Retired `M1-NGS-005` (native-mode parity) has no `CA-*` leaf yet — track under §2.2.3 open work.

## 2026-05-21 / CA-2213-02 (M1-DELIV-002) — progress tracking bootstrap

- Date: 2026-05-21
- TODO ID/Title: `CA-2213-02` Progress tracking documentation (historical `M1-DELIV-002`)
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
  - Open `CA-*` items need validation on real GPU/BAM hosts; update log when closed.

## 2026-05-21 (baseline rollup)

- Date: 2026-05-21
- TODO ID/Title: Pre-tracker baseline → `CA-2211` / `CA-2212` / `CA-2223` / `CA-223` / `CA-2213` (historical M1-FOUND / GPU / GEN / NGS / CLI / TEST / DELIV)
- Scope: Record merged work that predates this log (commits through `f47b593`).
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
  - **`CA-2211` Foundation:** `CA-2211-01` layout, `CA-2211-02` agent rules, `CA-2211-03` install, `CA-2211-04` native `.so` gitignore, `CA-2211-07` Rust I/O + PyO3, `CA-2211-08` pytest markers.
  - **`CA-2212` GPU:** `CA-2212-01` DeviceManager, `CA-2212-02` seq/stats Triton/PyTorch kernels.
  - **`CA-223` Genetics:** `CA-223-01` VariantCaller / LDCalculator / GWASEngine, `CA-223-02` SmithWatermanGPU.
  - **`CA-2223` NGS:** `CA-2223-22` in-process mpileup + FastCall3-style logic, `CA-2223-21` CLI `pileup` / `call-variants`; Rust owns CPU NGS hot path via `cropability.ngs`.
  - **`CA-2213` Delivery:** `CA-2213-01` core dev docs under `doc/`.
  - **CLI (rollup):** info, benchmark, snp, ld, gwas, align, pileup, call-variants.
- Risks/Follow-ups:
  - Historical Evo2 CLI experiments were removed during structure initiation; do not resurrect without explicit product decision.
  - README still shows placeholder clone URL — track under `CA-2211-05`.

## 2026-05-21 / CA-2211-01 (M1-FOUND-001)

- Date: 2026-05-21
- TODO ID/Title: `CA-2211-01` Canonical repository layout (historical `M1-FOUND-001`)
- Scope: Single Python package `cropability`, flat Rust crate, no nested workspaces, tests only under `src/test/python/`.
- Files Changed: repo structure (`1532036 Structure initiation` and follow-ups)
- Validation Steps: `pyproject.toml` includes only `cropability*`; root `Cargo.toml` single crate — pass
- Outcome: Architecture matches `AGENTS.md` binding rules.
- Risks/Follow-ups: Do not reintroduce Java, TorchScript, or `scripts/`.

## 2026-05-21 / CA-2211-02 (M1-FOUND-002)

- Date: 2026-05-21
- TODO ID/Title: `CA-2211-02` Agent and English-only policy (historical `M1-FOUND-002`)
- Scope: `AGENTS.md`, `.cursor/rules/*`, English-only docs and comments policy.
- Files Changed: `AGENTS.md`, `.cursor/rules/english-only.mdc`, `.cursor/rules/cropability-project.mdc`
- Validation Steps: Policy files present and referenced from README — pass
- Outcome: Agent workflow documented for Cursor/Codex.
- Risks/Follow-ups: Keep docs in `doc/` only (no README under `src/`).

## 2026-05-21 / CA-2211-03 (M1-FOUND-003)

- Date: 2026-05-21
- TODO ID/Title: `CA-2211-03` One-shot installation path (historical `M1-FOUND-003`)
- Scope: Conda + pip + maturin one-shot installer and requirement files.
- Files Changed: `install.py`, `environment.yml`, `requirements-gpu.txt`, `requirements-cpu.txt`, `doc/DEPENDENCIES.md`
- Validation Steps: `python install.py` documented in README — pass (documented)
- Outcome: Reproducible env bootstrap for GPU and CPU PyTorch variants.
- Risks/Follow-ups: htslib/samtools remain Conda/system deps for native BAM paths.

## 2026-05-21 / CA-2211-04 (M1-FOUND-004)

- Date: 2026-05-21
- TODO ID/Title: `CA-2211-04` Native extension git hygiene (historical `M1-FOUND-004`)
- Scope: Stop tracking maturin-built `_core*.so`; rebuild via maturin after clone.
- Files Changed: `.gitignore`, removed tracked `.so` (`f47b593`)
- Validation Steps: `git status` clean after local maturin build — pass
- Outcome: Wheels/extensions built per platform, not committed.
- Risks/Follow-ups: Document rebuild in every NGS onboarding path.

## 2026-05-21 / CA-2223-22 (M1-NGS-003)

- Date: 2026-05-21
- TODO ID/Title: `CA-2223-22` In-process mpileup plus FastCall3-style calling (historical `M1-NGS-003`)
- Scope: In-process mpileup and variant detection without external samtools/FastCall3 binaries.
- Files Changed: `src/main/rust/genomics/pileup.rs`, `fastcall3.rs`, `pipeline.rs`, Python `ngs/*`
- Validation Steps: `pytest src/test/python/test_ngs_pipeline.py -m native` (with maturin build) — pass when built
- Outcome: Hybrid pipeline default in CLI; Rust owns CPU NGS hot path.
- Risks/Follow-ups: Native-mode CLI parity has no `CA-*` leaf yet (formerly `M1-NGS-005`); track under §2.2.3 open work.

## 2026-05-21 / CA-2223-21 (M1-CLI-001, M1-NGS-004)

- Date: 2026-05-21
- TODO ID/Title: `CA-2223-21` CLI pileup / call-variants (historical `M1-CLI-001`, `M1-NGS-004`)
- Scope: User-facing `cropability` entry point and NGS/GPU commands (pileup, call-variants, and related subcommands).
- Files Changed: `src/main/python/cropability/cli/main.py`, `pyproject.toml` scripts
- Validation Steps: `pytest src/test/python/test_cli.py` — pass
- Outcome: Documented commands in README Quick start.
- Risks/Follow-ups: Structured output / viz entry points remain open (`CA-224-01`).
