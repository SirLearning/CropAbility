# CropAbility — project TODO

Master execution checklist for the **CropAbility** publishable genomics toolkit (Python host + Rust `_core`). Completed items stay checked for audit. Log every completion in [`doc/TODO_PROGRESS_LOG.md`](TODO_PROGRESS_LOG.md).

**Authority:** This file is the checkbox truth source for vault sync (see [`doc/TODO_PROGRESS_SYNC.md`](TODO_PROGRESS_SYNC.md)).

**Structure:** Sections follow the product architecture — foundation → GPU → genomics → NGS (Rust) → CLI → testing → delivery → backlog.

---

## ID prefix reference

| Prefix | Area |
|--------|------|
| `M1-FOUND-*` | Packaging, install, config, agent rules |
| `M1-GPU-*` | `cropability.gpu`, `cropability.kernels` |
| `M1-GEN-*` | GPU `cropability.genomics` |
| `M1-NGS-*` | Rust I/O + pipeline; Python `cropability.ngs` facade |
| `M1-CLI-*` | `cropability` CLI |
| `M1-TEST-*` | Pytest, `cargo test`, CI |
| `M1-DELIV-*` | Docs, release, publish gates |
| `M1-BACKLOG-*` | Post-M1 or cross-cutting research items |

## Traceability rules

- Every item has a stable ID: `M1-<AREA>-<NNN>`.
- Mark `[x]` only when the change is merged in this repository.
- Every completed item must have one entry in `doc/TODO_PROGRESS_LOG.md`.
- Use `LogRef` on the checklist line to point to that log entry (date and title).

---

## 1. Foundation and packaging — [[doc/DEPENDENCIES]]

- [x] `M1-FOUND-001` Canonical repo layout: Python host (`src/main/python/cropability/`), flat Rust crate (`src/main/rust/`), pytest under `src/test/python/` | LogRef: 2026-05-21 / M1-FOUND-001
- [x] `M1-FOUND-002` Agent and Cursor rules (`AGENTS.md`, `.cursor/rules/`), English-only policy | LogRef: 2026-05-21 / M1-FOUND-002
- [x] `M1-FOUND-003` One-shot install path (`install.py`, `environment.yml`, `requirements-gpu.txt` / `requirements-cpu.txt`) | LogRef: 2026-05-21 / M1-FOUND-003
- [x] `M1-FOUND-004` Git hygiene: ignore maturin-built `native/_core*.so`; document maturin rebuild | LogRef: 2026-05-21 / M1-FOUND-004
- [ ] `M1-FOUND-005` Confirm public GitHub remote and README clone URL (`SirLearning/CropAbility`) | LogRef: pending
- [ ] `M1-FOUND-006` Runtime config guide: `cropability.yaml` + `CROPABILITY_*` env vars with validation tests | LogRef: pending

---

## 2. GPU compute — [[doc/PYTHON_DEVELOPMENT]]

- [x] `M1-GPU-001` `DeviceManager` (H100/A2 detection, memory stats, primary device selection) | LogRef: 2026-05-21 / M1-GPU-001
- [x] `M1-GPU-002` Triton/PyTorch kernels: sequence encode, GC, stats (Welford, z-score, correlation, pairwise) | LogRef: 2026-05-21 / M1-GPU-002
- [ ] `M1-GPU-003` Multi-GPU DDP: document and pytest-smoke `launch_ddp` / `wrap_ddp` on dual-GPU hosts | LogRef: pending
- [ ] `M1-GPU-004` Kernel correctness vs CPU reference: expand benchmarks in pytest (`slow` marker) | LogRef: pending

---

## 3. GPU genomics — [[doc/PYTHON_DEVELOPMENT]]

- [x] `M1-GEN-001` Core GPU modules: `VariantCaller`, `LDCalculator`, `GWASEngine` | LogRef: 2026-05-21 / M1-GEN-001
- [x] `M1-GEN-002` `SmithWatermanGPU` batch scoring API | LogRef: 2026-05-21 / M1-GEN-002
- [ ] `M1-GEN-003` End-to-end GPU SNP/LD/GWAS CLI paths validated on CUDA hardware (`@pytest.mark.gpu`) | LogRef: pending
- [ ] `M1-GEN-004` Integration hooks for variation-library cohort outputs (VCF/PLINK ingest helpers) | LogRef: pending

---

## 4. NGS / native extension — [[doc/RUST_DEVELOPMENT]]

- [x] `M1-NGS-001` Rust I/O: FASTA, BAM (htslib), VCF; PyO3 exports in `python.rs` | LogRef: 2026-05-21 / M1-NGS-001
- [x] `M1-NGS-002` Thin Python facade `cropability.ngs` (pipeline, pileup, fastcall3, io) | LogRef: 2026-05-21 / M1-NGS-002
- [x] `M1-NGS-003` Native in-process mpileup + FastCall3-style variant logic | LogRef: 2026-05-21 / M1-NGS-003
- [x] `M1-NGS-004` CLI: `pileup`, `call-variants` (`--mode hybrid`) | LogRef: 2026-05-21 / M1-NGS-004
- [ ] `M1-NGS-005` `--mode native` parity and performance vs hybrid on real BAM fixtures | LogRef: pending
- [ ] `M1-NGS-006` CRAM input smoke + documented limitations | LogRef: pending
- [ ] `M1-NGS-007` Deprecation timeline for shims (`cropability.io`, `cropability.genomics.pipeline|pileup|fastcall3`) | LogRef: pending

---

## 5. CLI and visualization

- [x] `M1-CLI-001` Core subcommands: `info`, `benchmark`, `snp`, `ld`, `gwas`, `align`, `pileup`, `call-variants` | LogRef: 2026-05-21 / M1-CLI-001
- [ ] `M1-CLI-002` `cropability viz` or documented `cropability[viz]` plotting entry points | LogRef: pending
- [ ] `M1-CLI-003` `--version` / structured JSON output for automation | LogRef: pending

---

## 6. Testing — [[doc/TESTING]]

- [x] `M1-TEST-001` Pytest layout, markers (`native`, `gpu`, `slow`), `conftest.py` fixtures | LogRef: 2026-05-21 / M1-TEST-001
- [x] `M1-TEST-002` NGS + CLI parsing coverage (`test_ngs_*`, `test_cli.py`) | LogRef: 2026-05-21 / M1-TEST-002
- [ ] `M1-TEST-003` GitHub Actions: CPU pytest + optional `cargo test` on push/PR | LogRef: pending
- [ ] `M1-TEST-004` Checked-in or downloadable minimal BAM/FASTA fixture set for `@pytest.mark.native` | LogRef: pending
- [ ] `M1-TEST-005` GPU CI job or documented manual GPU test checklist | LogRef: pending

---

## 7. Delivery and documentation

- [x] `M1-DELIV-001` Dev docs: `TESTING.md`, `RUST_DEVELOPMENT.md`, `PYTHON_DEVELOPMENT.md`, `DEPENDENCIES.md` | LogRef: 2026-05-21 / M1-DELIV-001
- [x] `M1-DELIV-002` Progress tracking docs: `TODO.md`, `TODO_PROGRESS_LOG.md`, `TODO_PROGRESS_SYNC.md` | LogRef: 2026-05-21 / M1-DELIV-002
- [ ] `M1-DELIV-003` PyPI / wheel publish checklist (maturin manylinux, optional extras matrix) | LogRef: pending
- [ ] `M1-DELIV-004` Release gate for `0.2.0`: API stability review + changelog | LogRef: pending

---

## 8. Backlog (post-M1)

- [ ] `M1-BACKLOG-001` GPU acceleration for additional genomics kernels (k-mer, alignment at scale)
- [ ] `M1-BACKLOG-002` Tighter coupling with SirLearning/script variation-library outputs (assess/filter params)
- [ ] `M1-BACKLOG-003` Optional distributed multi-node execution story (beyond single-node DDP)
