# CropAbility - project TODO

Master execution checklist for the **CropAbility** publishable genomics toolkit
(Python host + Rust `_core`). This checklist follows the dated vault TODO source:
`2026-05-21.md` -> `## CropAbility`.

**Authority:** This file is the repository checkbox truth source for vault sync
(see [`doc/TODO_PROGRESS_SYNC.md`](TODO_PROGRESS_SYNC.md)).

**ID scheme:** Current task IDs use `CA-{section}-{item}`. Parent items end in
`-00`; leaf items end in `-01`, `-02`, and so on.

**Historical mapping:** The previous repository `M1-*` IDs map to current
`CA-*` IDs in the vault note `CropAbility.md`, section
`1.1.1 Historical ID migration`.

**Traceability rules:**

- Mark `[x]` only for items checked in the dated vault TODO source.
- Repository engineering completions should keep a `LogRef` to
  `doc/TODO_PROGRESS_LOG.md` when one exists (heading slug `YYYY-MM-DD / CA-…`,
  or `baseline rollup` for items covered only by the pre-tracker rollup).
- Historical exploration items that are complete only in the vault use
  `LogRef: vault`.

---

## 1. FlagOS operator competition

- [x] `CA-21-01` FlagOS operator development competition | LogRef: vault

---

## 2. CropAbility GPU-enabled bioinformatics toolkit

- [ ] `CA-22-00` Integrate GPU computing into the CropAbility bioinformatics toolkit

### 2.1 Project architecture

- [ ] `CA-221-00` Project architecture

#### 2.1.1 Foundation

- [ ] `CA-2211-00` Foundation
- [x] `CA-2211-01` Canonical repository layout | LogRef: 2026-05-21 / CA-2211-01
- [x] `CA-2211-02` Agent and English-only policy | LogRef: 2026-05-21 / CA-2211-02
- [x] `CA-2211-03` One-shot installation path | LogRef: 2026-05-21 / CA-2211-03
- [x] `CA-2211-04` Native `.so` git hygiene | LogRef: 2026-05-21 / CA-2211-04
- [ ] `CA-2211-05` Confirm public GitHub remote and README clone URL
- [ ] `CA-2211-06` `cropability.yaml` plus environment variable validation
- [x] `CA-2211-07` Rust I/O plus PyO3 native extension | LogRef: baseline rollup
- [x] `CA-2211-08` Pytest layout and markers | LogRef: baseline rollup

#### 2.1.2 GPU

- [ ] `CA-2212-00` GPU
- [x] `CA-2212-01` DeviceManager | LogRef: baseline rollup
- [x] `CA-2212-02` Triton/PyTorch kernels | LogRef: baseline rollup
- [ ] `CA-2212-03` Multi-GPU DDP documentation and smoke test
- [ ] `CA-2212-04` Kernel versus CPU benchmark

#### 2.1.3 Delivery

- [ ] `CA-2213-00` Delivery
- [x] `CA-2213-01` Development documentation suite | LogRef: baseline rollup
- [x] `CA-2213-02` Progress tracking documentation | LogRef: 2026-05-21 / CA-2213-02
- [ ] `CA-2213-03` PyPI / wheel release
- [ ] `CA-2213-04` `0.2.0` release gate

### 2.2 FastCall GPU heterogeneity

- [ ] `CA-222-00` FastCall GPU heterogeneity

#### 2.2.1 JNI method

- [x] `CA-2221-01` JNI method | LogRef: vault

#### 2.2.2 TorchScript method

- [x] `CA-2222-00` TorchScript method | LogRef: vault
- [x] `CA-2222-01` Test how sample size affects GPU speedup | LogRef: vault
- [x] `CA-2222-02` Speed up likelihood calculation | LogRef: vault
- [x] `CA-2222-03` Improve whole-program runtime | LogRef: vault
- [x] `CA-2222-04` Resolve Python object cleanup in GPU memory | LogRef: vault
- [x] `CA-2222-05` Design the new TorchScript approach | LogRef: vault
- [x] `CA-2222-06` Import Java test path | LogRef: vault
- [x] `CA-2222-07` Decide to drop Java and rewrite FastCall3 in Rust | LogRef: vault

#### 2.2.3 Software refactor: rewrite FastCall3 in Rust

- [ ] `CA-2223-00` Software refactor using Rust for FastCall3

##### 2.2.3.1 Foundation rewrite

- [ ] `CA-2223-10` Foundation rewrite
- [ ] `CA-2223-11` Variation-library integration
- [ ] `CA-2223-12` Coupling with variation-library

##### 2.2.3.2 Import samtools mpileup

- [ ] `CA-2223-20` Import samtools mpileup
- [x] `CA-2223-21` CLI pileup / call-variants | LogRef: 2026-05-21 / CA-2223-21
- [x] `CA-2223-22` In-process mpileup plus FastCall3-style calling | LogRef: 2026-05-21 / CA-2223-22

### 2.3 Personal genetics and breeding algorithms

- [ ] `CA-223-00` Personal genetics and breeding algorithm implementation
- [x] `CA-223-01` Variant/LD/GWAS engine | LogRef: baseline rollup
- [x] `CA-223-02` SmithWatermanGPU | LogRef: baseline rollup
- [ ] `CA-223-03` CRAM smoke test

### 2.4 Visualization

- [ ] `CA-224-00` Visualization
- [ ] `CA-224-01` Visualization entry point
