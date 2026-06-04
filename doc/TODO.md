# CropAbility - project TODO

Master execution checklist for the **CropAbility** publishable genomics toolkit
(Python host + Rust `_core`).

**Authority:** Repository execution checklist for CropAbility development.

**Traceability:**

- Mark `[x]` when the work is complete.
- Add or update an entry in [`doc/TODO_PROGRESS_LOG.md`](TODO_PROGRESS_LOG.md) when you
  close a checkbox (or when an existing rollup entry already covers the work).

---

## 1. FlagOS operator competition

- [x] FlagOS operator development competition

---

## 2. CropAbility GPU-enabled bioinformatics toolkit

- [ ] Integrate GPU computing into the CropAbility bioinformatics toolkit

### 2.1 Project architecture

- [ ] Project architecture

#### 2.1.1 Foundation

- [ ] Foundation
- [x] Canonical repository layout
- [x] Agent and English-only policy
- [x] One-shot installation path
- [x] Native `.so` git hygiene
- [ ] Confirm public GitHub remote and README clone URL
- [ ] `cropability.yaml` plus environment variable validation
- [x] Rust I/O plus PyO3 native extension
- [x] Pytest layout and markers

#### 2.1.2 GPU

- [ ] GPU
- [x] DeviceManager
- [x] Triton/PyTorch kernels
- [ ] Multi-GPU DDP documentation and smoke test
- [ ] Kernel versus CPU benchmark

#### 2.1.3 Delivery

- [ ] Delivery
- [x] Development documentation suite
- [x] Progress tracking documentation
- [ ] PyPI / wheel release
- [ ] `0.2.0` release gate

### 2.2 FastCall GPU heterogeneity

- [ ] FastCall GPU heterogeneity

#### 2.2.1 JNI method

- [x] JNI method

#### 2.2.2 TorchScript method

- [x] TorchScript method
- [x] Test how sample size affects GPU speedup
- [x] Speed up likelihood calculation
- [x] Improve whole-program runtime
- [x] Resolve Python object cleanup in GPU memory
- [x] Design the new TorchScript approach
- [x] Import Java test path
- [x] Decide to drop Java and rewrite FastCall3 in Rust

#### 2.2.3 Software refactor: rewrite FastCall3 in Rust

- [ ] Software refactor using Rust for FastCall3

##### 2.2.3.1 Foundation rewrite

- [ ] Foundation rewrite
- [ ] Variation-library integration
- [ ] Coupling with variation-library

##### 2.2.3.2 Import samtools mpileup

- [ ] Import samtools mpileup
- [x] CLI pileup / call-variants
- [x] In-process mpileup plus FastCall3-style calling

### 2.3 Personal genetics and breeding algorithms

- [ ] Personal genetics and breeding algorithm implementation
- [x] Variant/LD/GWAS engine
- [x] SmithWatermanGPU
- [ ] CRAM smoke test

### 2.4 Visualization

- [ ] Visualization
- [ ] Visualization entry point
