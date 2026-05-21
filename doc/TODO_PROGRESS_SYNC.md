# CropAbility progress sync (vault and GitHub)

This document defines how **`doc/TODO.md`** and **`doc/TODO_PROGRESS_LOG.md`** connect to Obsidian work notes and to the GitHub-based sync workflow used for SirLearning/script (see skill `genomics-script-ssh-todo-sync`).

**Direction:** CropAbility repository → vault notes (read-only from the repo side during sync). Do not edit the GitHub repo from vault sync agents.

---

## ID scheme (2026-05-21)

| Scheme | Status | Where used |
|--------|--------|------------|
| **`CA-*`** | **Current** — checkbox truth in `doc/TODO.md` | Vault daily note, `LogRef` slugs, new log entries |
| **`M1-*`** | **Historical** — retired for new checkboxes | Old log slugs, migration table in `TODO_PROGRESS_LOG.md`, vault `CropAbility.md` §1.1.1 |

**Format:** `CA-{section}-{item}`. Parent items end in `-00`; leaf items end in `-01`, `-02`, …

**LogRef:** Point to the primary **`CA-*`** heading slug in `doc/TODO_PROGRESS_LOG.md`
(e.g. `LogRef: 2026-05-21 / CA-2211-01`). Use `LogRef: vault` for items complete only in the vault.

Full `M1-*` → `CA-*` mapping: [`TODO_PROGRESS_LOG.md` — ID scheme and historical migration](TODO_PROGRESS_LOG.md#id-scheme-and-historical-migration).

---

## Authoritative sources (read-only)

**Repository root:** `D:/Zheng/Documents/2_NBS/Python/CropAbility/doc/`

| File | Role |
|------|------|
| [`TODO.md`](TODO.md) | Master `[ ]` / `[x]` checklist; stable **`CA-*`** IDs (checkbox truth source) |
| [`TODO_PROGRESS_LOG.md`](TODO_PROGRESS_LOG.md) | Dated engineering reports; read newest entries first |
| [`TESTING.md`](TESTING.md) | Pytest layout, markers, CI-style commands |
| [`RUST_DEVELOPMENT.md`](RUST_DEVELOPMENT.md) | PyO3, maturin, NGS Rust layout |
| [`PYTHON_DEVELOPMENT.md`](PYTHON_DEVELOPMENT.md) | Python API; legacy PGL blocks are archive reference only |
| [`DEPENDENCIES.md`](DEPENDENCIES.md) | Install and Conda/pip matrix |

### GitHub raw (default for remote sync)

| Purpose | Browse | Raw |
|---------|--------|-----|
| Task checklist | [TODO.md](https://github.com/SirLearning/CropAbility/blob/main/doc/TODO.md) | `https://raw.githubusercontent.com/SirLearning/CropAbility/main/doc/TODO.md` |
| Progress log | [TODO_PROGRESS_LOG.md](https://github.com/SirLearning/CropAbility/blob/main/doc/TODO_PROGRESS_LOG.md) | `https://raw.githubusercontent.com/SirLearning/CropAbility/main/doc/TODO_PROGRESS_LOG.md` |

**Branch:** `main` / **`doc/`** is canonical (same pattern as `SirLearning/script`).

**Local path (when working in this clone):** prefer reading `doc/*.md` directly instead of raw URLs.

---

## Vault layout (suggested)

Mirror the DBone (`2.DBone/`) and Genomics (`1.Genomics/`) pattern:

```
AEWT/AWGP_Vmap4/3.CropAbility/
├── 1.index.md              # §1.4 progress snapshot; links to daily note checklist
├── 2.architecture.md       # Python host → Rust _core; §2.1.1 Foundation (CA-2211)
├── 3.gpu-kernels.md        # §2.1.2 GPU (CA-2212)
├── 4.genomics-gpu.md       # §2.3 Personal genetics (CA-223)
├── 5.ngs-native.md         # §2.2 FastCall / Rust rewrite (CA-2223)
├── 6.cli-testing.md        # CLI, pytest, CI (CA-2211-08, CA-2223-21)
├── 7.roadmap.md            # §1.4 open items, §2.1.3 Delivery, §2.4 Viz, risks
└── logs/
    ├── testing_runbook.md
    └── rust_build_runbook.md
```

Adjust folder name if your vault already uses a different number; keep **`3.CropAbility`** consistent with sibling projects.

---

## Repository `doc/` → vault mapping

Aligned with [`doc/TODO.md`](TODO.md) §2 hierarchy (post–ID migration):

| `doc/TODO.md` section | `CA-*` prefix | Vault target | What to write |
|-----------------------|---------------|--------------|---------------|
| §1 FlagOS | `CA-21-*` | `7.roadmap.md` (historical) | Vault-only competition note; `LogRef: vault` |
| §2.1.1 Foundation | `CA-2211-*` | `2.architecture.md`, `7.roadmap.md` | Layout, install, config, PyO3, pytest |
| §2.1.2 GPU | `CA-2212-*` | `3.gpu-kernels.md` | DeviceManager, Triton kernels, DDP, benchmarks |
| §2.1.3 Delivery | `CA-2213-*` | `7.roadmap.md` | Docs, progress tracking, PyPI, release gate |
| §2.2.1–2.2.2 JNI / TorchScript | `CA-2221-*`, `CA-2222-*` | `5.ngs-native.md` (archive subsection) | Vault-only exploration; do not resurrect Java/TorchScript in product |
| §2.2.3 Rust FastCall rewrite | `CA-2223-*` | `5.ngs-native.md` | pileup, FastCall3, pipeline, maturin |
| §2.3 Personal genetics | `CA-223-*` | `4.genomics-gpu.md` | Variant, LD, GWAS, alignment, CRAM |
| §2.4 Visualization | `CA-224-*` | `7.roadmap.md` | Viz entry points |
| `TODO.md` (all) | all `CA-*` | Daily note **`## CropAbility`** | Update `- [ ]` / `- [x]` and short `LogRef` only |
| `TODO_PROGRESS_LOG.md` | log slugs | `1.index.md` **§1.4** + themed files | Chinese progress reports; 1–5 latest log bullets in §1.4 |
| `RUST_DEVELOPMENT.md` | — | `5.ngs-native.md`, `logs/rust_build_runbook.md` | Build/feature matrix summary (not full copy) |
| `TESTING.md` | — | `6.cli-testing.md`, `logs/testing_runbook.md` | Marker matrix, commands |
| Open items / Follow-ups | open `CA-*` | `7.roadmap.md` **§1.4** | Consolidated risks and next steps |

### Historical section map (pre–2026-05-21 flat checklist)

| Retired bucket | Maps to current `CA-*` home |
|----------------|----------------------------|
| M1-FOUND | `CA-2211-*` |
| M1-GPU | `CA-2212-*` |
| M1-DELIV | `CA-2213-*` |
| M1-NGS, M1-CLI | `CA-2223-*` |
| M1-GEN | `CA-223-*` |
| M1-TEST | `CA-2211-08` |

---

## Boundaries (aligned with DBone and Genomics skills)

| Location | Allowed | Forbidden |
|----------|---------|-----------|
| Daily note → CropAbility section | Checkbox tree; inline short `LogRef`; wikilinks | Long sync summaries, full doc path lists, design essays |
| `3.CropAbility/1.index.md` §1.4 | Sync date, scanned `doc/` files, open `CA-*` summary, latest log bullets, diary wikilink | Replacing the diary checkbox tree |
| `3.CropAbility/` other files | Themed Chinese technical reports, dated `### YYYY-MM-DD …`, tables | Dumping entire `doc/*.md` verbatim |
| Repository `doc/` | English checklists and logs | Vault-only notes |

**Do not** create vault-only “sync summary” index pages that duplicate §1.4.

---

## Agent execution steps (CropAbility)

1. **Fetch or read** all progress-related files under `doc/` (at minimum `TODO.md`, `TODO_PROGRESS_LOG.md`; add `TESTING.md` / `RUST_DEVELOPMENT.md` when log entries reference them).
2. **Build a mapping table:** each log entry / open **`CA-*`** ID → vault path + target heading. Resolve historical **`M1-*`** via [`TODO_PROGRESS_LOG.md`](TODO_PROGRESS_LOG.md#m1--ca-mapping-repository-completions).
3. **Read existing vault sections**; merge new facts as coherent Chinese prose (tables and timelines OK); mark conflicts `待确认` with **`CA-*`** TODO ID.
4. **Update the daily note** CropAbility section: checkboxes and `LogRef` only, using `doc/TODO.md` as truth (**`CA-*`** IDs, not `M1-*`).
5. **Replace** `3.CropAbility/1.index.md` **§1.4** with the snapshot template below.
6. **Update themed vault files** per mapping table.
7. **Report:** scanned `doc/` files, changed vault paths, checkbox deltas, remaining open **`CA-*`** IDs.

### Optional GitHub pull

When the local clone is behind or unpushed:

```bash
git -C "D:/Zheng/Documents/2_NBS/Python/CropAbility" fetch origin main
git -C "D:/Zheng/Documents/2_NBS/Python/CropAbility" log origin/main..HEAD --oneline
```

Default sync authority: **GitHub raw** on `main`. Note in vault body if SSH/local clone was used instead.

---

## `1.index.md` §1.4 template (Chinese)

Use in vault `3.CropAbility/1.index.md`:

```markdown
## 1.4 代码库进度快照（CropAbility）

- **同步日期**：YYYY-MM-DD
- **已扫描 doc/**：`TODO.md`、`TODO_PROGRESS_LOG.md`、…（列出本次实际读取的文件名）
- **CA 概况**：未完成 `CA-*`：…（数量 + ID 列表或概括）
- **最近完成（TODO_PROGRESS_LOG）**：…（1～5 条中文要点，含验证 pass/fail；引用 `CA-*` LogRef）
- **分主题落点**：Foundation → [[2.architecture]]；GPU → [[3.gpu-kernels]]；GPU genomics → [[4.genomics-gpu]]；NGS/Rust → [[5.ngs-native]]；CLI/测试 → [[6.cli-testing]]；风险/发布 → [[7.roadmap#1.4 待确认、待推进与已知问题（中文备忘）]]
- **勾选清单位置**：仅 `[[YYYY-MM-DD#CropAbility]]`；本节不替代日记勾选。
- **ID 说明**：清单以 `CA-*` 为准；历史 `M1-*` 见仓库 `TODO_PROGRESS_LOG.md` 迁移表。
```

Replace `YYYY-MM-DD` with the diary file used for that sync.

---

## Daily note checklist template (English IDs, vault-facing)

Copy structure from [`TODO.md`](TODO.md); use **`CA-*`** IDs only:

```markdown
## CropAbility

- [x] CA-2211-01 Canonical repository layout | LogRef: 2026-05-21 / CA-2211-01
- [ ] CA-2211-05 Confirm public GitHub remote and README clone URL | LogRef: pending
- [x] CA-2223-22 In-process mpileup plus FastCall3-style calling | LogRef: 2026-05-21 / CA-2223-22
```

Do not invent IDs not present in the repo file. For vault-only historical items, keep `LogRef: vault`.

---

## Cursor skill hook (optional)

To automate vault sync, add a skill (for example `cropability-vault-progress-sync`) that:

1. Points at this file for mapping rules.
2. Reads `D:/Zheng/Documents/2_NBS/Python/CropAbility/doc/` (or GitHub raw URLs above).
3. Writes into `AEWT/AWGP_Vmap4/3.CropAbility/` using the same merge discipline as `dbone-vault-progress-sync` and `genomics-script-ssh-todo-sync`.

Invoke via `@cropability-vault-progress-sync` once the skill file exists under `~/.cursor/skills/`.

---

## Writing constraints

- Vault narrative: **Chinese**; keep **`CA-*`** IDs, historical **`M1-*`** (when citing old logs), Rust/Python module paths, and CLI flags in **original casing**.
- Do not mark items complete in the vault unless `doc/TODO.md` shows `[x]`.
- Every completed repo item must have a matching `TODO_PROGRESS_LOG.md` entry (or baseline rollup row) before checking the box in `TODO.md`, except vault-only items (`LogRef: vault`).
