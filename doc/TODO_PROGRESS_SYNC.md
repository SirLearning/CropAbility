# CropAbility progress sync (vault and GitHub)

This document defines how **`doc/TODO.md`** and **`doc/TODO_PROGRESS_LOG.md`** connect to Obsidian work notes and to the GitHub-based sync workflow used for SirLearning/script (see skill `genomics-script-ssh-todo-sync`).

**Direction:** CropAbility repository → vault notes (read-only from the repo side during sync). Do not edit the GitHub repo from vault sync agents.

---

## Authoritative sources (read-only)

**Repository root:** `D:/Zheng/Documents/2_NBS/Python/CropAbility/doc/`

| File | Role |
|------|------|
| [`TODO.md`](TODO.md) | Master `[ ]` / `[x]` checklist; stable `M1-*` IDs (checkbox truth source) |
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
├── 2.architecture.md       # Python host → Rust _core; forbidden patterns
├── 3.gpu-kernels.md        # gpu/, kernels/
├── 4.genomics-gpu.md       # genomics/ (GPU)
├── 5.ngs-native.md         # ngs/, Rust io/genomics
├── 6.cli-testing.md        # CLI, pytest, CI
├── 7.roadmap.md            # §1.4 open items, risks, follow-ups
└── logs/
    ├── testing_runbook.md
    └── rust_build_runbook.md
```

Adjust folder name if your vault already uses a different number; keep **`3.CropAbility`** consistent with sibling projects.

---

## Repository `doc/` → vault mapping

| Repository document | Vault target | What to write |
|---------------------|--------------|---------------|
| `TODO.md` | Daily note section **`## CropAbility (GPU toolkit)`** (or user-specified heading) | Update `- [ ]` / `- [x]` and short `LogRef` only |
| `TODO_PROGRESS_LOG.md` | `1.index.md` **§1.4** + themed files below | Chinese progress reports; 1–5 latest log bullets in §1.4 |
| `TODO.md` §1 FOUND | `2.architecture.md`, `7.roadmap.md` | Packaging, install, config |
| `TODO.md` §2 GPU | `3.gpu-kernels.md` | DeviceManager, Triton kernels, DDP |
| `TODO.md` §3 GEN | `4.genomics-gpu.md` | Variant, LD, GWAS, alignment |
| `TODO.md` §4 NGS | `5.ngs-native.md` | pileup, FastCall3, pipeline, maturin |
| `TODO.md` §5–§6 CLI/TEST | `6.cli-testing.md`, `logs/testing_runbook.md` | CLI coverage, pytest, CI |
| `RUST_DEVELOPMENT.md` | `5.ngs-native.md`, `logs/rust_build_runbook.md` | Build/feature matrix summary (not full copy) |
| `TESTING.md` | `6.cli-testing.md`, `logs/testing_runbook.md` | Marker matrix, commands |
| Open items / Follow-ups | `7.roadmap.md` **§1.4** | Consolidated risks and next steps |

---

## Boundaries (aligned with DBone and Genomics skills)

| Location | Allowed | Forbidden |
|----------|---------|-----------|
| Daily note → CropAbility section | Checkbox tree; inline short `LogRef`; wikilinks | Long sync summaries, full doc path lists, design essays |
| `3.CropAbility/1.index.md` §1.4 | Sync date, scanned `doc/` files, M1 summary, latest log bullets, diary wikilink | Replacing the diary checkbox tree |
| `3.CropAbility/` other files | Themed Chinese technical reports, dated `### YYYY-MM-DD …`, tables | Dumping entire `doc/*.md` verbatim |
| Repository `doc/` | English checklists and logs | Vault-only notes |

**Do not** create vault-only “sync summary” index pages that duplicate §1.4.

---

## Agent execution steps (CropAbility)

1. **Fetch or read** all progress-related files under `doc/` (at minimum `TODO.md`, `TODO_PROGRESS_LOG.md`; add `TESTING.md` / `RUST_DEVELOPMENT.md` when log entries reference them).
2. **Build a mapping table:** each log entry / open `M1-*` ID → vault path + target heading.
3. **Read existing vault sections**; merge new facts as coherent Chinese prose (tables and timelines OK); mark conflicts `待确认` with TODO ID.
4. **Update the daily note** CropAbility section: checkboxes and `LogRef` only, using `doc/TODO.md` as truth.
5. **Replace** `3.CropAbility/1.index.md` **§1.4** with the snapshot template below.
6. **Update themed vault files** per mapping table.
7. **Report:** scanned `doc/` files, changed vault paths, checkbox deltas, remaining open `M1-*` IDs.

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
- **M1 概况**：未完成 `M1-*`：…（数量 + ID 列表或概括）
- **最近完成（TODO_PROGRESS_LOG）**：…（1～5 条中文要点，含验证 pass/fail）
- **分主题落点**：GPU → [[3.gpu-kernels]]；GPU genomics → [[4.genomics-gpu]]；NGS/Rust → [[5.ngs-native]]；CLI/测试 → [[6.cli-testing]]；风险 → [[7.roadmap#1.4 待确认、待推进与已知问题（中文备忘）]]
- **勾选清单位置**：仅 `[[YYYY-MM-DD#CropAbility (GPU toolkit)]]`；本节不替代日记勾选。
```

Replace `YYYY-MM-DD` with the diary file used for that sync.

---

## Daily note checklist template (English IDs, vault-facing)

```markdown
## CropAbility (GPU toolkit)

- [x] M1-FOUND-001 … | LogRef: 2026-05-21 / M1-FOUND-001
- [ ] M1-NGS-005 … | LogRef: pending
```

Copy structure from [`TODO.md`](TODO.md); do not invent IDs not present in the repo file.

---

## Cursor skill hook (optional)

To automate vault sync, add a skill (for example `cropability-vault-progress-sync`) that:

1. Points at this file for mapping rules.
2. Reads `D:/Zheng/Documents/2_NBS/Python/CropAbility/doc/` (or GitHub raw URLs above).
3. Writes into `AEWT/AWGP_Vmap4/3.CropAbility/` using the same merge discipline as `dbone-vault-progress-sync` and `genomics-script-ssh-todo-sync`.

Invoke via `@cropability-vault-progress-sync` once the skill file exists under `~/.cursor/skills/`.

---

## Writing constraints

- Vault narrative: **Chinese**; keep `M1-*` IDs, Rust/Python module paths, and CLI flags in **original casing**.
- Do not mark items complete in the vault unless `doc/TODO.md` shows `[x]`.
- Every completed repo item must have a matching `TODO_PROGRESS_LOG.md` entry before checking the box in `TODO.md`.
