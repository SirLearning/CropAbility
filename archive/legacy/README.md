# Legacy code (`archive/legacy/pgl/`)

## What this is

**PGL (Performance GPU Library)** — the pre-CropAbility prototype:

| Area | Contents |
|------|----------|
| `pgl/ops/` | Triton/PyTorch `add`, `gtp_gpu`, TorchScript export, benchmarks |
| `pgl/ca/gpu/` | Early GPU kernel experiments (e.g. `birth.py` template) |
| Tests | `test.py`, `test_gtp_gpu.py`, etc. |

It targeted **Java + TorchScript + vendored libtorch** under repo `lib/`. That stack was removed in favor of:

- **Product**: `cropability` (`src/main/python/cropability/`)
- **CPU/NGS**: Rust PyO3 extension (`src/main/rust/`)
- **GPU**: Python PyTorch/Triton via Conda/pip (no vendored `lib/`)

## Should this move into `src/`?

**No.** Keep it here (or delete the whole `archive/` tree if you no longer need reference).

| Reason | Detail |
|--------|--------|
| Different product | Package name `pgl` ≠ `cropability`; not in `pyproject.toml` |
| Wrong architecture | TorchScript-for-Java, duplicated ops already in `cropability/kernels` |
| Publishable wheel | Only `cropability*` is installed; `archive/` is intentionally excluded |
| Maintenance | Merging would confuse agents and users about the supported API |

To reuse an idea (e.g. a Triton kernel), **port** the logic into `cropability/kernels/` or `cropability/genomics/`, do not move `pgl/` into `src/main/python/`.

## Safe to delete?

Yes, if you do not need historical reference. Deleting `archive/` does not affect `pip install` or `maturin develop`.
