# Application resources

Runtime configuration and private assets for CropAbility.

| Path | Purpose |
|------|---------|
| `simplelogger.properties` | Logging defaults |
| `triton.properties` | Triton-related settings |
| `private/` | Local-only files (gitignored) |

PyTorch models and checkpoints are **not** stored here; keep them outside the repo or in your own data paths. GPU inference uses Python directly (no TorchScript export for Rust).
