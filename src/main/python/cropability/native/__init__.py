"""Rust native extension (`cropability-native` crate). Built with maturin."""

try:
    from cropability.native import _core
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "cropability.native._core not found. Build with:\n"
        "  maturin develop --release --features python,htslib"
    ) from e

__all__ = ["_core"]
