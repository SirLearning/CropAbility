//! CropAbility native layer — CPU computation, I/O, and NGS pipeline.
//! Python (`src/main/python/cropability`) owns PyTorch GPU work.

pub mod io;
pub mod genomics;

#[cfg(feature = "python")]
mod python;

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Python module: `from cropability.native import _core`
#[cfg(feature = "python")]
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    python::register(m)
}
