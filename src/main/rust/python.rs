//! PyO3 bindings for `cropability.native._core`.

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::path::PathBuf;

use crate::genomics::pipeline::{QCThresholds, VariantPipeline};
use crate::io::bam::AlignmentInputManager;
use crate::io::fasta::FastaReader;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFastaReader>()?;
    m.add_class::<PyVariantPipeline>()?;
    m.add_class::<PyQCThresholds>()?;
    Ok(())
}

#[pyclass(name = "FastaReader")]
pub struct PyFastaReader {
    inner: FastaReader,
}

#[pymethods]
impl PyFastaReader {
    #[new]
    fn new(path: String) -> PyResult<Self> {
        Ok(Self {
            inner: FastaReader::new(path).map_err(|e| PyValueError::new_err(e.to_string()))?,
        })
    }

    fn read_all(&self, py: Python<'_>) -> PyResult<PyObject> {
        let records = self
            .inner
            .read_all()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let dict = PyDict::new_bound(py);
        for (name, seq) in records {
            dict.set_item(name, seq)?;
        }
        Ok(dict.into())
    }
}

#[pyclass(name = "QCThresholds")]
#[derive(Clone)]
pub struct PyQCThresholds {
    pub min_depth: u32,
    pub min_base_quality: u8,
    pub min_mapping_quality: u8,
    pub min_alt_freq: f64,
}

#[pymethods]
impl PyQCThresholds {
    #[new]
    #[pyo3(signature = (min_depth=10, min_base_quality=20, min_mapping_quality=20, min_alt_freq=0.05))]
    fn new(min_depth: u32, min_base_quality: u8, min_mapping_quality: u8, min_alt_freq: f64) -> Self {
        Self {
            min_depth,
            min_base_quality,
            min_mapping_quality,
            min_alt_freq,
        }
    }
}

impl From<&PyQCThresholds> for QCThresholds {
    fn from(p: &PyQCThresholds) -> Self {
        QCThresholds {
            min_depth: p.min_depth,
            min_base_quality: p.min_base_quality,
            min_mapping_quality: p.min_mapping_quality,
            min_alt_freq: p.min_alt_freq,
        }
    }
}

#[pyclass(name = "VariantPipeline")]
pub struct PyVariantPipeline {
    inner: VariantPipeline,
}

#[pymethods]
impl PyVariantPipeline {
    #[new]
    fn new() -> Self {
        Self {
            inner: VariantPipeline::new(),
        }
    }

    #[pyo3(signature = (mode, reference, bam_files, output, qc, regions=None, mpileup_output=None, dry_run=false))]
    fn run(
        &self,
        py: Python<'_>,
        mode: String,
        reference: String,
        bam_files: Vec<String>,
        output: String,
        qc: &PyQCThresholds,
        regions: Option<String>,
        mpileup_output: Option<String>,
        dry_run: bool,
    ) -> PyResult<PyObject> {
        let bam_paths: Vec<PathBuf> = bam_files.into_iter().map(PathBuf::from).collect();
        let report = self
            .inner
            .run(
                &mode,
                PathBuf::from(reference).as_path(),
                &bam_paths,
                PathBuf::from(output).as_path(),
                &qc.into(),
                regions.as_deref(),
                mpileup_output.as_deref().map(PathBuf::from).as_deref(),
                dry_run,
            )
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let d = PyDict::new_bound(py);
        d.set_item("mode", report.mode)?;
        d.set_item("reference", report.reference)?;
        d.set_item("engine", report.engine)?;
        d.set_item("elapsed_seconds", report.elapsed_seconds)?;
        if let Some(mp) = report.mpileup {
            let m = PyDict::new_bound(py);
            m.set_item("engine", mp.engine)?;
            m.set_item("output", mp.output)?;
            m.set_item("returncode", mp.returncode)?;
            m.set_item("n_sites", mp.n_sites)?;
            d.set_item("mpileup", m)?;
        }
        if let Some(fc) = report.fastcall3 {
            let f = PyDict::new_bound(py);
            let engine = fc.backend.clone();
            f.set_item("backend", engine.clone())?;
            f.set_item("engine", engine)?;
            f.set_item("output_vcf", fc.output_vcf.display().to_string())?;
            f.set_item("returncode", fc.returncode)?;
            f.set_item("n_records", fc.n_records)?;
            f.set_item("elapsed_seconds", fc.elapsed_seconds)?;
            d.set_item("fastcall3", f)?;
        }
        Ok(d.into())
    }
}

#[allow(dead_code)]
pub fn check_bam_inputs(paths: Vec<String>, require_index: bool) -> PyResult<Vec<String>> {
    let pb: Vec<PathBuf> = paths.into_iter().map(PathBuf::from).collect();
    let manager = AlignmentInputManager::new(pb).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let files = manager
        .collect(require_index)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(files.into_iter().map(|f| f.path.display().to_string()).collect())
}
