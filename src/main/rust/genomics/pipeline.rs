//! NGS variant pipeline orchestration (CPU).

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::genomics::fastcall3::FastCall3Runner;
use crate::genomics::pileup::{write_pileup_summary_tsv, NativePileupEngine};
use crate::io::bam::AlignmentInputManager;

#[derive(Clone, Debug)]
pub struct QCThresholds {
    pub min_depth: u32,
    pub min_base_quality: u8,
    pub min_mapping_quality: u8,
    pub min_alt_freq: f64,
}

impl Default for QCThresholds {
    fn default() -> Self {
        Self {
            min_depth: 10,
            min_base_quality: 20,
            min_mapping_quality: 20,
            min_alt_freq: 0.05,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct PipelineConfig;

pub struct VariantPipeline {
    pub pileup_engine: NativePileupEngine,
    pub fastcall3: FastCall3Runner,
}

impl VariantPipeline {
    pub fn new() -> Self {
        Self {
            pileup_engine: NativePileupEngine::new(),
            fastcall3: FastCall3Runner::new(),
        }
    }

    pub fn run(
        &self,
        mode: &str,
        reference: &Path,
        bam_files: &[PathBuf],
        output: &Path,
        qc: &QCThresholds,
        regions: Option<&str>,
        mpileup_output: Option<&Path>,
        dry_run: bool,
    ) -> Result<PipelineReport> {
        let mode = mode.to_lowercase();
        if !matches!(mode.as_str(), "mpileup" | "fastcall3" | "hybrid") {
            bail!("mode must be mpileup, fastcall3, or hybrid");
        }
        let manager = AlignmentInputManager::new(bam_files.to_vec())?;
        let aligned = manager.collect(!dry_run)?;
        let bam_paths: Vec<PathBuf> = aligned.iter().map(|a| a.path.clone()).collect();

        let start = Instant::now();
        let mut report = PipelineReport {
            mode: mode.clone(),
            reference: reference.display().to_string(),
            engine: "rust".into(),
            mpileup: None,
            fastcall3: None,
            elapsed_seconds: 0.0,
        };

        match mode.as_str() {
            "mpileup" => {
                report.mpileup = Some(self.run_mpileup(
                    reference,
                    &bam_paths,
                    output,
                    qc,
                    regions,
                    dry_run,
                )?);
            }
            "fastcall3" => {
                report.fastcall3 = Some(self.run_fastcall3(
                    reference,
                    &bam_paths,
                    output,
                    qc,
                    regions,
                    dry_run,
                )?);
            }
            _ => {
                let mp_out = mpileup_output
                    .map(|p| p.to_path_buf())
                    .unwrap_or_else(|| PathBuf::from(format!("{}.pileup.summary.tsv", output.display())));
                report.mpileup = Some(self.run_mpileup(
                    reference,
                    &bam_paths,
                    &mp_out,
                    qc,
                    regions,
                    dry_run,
                )?);
                report.fastcall3 = Some(self.run_fastcall3(
                    reference,
                    &bam_paths,
                    output,
                    qc,
                    regions,
                    dry_run,
                )?);
            }
        }
        report.elapsed_seconds = start.elapsed().as_secs_f64();
        Ok(report)
    }

    pub fn run_mpileup(
        &self,
        reference: &Path,
        bam_files: &[PathBuf],
        output: &Path,
        qc: &QCThresholds,
        regions: Option<&str>,
        dry_run: bool,
    ) -> Result<MpileupReport> {
        if dry_run {
            return Ok(MpileupReport {
                engine: "rust".into(),
                command: vec![
                    "cropability-native-pileup".into(),
                    "--reference".into(),
                    reference.display().to_string(),
                    "--output".into(),
                    output.display().to_string(),
                ],
                output: output.display().to_string(),
                returncode: 0,
                n_sites: 0,
            });
        }
        let records = self.pileup_engine.generate_records(
            reference,
            bam_files,
            regions,
            qc.min_base_quality,
            qc.min_mapping_quality,
        )?;
        let n_sites = write_pileup_summary_tsv(output, records, qc.min_depth, qc.min_alt_freq)?;
        Ok(MpileupReport {
            engine: "rust".into(),
            command: vec![],
            output: output.display().to_string(),
            returncode: 0,
            n_sites,
        })
    }

    pub fn run_fastcall3(
        &self,
        reference: &Path,
        bam_files: &[PathBuf],
        output_vcf: &Path,
        qc: &QCThresholds,
        regions: Option<&str>,
        dry_run: bool,
    ) -> Result<crate::genomics::fastcall3::FastCall3RunResult> {
        self.fastcall3.run(
            reference,
            bam_files,
            output_vcf,
            regions,
            qc.min_base_quality,
            qc.min_mapping_quality,
            qc.min_depth,
            qc.min_alt_freq,
            dry_run,
        )
    }
}

#[derive(Clone, Debug)]
pub struct MpileupReport {
    pub engine: String,
    pub command: Vec<String>,
    pub output: String,
    pub returncode: i32,
    pub n_sites: u32,
}

#[derive(Clone, Debug)]
pub struct PipelineReport {
    pub mode: String,
    pub reference: String,
    pub engine: String,
    pub mpileup: Option<MpileupReport>,
    pub fastcall3: Option<crate::genomics::fastcall3::FastCall3RunResult>,
    pub elapsed_seconds: f64,
}
