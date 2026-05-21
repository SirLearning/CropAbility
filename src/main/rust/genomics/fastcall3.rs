//! Native FastCall3-style variant caller (CPU).

use anyhow::Result;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use crate::genomics::pileup::{NativePileupEngine, PileupRecord};
use crate::io::vcf::{VcfRecord, VcfWriter};

#[derive(Clone, Debug)]
pub struct FastCall3RunResult {
    pub command: Vec<String>,
    pub returncode: i32,
    pub stdout: String,
    pub stderr: String,
    pub output_vcf: PathBuf,
    pub backend: String,
    pub n_records: u32,
    pub elapsed_seconds: f64,
}

pub struct FastCall3Runner {
    pub engine: NativePileupEngine,
}

impl FastCall3Runner {
    pub fn new() -> Self {
        Self {
            engine: NativePileupEngine::new(),
        }
    }

    pub fn run(
        &self,
        reference: &Path,
        bam_files: &[PathBuf],
        output_vcf: &Path,
        regions: Option<&str>,
        min_base_quality: u8,
        min_mapping_quality: u8,
        min_depth: u32,
        min_alt_freq: f64,
        dry_run: bool,
    ) -> Result<FastCall3RunResult> {
        let command = build_command(reference, bam_files, output_vcf, regions, min_base_quality, min_mapping_quality, min_depth, min_alt_freq);
        if dry_run {
            return Ok(FastCall3RunResult {
                command,
                returncode: 0,
                stdout: "dry-run".into(),
                stderr: String::new(),
                output_vcf: output_vcf.to_path_buf(),
                backend: "dry-run".into(),
                n_records: 0,
                elapsed_seconds: 0.0,
            });
        }

        let start = Instant::now();
        let records = self.engine.generate_records(
            reference,
            bam_files,
            regions,
            min_base_quality,
            min_mapping_quality,
        )?;
        let sample_names: Vec<String> = bam_files
            .iter()
            .map(|p| {
                p.file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("sample")
                    .to_string()
            })
            .collect();

        let meta = [
            "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"Total Depth\">",
            "##INFO=<ID=AF,Number=1,Type=Float,Description=\"Alt Allele Frequency\">",
            "##INFO=<ID=AC,Number=1,Type=Integer,Description=\"Alt Allele Count\">",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
            "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"Read Depth\">",
            "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"Allele depths\">",
        ];
        let mut writer = VcfWriter::create(output_vcf, &sample_names)?;
        writer.write_header("CropAbility.NativeFastCall3", &meta.to_vec())?;
        let mut n_records = 0u32;
        for rec in records {
            if let Some(vcf) = call_record(&rec, &sample_names, min_depth, min_alt_freq) {
                writer.write_record(&vcf)?;
                n_records += 1;
            }
        }
        let elapsed = start.elapsed().as_secs_f64();
        Ok(FastCall3RunResult {
            command,
            returncode: 0,
            stdout: format!("native fastcall3 completed with {n_records} records"),
            stderr: String::new(),
            output_vcf: output_vcf.to_path_buf(),
            backend: "rust".into(),
            n_records,
            elapsed_seconds: elapsed,
        })
    }
}

fn build_command(
    reference: &Path,
    bam_files: &[PathBuf],
    output_vcf: &Path,
    regions: Option<&str>,
    mbq: u8,
    mmq: u8,
    mdp: u32,
    maf: f64,
) -> Vec<String> {
    let mut cmd = vec![
        "cropability-fastcall3-native".into(),
        "--reference".into(),
        reference.display().to_string(),
        "--output".into(),
        output_vcf.display().to_string(),
        "--min-base-quality".into(),
        mbq.to_string(),
        "--min-mapping-quality".into(),
        mmq.to_string(),
        "--min-depth".into(),
        mdp.to_string(),
        "--min-alt-freq".into(),
        maf.to_string(),
    ];
    for b in bam_files {
        cmd.push("--bam".into());
        cmd.push(b.display().to_string());
    }
    if let Some(r) = regions {
        cmd.push("--regions".into());
        cmd.push(r.to_string());
    }
    cmd
}

fn choose_genotype(ref_count: u32, alt_count: u32, depth: u32, min_depth: u32) -> String {
    if depth < min_depth {
        return "./.".into();
    }
    if depth == 0 {
        return "./.".into();
    }
    let af = alt_count as f64 / depth as f64;
    if af >= 0.8 {
        "1/1".into()
    } else if af >= 0.2 {
        "0/1".into()
    } else {
        "0/0".into()
    }
}

fn call_record(
    rec: &PileupRecord,
    sample_names: &[String],
    min_depth: u32,
    min_alt_freq: f64,
) -> Option<VcfRecord> {
    let mut total_depth = 0u32;
    let mut merged: BTreeMap<String, u32> = ["A", "C", "G", "T", "N"]
        .iter()
        .map(|b| (b.to_string(), 0))
        .collect();
    for sample in rec.samples.values() {
        total_depth += sample.depth;
        for (base, count) in &sample.base_counts {
            *merged.entry(base.clone()).or_insert(0) += count;
        }
    }
    if total_depth < min_depth {
        return None;
    }
    let ref_b = rec.ref_base.to_uppercase();
    let candidates: Vec<_> = merged
        .iter()
        .filter(|(b, c)| matches!(b.as_str(), "A" | "C" | "G" | "T") && b.as_str() != ref_b && **c > 0)
        .collect();
    if candidates.is_empty() {
        return None;
    }
    let (alt, alt_count) = candidates
        .into_iter()
        .max_by_key(|(_, c)| *c)
        .map(|(b, c)| (b.clone(), *c))?;
    let alt_freq = alt_count as f64 / total_depth.max(1) as f64;
    if alt_freq < min_alt_freq {
        return None;
    }
    let mut sample_values = Vec::new();
    for name in sample_names {
        let s = rec.samples.get(name)?;
        let depth = s.depth;
        let ref_count = *s.base_counts.get(&ref_b).unwrap_or(&0);
        let sample_alt = *s.base_counts.get(&alt).unwrap_or(&0);
        let gt = choose_genotype(ref_count, sample_alt, depth, min_depth);
        sample_values.push(format!("{gt}:{depth}:{ref_count},{sample_alt}"));
    }
    Some(VcfRecord {
        chrom: rec.chrom.clone(),
        pos: rec.pos,
        ref_allele: ref_b,
        alt: vec![alt],
        info: vec![
            ("DP".into(), total_depth.to_string()),
            ("AF".into(), format!("{alt_freq:.6}")),
            ("AC".into(), alt_count.to_string()),
        ],
        format_keys: vec!["GT".into(), "DP".into(), "AD".into()],
        sample_values,
    })
}
