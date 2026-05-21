//! mpileup parsing and native pileup generation (CPU).

use anyhow::{bail, Result};
use std::collections::BTreeMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

const BASES: [&str; 5] = ["A", "C", "G", "T", "N"];

#[derive(Clone, Debug, Default)]
pub struct PileupSample {
    pub depth: u32,
    pub base_counts: BTreeMap<String, u32>,
    pub insertions: u32,
    pub deletions: u32,
}

#[derive(Clone, Debug)]
pub struct PileupRecord {
    pub chrom: String,
    pub pos: u32,
    pub ref_base: String,
    pub samples: BTreeMap<String, PileupSample>,
}

#[derive(Clone, Debug)]
pub struct PileupSiteSummary {
    pub chrom: String,
    pub pos: u32,
    pub ref_base: String,
    pub depth: u32,
    pub alt_base: Option<String>,
    pub alt_count: u32,
    pub alt_freq: f64,
}

pub fn parse_pileup_bases(bases: &str, ref_base: &str) -> (BTreeMap<String, u32>, u32, u32) {
    let mut counts: BTreeMap<String, u32> = BASES.iter().map(|b| (b.to_string(), 0)).collect();
    let mut insertions = 0u32;
    let mut deletions = 0u32;
    let mut i = 0usize;
    let ref_u = ref_base.to_uppercase();
    let bytes = bases.as_bytes();
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '^' {
            i += 2;
            continue;
        }
        if c == '$' {
            i += 1;
            continue;
        }
        if c == '+' || c == '-' {
            let sign = c;
            i += 1;
            let mut nbuf = String::new();
            while i < bytes.len() && bytes[i].is_ascii_digit() {
                nbuf.push(bytes[i] as char);
                i += 1;
            }
            let length: usize = nbuf.parse().unwrap_or(0);
            if sign == '+' {
                insertions += 1;
            } else {
                deletions += 1;
            }
            i += length;
            continue;
        }
        if c == '*' {
            deletions += 1;
            i += 1;
            continue;
        }
        if c == '.' || c == ',' {
            let key = if ref_u.len() == 1 { ref_u.clone() } else { "N".into() };
            *counts.entry(key).or_insert(0) += 1;
            i += 1;
            continue;
        }
        let b = c.to_ascii_uppercase().to_string();
        if BASES.contains(&b.as_str()) {
            *counts.entry(b).or_insert(0) += 1;
        } else {
            *counts.entry("N".into()).or_insert(0) += 1;
        }
        i += 1;
    }
    (counts, insertions, deletions)
}

pub struct MpileupParser {
    pub sample_names: Option<Vec<String>>,
}

impl MpileupParser {
    pub fn new(sample_names: Option<Vec<String>>) -> Self {
        Self { sample_names }
    }

    pub fn parse_line(&self, line: &str) -> Result<Option<PileupRecord>> {
        let line = line.trim_end();
        if line.is_empty() {
            return Ok(None);
        }
        let cols: Vec<&str> = line.split('\t').collect();
        if cols.len() < 6 || (cols.len() - 3) % 3 != 0 {
            return Ok(None);
        }
        let chrom = cols[0].to_string();
        let pos: u32 = cols[1].parse()?;
        let ref_base = cols[2].to_uppercase();
        let per_sample = &cols[3..];
        let n_samples = per_sample.len() / 3;
        let names: Vec<String> = if let Some(ref sn) = self.sample_names {
            if sn.len() != n_samples {
                bail!("sample_names count mismatch");
            }
            sn.clone()
        } else {
            (0..n_samples).map(|i| format!("sample{}", i + 1)).collect()
        };
        let mut samples = BTreeMap::new();
        for (i, name) in names.iter().enumerate() {
            let depth: u32 = per_sample[i * 3].parse()?;
            let bases = per_sample[i * 3 + 1];
            let (counts, ins, dels) = parse_pileup_bases(bases, &ref_base);
            samples.insert(
                name.clone(),
                PileupSample {
                    depth,
                    base_counts: counts,
                    insertions: ins,
                    deletions: dels,
                },
            );
        }
        Ok(Some(PileupRecord {
            chrom,
            pos,
            ref_base,
            samples,
        }))
    }

    pub fn summarize_sites(
        &self,
        records: impl IntoIterator<Item = PileupRecord>,
        min_depth: u32,
        min_alt_freq: f64,
    ) -> Vec<PileupSiteSummary> {
        let mut summaries = Vec::new();
        for rec in records {
            let mut merged: BTreeMap<String, u32> = BASES.iter().map(|b| (b.to_string(), 0)).collect();
            let mut depth = 0u32;
            for sample in rec.samples.values() {
                depth += sample.depth;
                for (b, c) in &sample.base_counts {
                    *merged.entry(b.clone()).or_insert(0) += c;
                }
            }
            if depth < min_depth {
                continue;
            }
            let ref_b = &rec.ref_base;
            let alt_candidates: Vec<_> = merged
                .iter()
                .filter(|(b, c)| matches!(b.as_str(), "A" | "C" | "G" | "T") && b != ref_b && **c > 0)
                .collect();
            if alt_candidates.is_empty() {
                continue;
            }
            let (alt_base, alt_count) = alt_candidates
                .into_iter()
                .max_by_key(|(_, c)| *c)
                .map(|(b, c)| (b.clone(), *c))
                .unwrap();
            let alt_freq = alt_count as f64 / depth.max(1) as f64;
            if alt_freq < min_alt_freq {
                continue;
            }
            summaries.push(PileupSiteSummary {
                chrom: rec.chrom.clone(),
                pos: rec.pos,
                ref_base: rec.ref_base.clone(),
                depth,
                alt_base: Some(alt_base),
                alt_count,
                alt_freq,
            });
        }
        summaries
    }
}

pub struct NativePileupEngine;

impl NativePileupEngine {
    pub fn new() -> Self {
        Self
    }

    pub fn generate_records(
        &self,
        reference: &Path,
        bam_files: &[PathBuf],
        regions: Option<&str>,
        min_base_quality: u8,
        min_mapping_quality: u8,
    ) -> Result<Vec<PileupRecord>> {
        #[cfg(feature = "htslib")]
        {
            return generate_records_htslib(
                reference,
                bam_files,
                regions,
                min_base_quality,
                min_mapping_quality,
            );
        }
        #[cfg(not(feature = "htslib"))]
        {
            let _ = (reference, bam_files, regions, min_base_quality, min_mapping_quality);
            bail!("native pileup requires building cropability-native with feature `htslib` (maturin develop --features python,htslib)")
        }
    }
}

#[cfg(feature = "htslib")]
fn generate_records_htslib(
    reference: &Path,
    bam_files: &[PathBuf],
    regions: Option<&str>,
    min_base_quality: u8,
    min_mapping_quality: u8,
) -> Result<Vec<PileupRecord>> {
    use rust_htslib::bam::Read;
    use rust_htslib::faidx::Reader as FaReader;

    let (chrom_filter, start, end) = parse_region(regions);
    let ref_reader = FaReader::from_path(reference)?;
    let sample_names: Vec<String> = bam_files
        .iter()
        .map(|p| {
            p.file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("sample")
                .to_string()
        })
        .collect();

    let mut site_table: BTreeMap<(String, u32), BTreeMap<String, PileupSample>> = BTreeMap::new();
    let mut ref_table: BTreeMap<(String, u32), String> = BTreeMap::new();

    for (sample_name, bam_path) in sample_names.iter().zip(bam_files.iter()) {
        let mut bam = rust_htslib::bam::Reader::from_path(bam_path)?;
        let header = bam.header().clone();
        let mut pileups = bam.pileup();
        pileups.set_max_depth(100_000);

        for pileup in pileups {
            let pileup = pileup?;
            let tid = pileup.tid();
            let chrom = String::from_utf8_lossy(header.tid2name(tid)).to_string();
            if let Some(ref cf) = chrom_filter {
                if &chrom != cf {
                    continue;
                }
            }
            let pos0 = pileup.pos();
            let pos1 = pos0 + 1;
            if let Some(s) = start {
                if pos0 < s {
                    continue;
                }
            }
            if let Some(e) = end {
                if pos0 >= e {
                    continue;
                }
            }
            let key = (chrom.clone(), pos1);
            if !ref_table.contains_key(&key) {
                let seq = ref_reader.fetch(&chrom, pos0, pos0 + 1)?;
                let rb = if seq.is_empty() {
                    "N".to_string()
                } else {
                    seq.to_uppercase()
                };
                ref_table.insert(key.clone(), rb);
            }
            let ref_base = ref_table.get(&key).cloned().unwrap_or_else(|| "N".into());
            let sample_pileup = count_column(&pileup, &ref_base, min_base_quality, min_mapping_quality);
            site_table.entry(key).or_default().insert(sample_name.clone(), sample_pileup);
        }
    }

    let mut out = Vec::new();
    for ((chrom, pos1), mut sample_data) in site_table {
        for sn in &sample_names {
            sample_data.entry(sn.clone()).or_insert_with(empty_sample);
        }
        let ref_base = ref_table.get(&(chrom.clone(), pos1)).cloned().unwrap_or_else(|| "N".into());
        out.push(PileupRecord {
            chrom,
            pos: pos1,
            ref_base,
            samples: sample_data,
        });
    }
    out.sort_by(|a, b| a.chrom.cmp(&b.chrom).then(a.pos.cmp(&b.pos)));
    Ok(out)
}

#[cfg(feature = "htslib")]
fn empty_sample() -> PileupSample {
    PileupSample {
        depth: 0,
        base_counts: BASES.iter().map(|b| (b.to_string(), 0)).collect(),
        insertions: 0,
        deletions: 0,
    }
}

#[cfg(feature = "htslib")]
fn count_column(
    pileup: &rust_htslib::bam::pileup::Pileup<'_>,
    ref_base: &str,
    min_base_quality: u8,
    min_mapping_quality: u8,
) -> PileupSample {
    let mut counts: BTreeMap<String, u32> = BASES.iter().map(|b| (b.to_string(), 0)).collect();
    let mut insertions = 0u32;
    let mut deletions = 0u32;
    let mut depth = 0u32;
    let ref_u = ref_base.to_uppercase();
    for alignment in pileup.alignments() {
        if alignment.is_del() {
            deletions += 1;
            continue;
        }
        if alignment.is_refskip() {
            continue;
        }
        let record = alignment.record();
        if record.mapq() < min_mapping_quality {
            continue;
        }
        let qpos = match alignment.qpos() {
            Some(q) => q,
            None => continue,
        };
        let quals = record.qual();
        if qpos >= quals.len() || quals[qpos] < min_base_quality {
            continue;
        }
        depth += 1;
        let indel = alignment.indel();
        if indel > 0 {
            insertions += 1;
        } else if indel < 0 {
            deletions += 1;
        }
        let base_byte = record.seq()[qpos as usize];
        let b = if let Ok(ch) = std::str::from_utf8(&[base_byte]) {
            ch.to_uppercase()
        } else {
            "N".into()
        };
        let b = if BASES.contains(&b.as_str()) {
            b
        } else {
            "N".into()
        };
        if b == ref_u {
            *counts.entry(ref_u.clone()).or_insert(0) += 1;
        } else {
            *counts.entry(b).or_insert(0) += 1;
        }
    }
    PileupSample {
        depth,
        base_counts: counts,
        insertions,
        deletions,
    }
}

fn parse_region(region: Option<&str>) -> (Option<String>, Option<u32>, Option<u32>) {
    let Some(r) = region else {
        return (None, None, None);
    };
    if !r.contains(':') {
        return (Some(r.to_string()), None, None);
    }
    let parts: Vec<&str> = r.splitn(2, ':').collect();
    let chrom = parts[0].to_string();
    if parts.len() < 2 || !parts[1].contains('-') {
        return (Some(chrom), None, None);
    }
    let span: Vec<&str> = parts[1].splitn(2, '-').collect();
    let start: u32 = span[0].replace(',', "").parse().unwrap_or(1);
    let end: u32 = span.get(1).and_then(|s| s.replace(',', "").parse().ok()).unwrap_or(0);
    (Some(chrom), Some(start.saturating_sub(1)), Some(end))
}

pub fn write_pileup_summary_tsv(
    path: &Path,
    records: Vec<PileupRecord>,
    min_depth: u32,
    min_alt_freq: f64,
) -> Result<u32> {
    let parser = MpileupParser::new(None);
    let summaries = parser.summarize_sites(records, min_depth, min_alt_freq);
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut f = File::create(path)?;
    writeln!(f, "#CHROM\tPOS\tREF\tDP\tALT\tAC\tAF")?;
    for s in &summaries {
        writeln!(
            f,
            "{}\t{}\t{}\t{}\t{}\t{}\t{:.6}",
            s.chrom,
            s.pos,
            s.ref_base,
            s.depth,
            s.alt_base.as_deref().unwrap_or("."),
            s.alt_count,
            s.alt_freq
        )?;
    }
    Ok(summaries.len() as u32)
}
