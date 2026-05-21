//! FASTA/FASTQ reader (CPU).

use anyhow::{bail, Result};
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

pub struct FastaReader {
    path: PathBuf,
    is_fastq: bool,
}

impl FastaReader {
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref().to_path_buf();
        if !path.exists() {
            bail!("Sequence file not found: {}", path.display());
        }
        let lower = path.to_string_lossy().to_lowercase();
        let is_fastq = lower.ends_with(".fq")
            || lower.ends_with(".fastq")
            || lower.ends_with(".fq.gz")
            || lower.ends_with(".fastq.gz");
        if lower.ends_with(".gz") {
            bail!(
                "gzip not supported in native reader yet; decompress first: {}",
                path.display()
            );
        }
        Ok(Self { path, is_fastq })
    }

    pub fn read_all(&self) -> Result<Vec<(String, String)>> {
        let file = File::open(&self.path)?;
        let reader = BufReader::new(file);
        if self.is_fastq {
            read_fastq(reader)
        } else {
            read_fasta(reader)
        }
    }
}

fn read_fasta<R: BufRead>(reader: R) -> Result<Vec<(String, String)>> {
    let mut out = Vec::new();
    let mut name: Option<String> = None;
    let mut buf = String::new();
    for line in reader.lines() {
        let line = line?;
        if line.starts_with('>') {
            if let Some(n) = name.take() {
                if !buf.is_empty() {
                    out.push((n, std::mem::take(&mut buf)));
                }
            }
            name = Some(line[1..].split_whitespace().next().unwrap_or("seq").to_string());
        } else if !line.is_empty() {
            if name.is_some() {
                buf.push_str(&line);
            }
        }
    }
    if let Some(n) = name {
        if !buf.is_empty() {
            out.push((n, buf));
        }
    }
    Ok(out)
}

fn read_fastq<R: BufRead>(reader: R) -> Result<Vec<(String, String)>> {
    let mut lines = reader.lines();
    let mut out = Vec::new();
    while let Some(h) = lines.next() {
        let h = h?;
        if !h.starts_with('@') {
            bail!("invalid FASTQ");
        }
        let name = h[1..].split_whitespace().next().unwrap_or("seq").to_string();
        let seq = lines.next().ok_or_else(|| anyhow::anyhow!("truncated FASTQ"))??;
        let _ = lines.next();
        let _ = lines.next();
        out.push((name, seq));
    }
    Ok(out)
}
