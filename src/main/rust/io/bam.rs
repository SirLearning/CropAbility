//! BAM/CRAM input validation (CPU).

use anyhow::{bail, Result};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug)]
pub struct AlignmentFile {
    pub path: PathBuf,
    pub sample_name: String,
    pub fmt: String,
    pub has_index: bool,
}

pub struct AlignmentInputManager {
    paths: Vec<PathBuf>,
}

impl AlignmentInputManager {
    pub fn new(paths: Vec<PathBuf>) -> Result<Self> {
        if paths.is_empty() {
            bail!("At least one alignment file is required");
        }
        Ok(Self { paths })
    }

    pub fn collect(&self, require_index: bool) -> Result<Vec<AlignmentFile>> {
        let mut out = Vec::new();
        for p in &self.paths {
            if !p.exists() {
                bail!("Alignment file not found: {}", p.display());
            }
            let fmt = detect_format(p)?;
            let idx = resolve_index(p, &fmt);
            let has_index = idx.is_some();
            if require_index && !has_index {
                bail!("Missing index for {}", p.display());
            }
            out.push(AlignmentFile {
                path: p.clone(),
                sample_name: p.file_stem().and_then(|s| s.to_str()).unwrap_or("sample").to_string(),
                fmt,
                has_index,
            });
        }
        Ok(out)
    }
}

fn detect_format(path: &Path) -> Result<String> {
    match path.extension().and_then(|s| s.to_str()) {
        Some("bam") => Ok("bam".into()),
        Some("cram") => Ok("cram".into()),
        _ => bail!("Unsupported alignment format: {}", path.display()),
    }
}

fn resolve_index(path: &Path, fmt: &str) -> Option<PathBuf> {
    let candidates: Vec<PathBuf> = if fmt == "bam" {
        vec![path.with_extension("bai"), PathBuf::from(format!("{}.bai", path.display()))]
    } else {
        vec![path.with_extension("crai"), PathBuf::from(format!("{}.crai", path.display()))]
    };
    candidates.into_iter().find(|c| c.exists())
}
