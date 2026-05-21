//! Minimal VCF writer (CPU).

use anyhow::Result;
use std::fs::File;
use std::io::Write;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct VcfRecord {
    pub chrom: String,
    pub pos: u32,
    pub ref_allele: String,
    pub alt: Vec<String>,
    pub info: Vec<(String, String)>,
    pub format_keys: Vec<String>,
    pub sample_values: Vec<String>,
}

pub struct VcfWriter {
    file: File,
    samples: Vec<String>,
    header_written: bool,
}

impl VcfWriter {
    pub fn create(path: impl AsRef<Path>, sample_names: &[String]) -> Result<Self> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        Ok(Self {
            file: File::create(path)?,
            samples: sample_names.to_vec(),
            header_written: false,
        })
    }

    pub fn write_header(&mut self, source: &str, extra_meta: &[String]) -> Result<()> {
        writeln!(self.file, "##fileformat=VCFv4.2")?;
        writeln!(self.file, "##source={source}")?;
        for m in extra_meta {
            writeln!(self.file, "{m}")?;
        }
        let fmt = if self.samples.is_empty() {
            "CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO".to_string()
        } else {
            format!(
                "CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}",
                self.samples.join("\t")
            )
        };
        writeln!(self.file, "#{}", fmt)?;
        self.header_written = true;
        Ok(())
    }

    pub fn write_record(&mut self, rec: &VcfRecord) -> Result<()> {
        if !self.header_written {
            self.write_header("CropAbility", &[])?;
        }
        let info = rec
            .info
            .iter()
            .map(|(k, v)| format!("{k}={v}"))
            .collect::<Vec<_>>()
            .join(";");
        let alt = rec.alt.join(",");
        if self.samples.is_empty() {
            writeln!(
                self.file,
                "{}\t{}\t.\t{}\t{}\t.\t.\t{}",
                rec.chrom, rec.pos, rec.ref_allele, alt, info
            )?;
        } else {
            let fmt = rec.format_keys.join(":");
            let samples = rec.sample_values.join("\t");
            writeln!(
                self.file,
                "{}\t{}\t.\t{}\t{}\t.\t.\t{}\t{}\t{}",
                rec.chrom, rec.pos, rec.ref_allele, alt, info, fmt, samples
            )?;
        }
        Ok(())
    }
}
