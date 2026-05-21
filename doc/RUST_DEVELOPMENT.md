# CropAbility Rust development guide

## Role

**Python** is the host: PyTorch GPU, CLI, plotting.  
**Rust** is the native extension: CPU I/O, NGS pipeline, exposed as `cropability.native._core`.

There is **no** TorchScript or libtorch path in Rust.

## Layout

```
src/main/rust/
├── lib.rs
├── python.rs        # PyO3 exports
├── io/              # fasta, bam, vcf
└── genomics/        # pileup, fastcall3, pipeline
```

Root: `Cargo.toml` (single crate), `pyproject.toml` (`[tool.maturin]`).

## Features

| Feature | Enables |
|---------|---------|
| `python` | PyO3 → `cropability.native._core` |
| `htslib` | BAM pileup via `rust-htslib` |

## Build

```bash
conda env create -f environment.yml
conda activate cropability
pip install -e ".[gpu,dev,io,rust]"
maturin develop --release --features python,htslib
cargo test
```

## Python usage

```python
from cropability.native._core import FastaReader, VariantPipeline, QCThresholds

reader = FastaReader("samples.fa")
data = reader.read_all()

pipe = VariantPipeline()
report = pipe.run(
    mode="hybrid",
    reference="ref.fa",
    bam_files=["s1.bam"],
    output="out.vcf",
    qc=QCThresholds(),
)
```
