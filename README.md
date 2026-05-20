# CropAbility

**植物基因组高性能 GPU 计算框架**  
*Plant Genomics High-Performance GPU Computing Framework*

基于 [Triton](https://github.com/openai/triton) / PyTorch 的 GPU 加速植物基因组分析工具包，
针对 **NVIDIA H100 PCIe** 和 **A2** 双卡环境优化，同时保持 CPU 兼容性。

---

## 架构概览

```
cropability/
├── gpu/           # GPU 设备管理与分布式支持（H100/A2 多卡）
├── kernels/       # Triton JIT GPU 内核
│   ├── seq.py     #   序列编码、GC含量、反向互补、k-mer
│   ├── stats.py   #   Welford 均值/方差、z-score、Pearson 相关
│   ├── matrix.py  #   对称矩阵乘法、批量外积
│   └── pairwise.py#   Hamming 距离矩阵、Jaccard 相似度
├── genomics/      # 植物基因组分析算法
│   ├── variant.py #   SNP/Indel 检测
│   ├── ld.py      #   连锁不平衡（LD）分析
│   ├── gwas.py    #   全基因组关联分析（GWAS）
│   └── alignment.py#  Smith-Waterman 序列比对
├── models/        # TorchScript 模型（供 Java/C++ 调用）
├── io/            # FASTA/FASTQ/VCF 读写
├── utils/         # 配置管理、日志、计时
└── cli/           # 命令行工具
```

Java 集成层（`src/main/java/`）通过 PyTorch Java API 加载 TorchScript 模型，
与 Python GPU 计算结果无缝交互。

---

## 快速开始

### 环境要求

| 依赖 | 版本 |
|------|------|
| Python | ≥ 3.9 |
| PyTorch | ≥ 2.1 (CUDA 12.x) |
| Triton | ≥ 2.1 |
| CUDA Driver | ≥ 520 (H100 需要 ≥ 525) |
| Java (可选) | 11+ |

### 安装

```bash
# 克隆仓库
git clone https://github.com/example/CropAbility.git
cd CropAbility

# 安装 Python 包（开发模式）
pip install -e ".[gpu,dev]"

# 或者仅安装核心依赖
pip install -r requirements.txt
pip install -e .
```

### 基本使用

```bash
# 查看系统和 GPU 信息
cropability info

# GPU 性能基准测试
cropability benchmark --n-seqs 5000 --seq-len 512 --matrix-size 8192

# 从 FASTA 文件检测 SNP
cropability snp -i samples.fa --min-af 0.05 --min-depth 10

# 计算 LD 矩阵
cropability ld --n-samples 500 --n-snps 1000

# 运行原生 mpileup（真实 BAM/CRAM 输入）
cropability pileup -r ref.fa -b sample1.bam sample2.bam -o cohort.mpileup

# 运行变异检测流程（默认 hybrid: 原生 mpileup + 原生 FastCall3 逻辑）
cropability call-variants -r ref.fa -b sample1.bam sample2.bam -o cohort.vcf --mode hybrid

# 导出 TorchScript 模型（供 Java 调用）
cropability export --model add --output model.pt
cropability export --model embedding --output embed.pt
```

> NGS 流程默认在 CropAbility 内部执行，不依赖外部 `samtools/FastCall3` 命令。
> 运行时需要 `pysam`（`pip install "cropability[io]"`）。可选 Rust 后端可用于性能加速。

### Python API

```python
from cropability.gpu import get_device_manager
from cropability.kernels.seq import encode_sequences, gc_content_kernel
from cropability.kernels.stats import zscore_normalize, pearson_correlation
from cropability.genomics.variant import VariantCaller
from cropability.genomics.ld import LDCalculator
from cropability.genomics.gwas import GWASEngine

# GPU 设备管理（自动识别 H100/A2）
dm = get_device_manager(device_ids=[0, 1])
dm.print_memory_stats()

# 序列编码与 GC 含量计算
sequences = ["ATCGATCGATCG", "GCTAGCTAGCTA", "NNNATCGAAANN"]
encoded = encode_sequences(sequences, device=dm.get_primary_device())
gc = gc_content_kernel(encoded)

# SNP 检测
caller = VariantCaller(device=dm.get_primary_device())
snps = caller.call_snps(sample_seqs, reference)

# GWAS（GPU 加速线性回归）
import torch
engine = GWASEngine(device=dm.get_primary_device())
results = engine.run_linear_gwas(
    genotypes=torch.randint(0, 3, (1000, 50000)).float(),
    phenotype=torch.randn(1000),
)
hits = [r for r in results if r.is_significant()]
```

### 多 GPU 分布式计算

```python
from cropability.gpu.distributed import launch_ddp

def train_fn(rank: int, world_size: int, **kwargs):
    from cropability.gpu.distributed import wrap_ddp
    model = wrap_ddp(MyModel())
    # ... 训练循环 ...

# 在 2 块 H100/A2 上启动
launch_ddp(train_fn, num_gpus=2)
```

### Java 集成

```bash
# 1. 导出模型
cropability export --model add --output model.pt

# 2. 编译 Java 代码
mvn package -DskipTests

# 3. 运行集成测试
java -Djava.library.path=pytorch_java_libs \
     -jar target/CropAbility-0.1.0-fat.jar model.pt
```

```java
// Java 调用示例
try (TritonIntegration integration = new TritonIntegration("model.pt")) {
    float[] x = {1.0f, 2.0f, 3.0f};
    float[] y = {4.0f, 5.0f, 6.0f};
    float[] result = integration.add(x, y);  // [5.0, 7.0, 9.0]
}
```

---

## 测试

```bash
# 运行所有测试
pytest

# 带覆盖率报告
pytest --cov=cropability --cov-report=html

# 仅运行 GPU 相关测试（需要 CUDA）
pytest -m gpu

# 并行加速
pytest -n auto
```

---

## 配置

在项目根目录创建 `cropability.yaml` 可覆盖默认设置：

```yaml
gpu:
  device_ids: [0, 1]
  memory_fraction: 0.90
  mixed_precision: true

compute:
  batch_size: 256
  block_size: 2048

genomics:
  min_base_quality: 20
  kmer_size: 31

logging:
  level: INFO
  file: cropability.log
```

也可通过环境变量覆盖（格式 `CROPABILITY_<SECTION>__<KEY>`）：

```bash
export CROPABILITY_GPU__DEVICE_IDS=0,1
export CROPABILITY_COMPUTE__BATCH_SIZE=512
```

---

## 模块说明

### `cropability.gpu`

- `DeviceManager`：枚举 GPU（H100/A2 自动识别），管理显存配额，提供最空闲设备选择
- `get_device_manager()`：全局单例，首次调用自动初始化
- `launch_ddp()`：单机多卡 DDP 训练启动器
- `wrap_ddp()`：模型包装为 DistributedDataParallel

### `cropability.kernels`

所有内核优先走 Triton GPU 路径，无 CUDA 时自动降级至 PyTorch CPU 实现。

| 内核 | 功能 |
|------|------|
| `encode_sequences` | ASCII → int8（A=0,C=1,G=2,T=3,N=4）|
| `gc_content_kernel` | 批量 GC 含量（含 N 过滤）|
| `reverse_complement_kernel` | GPU 批量反向互补 |
| `kmer_count_kernel` | k-mer 频率归一化向量 |
| `welford_mean_var` | 数值稳定均值/方差 |
| `zscore_normalize` | 行级 z-score 标准化 |
| `pearson_correlation` | [M,D]×[N,D] → [M,N] 相关矩阵 |
| `hamming_distance_matrix` | 成对 Hamming 距离 |
| `jaccard_similarity_matrix` | k-mer Jaccard 相似度 |

### `cropability.genomics`

| 类 | 功能 |
|-----|------|
| `VariantCaller` | 多样本 SNP 检测（频率/深度/Phred质量过滤）|
| `LDCalculator` | r² LD 矩阵（分块大矩阵支持）、LD pruning |
| `GWASEngine` | 线性 GWAS（OLS + PC协变量校正 + SVD-PCA）|
| `SmithWatermanGPU` | 批量 Smith-Waterman 比对评分矩阵 |

---

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 代码格式检查
ruff check cropability/ tests/

# 类型检查
mypy cropability/

# 构建 Java
mvn compile

# 全量测试
mvn test && pytest
```

---

## 许可证

MIT License — 详见 [LICENSE](LICENSE)

---

## 致谢

本项目基于以下开源工具构建：
- [OpenAI Triton](https://github.com/openai/triton)
- [PyTorch](https://pytorch.org)
- [PyTorch Java API](https://pytorch.org/javadoc/)
