"""
GPU 计算内核
============
基于 Triton JIT 的高性能 GPU 内核，涵盖植物基因组常用算法：

- seq      : 序列处理（碱基编码、互补、GC含量）
- stats    : 统计运算（均值、方差、协方差、z-score）
- matrix   : 矩阵运算（对称矩阵向量积、批量转置）
- pairwise : 成对距离与相似度矩阵
"""

from cropability.kernels.seq import (
    encode_sequences,
    gc_content_kernel,
    reverse_complement_kernel,
    kmer_count_kernel,
)
from cropability.kernels.stats import (
    welford_mean_var,
    zscore_normalize,
    pearson_correlation,
)
from cropability.kernels.matrix import (
    symm_matmul,
    batch_outer_product,
    triangular_sum,
)
from cropability.kernels.pairwise import (
    hamming_distance_matrix,
    jaccard_similarity_matrix,
)

__all__ = [
    # seq
    "encode_sequences",
    "gc_content_kernel",
    "reverse_complement_kernel",
    "kmer_count_kernel",
    # stats
    "welford_mean_var",
    "zscore_normalize",
    "pearson_correlation",
    # matrix
    "symm_matmul",
    "batch_outer_product",
    "triangular_sum",
    # pairwise
    "hamming_distance_matrix",
    "jaccard_similarity_matrix",
]
