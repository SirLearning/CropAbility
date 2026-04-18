"""
植物基因组分析模块
==================
提供以下核心分析功能：

- variant  : 变异检测（SNP/Indel）
- ld       : 连锁不平衡（LD）分析
- gwas     : 全基因组关联分析（GWAS）辅助工具
- alignment: 序列比对评分
"""

from cropability.genomics.variant import VariantCaller, SNPResult
from cropability.genomics.ld import LDCalculator, LDResult
from cropability.genomics.gwas import GWASEngine, GWASResult
from cropability.genomics.alignment import SmithWatermanGPU

__all__ = [
    "VariantCaller",
    "SNPResult",
    "LDCalculator",
    "LDResult",
    "GWASEngine",
    "GWASResult",
    "SmithWatermanGPU",
]
