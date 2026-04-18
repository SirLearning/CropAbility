"""
TorchScript 模型导出
=====================
定义可被 TorchScript 序列化的模型（供 Java TritonIntegration / C++ libtorch 加载）。
Triton 内核不可序列化，导出版本使用等价 PyTorch 算子。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch
import torch.nn as nn

from cropability.utils.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 基础算术模型（向下兼容原有 pgl 模块的 add 语义）
# ---------------------------------------------------------------------------

class AddModule(nn.Module):
    """
    向量逐元素加法模块。
    等价于原 TritonAddModule，保持与 Java TritonIntegration 接口兼容。
    """

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return x + y


# ---------------------------------------------------------------------------
# 基因组序列嵌入模型
# ---------------------------------------------------------------------------

class GenomicEmbedding(nn.Module):
    """
    DNA 序列嵌入网络（可 TorchScript 导出）。

    架构：
        int8 编码序列 → Embedding(5, embed_dim) → 1D Conv → 全局池化 → 线性投影

    Args:
        embed_dim  : 碱基嵌入维度
        hidden_dim : 卷积隐层维度
        output_dim : 输出向量维度
        kernel_size: 卷积核大小（感受野 = 局部 motif 长度）
    """

    def __init__(
        self,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        output_dim: int = 256,
        kernel_size: int = 9,
    ) -> None:
        super().__init__()
        # 5 种碱基：A/C/G/T/N
        self.embedding = nn.Embedding(5, embed_dim, padding_idx=4)
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, L] int64 编码序列（0-4）

        Returns:
            [B, output_dim] 序列嵌入向量
        """
        emb = self.embedding(x)          # [B, L, embed_dim]
        emb = emb.transpose(1, 2)        # [B, embed_dim, L]
        h = self.conv(emb)               # [B, hidden_dim, L]
        pooled = self.pool(h).squeeze(2) # [B, hidden_dim]
        return self.proj(pooled)          # [B, output_dim]


class SNPEffectPredictor(nn.Module):
    """
    SNP 效应预测模型（二分类：有害/中性）。

    输入：SNP 位点周围的序列嵌入 + 基因组特征
    输出：效应概率
    """

    def __init__(
        self,
        seq_embed_dim: int = 256,
        genomic_features: int = 32,
    ) -> None:
        super().__init__()
        self.seq_encoder = GenomicEmbedding(output_dim=seq_embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(seq_embed_dim + genomic_features, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(
        self, seq: torch.Tensor, features: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            seq     : [B, L] int64 序列
            features: [B, genomic_features] 额外特征

        Returns:
            [B] 效应概率
        """
        seq_emb = self.seq_encoder(seq)                    # [B, seq_embed_dim]
        combined = torch.cat([seq_emb, features], dim=1)   # [B, seq_embed_dim+F]
        return self.classifier(combined).squeeze(1)         # [B]


# ---------------------------------------------------------------------------
# 导出/加载工具函数
# ---------------------------------------------------------------------------

def export_model(
    model: nn.Module,
    path: Union[str, Path],
    example_inputs: Optional[tuple] = None,
    optimize: bool = True,
) -> Path:
    """
    将 PyTorch 模型导出为 TorchScript（用于 Java/C++ 推理）。

    Args:
        model         : 待导出的模型
        path          : 输出 .pt 文件路径
        example_inputs: 用于 trace 的示例输入（None 则使用 script 模式）
        optimize      : 是否优化导出图

    Returns:
        导出文件路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    model.eval()
    with torch.no_grad():
        if example_inputs is not None:
            logger.info(f"TorchScript trace export → {path}")
            scripted = torch.jit.trace(model, example_inputs)
        else:
            logger.info(f"TorchScript script export → {path}")
            scripted = torch.jit.script(model)

        if optimize:
            scripted = torch.jit.optimize_for_inference(scripted)

        scripted.save(str(path))

    logger.info(f"Model exported: {path} ({path.stat().st_size / 1024:.1f} KB)")
    return path


def load_model(path: Union[str, Path], device: Optional[torch.device] = None) -> torch.jit.ScriptModule:
    """
    加载 TorchScript 模型。

    Args:
        path  : .pt 文件路径
        device: 目标设备（None 则保持原设备）

    Returns:
        加载好的 ScriptModule
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    model = torch.jit.load(str(path), map_location=device)
    model.eval()
    logger.info(f"Model loaded from {path}")
    return model
