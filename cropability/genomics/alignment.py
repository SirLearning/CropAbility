"""
GPU 加速序列比对
================
Smith-Waterman 局部比对算法的 GPU 批处理实现。
支持批量查询序列与数据库序列的快速比对评分矩阵计算。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from cropability.kernels.seq import encode_sequences
from cropability.utils.logging import get_logger
from cropability.utils.timer import Timer

logger = get_logger(__name__)


@dataclass
class AlignmentResult:
    """局部比对结果。"""
    query_idx: int
    target_idx: int
    score: float
    query_start: int
    query_end: int
    target_start: int
    target_end: int
    cigar: Optional[str] = None  # 暂时占位

    def __repr__(self) -> str:
        return (
            f"Align(q={self.query_idx}[{self.query_start}:{self.query_end}] "
            f"t={self.target_idx}[{self.target_start}:{self.target_end}] "
            f"score={self.score:.1f})"
        )


class SmithWatermanGPU:
    """
    GPU 批量 Smith-Waterman 局部比对。

    策略：利用 PyTorch 向量化计算所有查询-目标对的得分矩阵，
    并行追踪最高得分位置。适合中短读长序列（<= 1000 bp）的大批量筛选。

    Args:
        device        : GPU 设备
        match_score   : 碱基匹配得分
        mismatch_score: 碱基不匹配惩罚（负值）
        gap_open      : 开gap惩罚（负值）
        gap_extend    : 延伸gap惩罚（负值）
    """

    def __init__(
        self,
        device: Optional[torch.device] = None,
        match_score: float = 2.0,
        mismatch_score: float = -1.0,
        gap_open: float = -2.0,
        gap_extend: float = -0.5,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.match = match_score
        self.mismatch = mismatch_score
        self.gap_open = gap_open
        self.gap_extend = gap_extend
        logger.info(
            f"SmithWatermanGPU on {device}: "
            f"match={match_score}, mismatch={mismatch_score}"
        )

    def score_matrix(
        self,
        queries: List[str],
        targets: List[str],
    ) -> torch.Tensor:
        """
        计算所有 query-target 对的最优局部比对得分。

        Args:
            queries : 查询序列列表（长度 <= 1000）
            targets : 目标序列列表

        Returns:
            [Q, T] float32 得分矩阵
        """
        q_enc = encode_sequences(queries, device=self.device)   # [Q, Lq]
        t_enc = encode_sequences(targets, device=self.device)   # [T, Lt]
        Q, Lq = q_enc.shape
        T, Lt = t_enc.shape

        logger.info(f"Smith-Waterman: {Q} queries × {T} targets, Lq={Lq}, Lt={Lt}")

        with Timer("sw_score") as timer:
            scores = self._batch_sw_score(q_enc, t_enc)

        logger.info(f"SW scoring done in {timer.elapsed_ms:.1f} ms")
        return scores

    def _batch_sw_score(
        self,
        q_enc: torch.Tensor,   # [Q, Lq]
        t_enc: torch.Tensor,   # [T, Lt]
    ) -> torch.Tensor:
        """
        向量化 Smith-Waterman 评分（线性 gap 惩罚，无 traceback）。
        使用对角线扫描（anti-diagonal parallelism）。
        """
        Q, Lq = q_enc.shape
        T, Lt = t_enc.shape

        # 替换矩阵：相同碱基给 match，不同给 mismatch，N 碱基给 0
        # 预计算 [Q, T, Lq, Lt] 替换得分（可能内存较大，分块处理）
        # 简化版：用对角线推进，[Q*T, 1] 状态 tensor

        # 为避免 Q*T*Lq*Lt 内存爆炸，改用逐列扫描 + GPU 并行
        # H[i,j] = max(0, H[i-1,j-1]+s(q[i],t[j]), H[i-1,j]+gap, H[i,j-1]+gap)

        # 广播扩展：[Q, 1, Lq] vs [1, T, Lt] → [Q, T, Lq, Lt] 太大
        # 采用逐列迭代，每步处理整个 [Q, T] 矩阵一列
        H = torch.zeros(Q, T, Lq + 1, device=self.device)   # 仅保留滑动窗口
        best = torch.zeros(Q, T, device=self.device)

        for j in range(Lt):
            target_col = t_enc[:, j].long()           # [T]
            # 计算替换得分 sub[q, t, i] = match if q_enc[q,i]==target_col[t] else mismatch
            # q_enc[:, :] [Q, Lq]; target_col [T]
            q_bases = q_enc.long()                    # [Q, Lq]
            t_base_j = target_col.unsqueeze(0).unsqueeze(2)  # [1, T, 1]
            q_bases_exp = q_bases.unsqueeze(1)        # [Q, 1, Lq]
            match_mask = (q_bases_exp == t_base_j)    # [Q, T, Lq]
            n_mask = (q_bases_exp == 4) | (t_base_j == 4)
            sub = torch.where(
                n_mask, torch.zeros_like(match_mask, dtype=torch.float32),
                torch.where(match_mask,
                            torch.full_like(match_mask, self.match, dtype=torch.float32),
                            torch.full_like(match_mask, self.mismatch, dtype=torch.float32))
            )  # [Q, T, Lq]

            # DP 递推 (简化线性gap)
            H_prev = H[:, :, :-1]   # [Q, T, Lq]
            H_new = (H_prev + sub).clamp(min=0)
            # gap 惩罚（简化为均一惩罚，不区分开/延伸）
            H_new = torch.max(H_new, (H[:, :, 1:] + self.gap_open).clamp(min=0))

            H[:, :, 1:] = H_new
            best = torch.max(best, H_new.max(dim=2).values)

        return best

    def find_top_hits(
        self,
        scores: torch.Tensor,
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> List[Tuple[int, int, float]]:
        """
        找到得分矩阵中的 top-k 比对结果。

        Args:
            scores   : [Q, T] 得分矩阵
            top_k    : 每条查询返回的最佳目标数
            threshold: 最低得分阈值

        Returns:
            (query_idx, target_idx, score) 三元组列表
        """
        Q = scores.shape[0]
        top_scores, top_idx = torch.topk(scores, min(top_k, scores.shape[1]), dim=1)
        hits = []
        for q in range(Q):
            for k in range(top_scores.shape[1]):
                s = float(top_scores[q, k].item())
                if s >= threshold:
                    hits.append((q, int(top_idx[q, k].item()), s))
        return hits
