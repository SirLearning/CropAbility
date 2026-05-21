"""GPU result visualization helpers."""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def plot_speedup(
    sizes: Sequence[int],
    times_ms: Sequence[float],
    title: str = "GPU benchmark",
    output: Optional[str] = None,
) -> None:
    """Bar chart of kernel runtime vs problem size."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(s) for s in sizes], times_ms, color="steelblue")
    ax.set_xlabel("Problem size")
    ax.set_ylabel("Time (ms)")
    ax.set_title(title)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()
    plt.close(fig)


def plot_ld_heatmap(
    matrix: np.ndarray,
    title: str = "LD r² matrix",
    output: Optional[str] = None,
) -> None:
    """Heatmap for LD results computed on GPU."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if output:
        fig.savefig(output, dpi=150)
    else:
        plt.show()
    plt.close(fig)
