"""Plotting helpers shared by the phase scripts."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from metamorph.model import (
    IDX_A_HEPATIC,
    IDX_A_RENAL,
    IDX_B_BARE,
    IDX_B_COAT,
    SLICE_T_BARE,
    SLICE_T_COAT,
    TISSUES,
)

# Validated categorical colors (blue, orange) and neutral ink for totals.
COLOR_BARE = "#2a78d6"
COLOR_COAT = "#eb6834"
COLOR_TOTAL = "#0b0b0b"
COLOR_MUTED = "#52514e"


def apply_style() -> None:
    """Set a plain, publication-like Matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 200,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": COLOR_MUTED,
            "axes.labelcolor": COLOR_TOTAL,
            "xtick.color": COLOR_MUTED,
            "ytick.color": COLOR_MUTED,
            "axes.grid": True,
            "grid.color": "#e6e5e0",
            "grid.linewidth": 0.6,
            "lines.linewidth": 2.0,
            "legend.frameon": False,
        }
    )


def plot_trajectories(ts: np.ndarray, ys: np.ndarray, title: str | None = None) -> Figure:
    """Plot bare, coated and total amounts per compartment, plus cleared amounts.

    Args:
        ts: Times in hours, shape ``(n,)``.
        ys: States from ``simulate``, shape ``(n, 14)``.
        title: Optional figure title.

    Returns:
        The Matplotlib figure (2 x 4 small multiples).
    """
    apply_style()
    ys = np.asarray(ys)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5.2), sharex=True)
    axes = axes.ravel()

    panels = [("Blood", ys[:, IDX_B_BARE], ys[:, IDX_B_COAT])]
    for i, tissue in enumerate(TISSUES):
        panels.append(
            (tissue.capitalize(), ys[:, SLICE_T_BARE.start + i], ys[:, SLICE_T_COAT.start + i])
        )

    for ax, (name, bare, coat) in zip(axes, panels):
        ax.plot(ts, bare + coat, color=COLOR_TOTAL, label="total")
        ax.plot(ts, bare, color=COLOR_BARE, label="bare")
        ax.plot(ts, coat, color=COLOR_COAT, label="coated")
        ax.set_title(name, loc="left")
        ax.set_ylim(bottom=0)

    ax = axes[6]
    ax.plot(ts, ys[:, IDX_A_RENAL], color=COLOR_BARE, label="renal (urine)")
    ax.plot(ts, ys[:, IDX_A_HEPATIC], color=COLOR_COAT, label="hepatic")
    ax.set_title("Cumulative cleared", loc="left")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center right")

    ax = axes[7]
    ax.plot(ts, ys.sum(axis=1) - ys[0].sum(), color=COLOR_TOTAL)
    ax.set_title("Mass balance error (sum - dose)", loc="left")
    ax.set_ylabel("%ID")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    for ax in axes[4:]:
        ax.set_xlabel("time (h)")
    for ax in axes[[0, 4]]:
        ax.set_ylabel("amount (%ID)")
    axes[0].legend(loc="upper right")
    if title:
        fig.suptitle(title, x=0.01, ha="left")
    fig.tight_layout()
    return fig
