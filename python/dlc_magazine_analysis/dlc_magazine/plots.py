"""Shared colours and the per-session summary figure."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from . import settings

CUE_ORDER = ["Cue1p", "Cue4p"]
CUE_LABELS = {"Cue1p": "1 pellet", "Cue4p": "4 pellets"}
CUE_COLORS = {"Cue1p": "#B22222", "Cue4p": "#1F78B4"}   # 1p red, 4p blue
AWAY = "#e4e4e0"
AT_MAGAZINE = "#5fb86a"
HEAD_IN = "#e8811a"
ROTATION = "#d63a92"
TEXT = "#222222"
TEXT_DIM = "#6b6b66"
GRID = "#e6e6e6"
REWARD_S = 5.0          # pellet delivery, 5 s after cue onset

STATE_CMAP = ListedColormap([AWAY, AT_MAGAZINE, ROTATION])


def style(ax: plt.Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)


def mark_epochs(ax: plt.Axes) -> None:
    ax.axvline(0, color=TEXT, lw=0.9, ls="--", alpha=0.7)
    ax.axvline(REWARD_S, color=TEXT_DIM, lw=0.9, ls=":", alpha=0.9)


def state_matrix(table: pd.DataFrame, cue: str) -> np.ndarray:
    """trials x frames: 0 away, 1 at magazine, 2 head-anchored rotation bout."""
    block = table[table["trial_type"] == cue]
    state = block["at_magazine"].astype(int) + block["rotation_bout"].astype(int)
    return (block.assign(state=state).pivot(index="trial", columns="frame", values="state")
            .to_numpy())


def raster(ax: plt.Axes, table: pd.DataFrame, cue: str, fps: float = settings.FPS) -> None:
    matrix = state_matrix(table, cue)
    t0, t1 = -settings.COURSE_PRE_S, settings.COURSE_POST_S
    ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=STATE_CMAP, vmin=0, vmax=2,
              extent=[t0, t1, len(matrix), 0])
    ax.axvline(0, color=TEXT, lw=0.9, ls="--")
    ax.axvline(REWARD_S, color=TEXT, lw=0.9, ls=":")
    ax.set_ylabel(f"{CUE_LABELS[cue]} trials")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.spines[["top", "right"]].set_visible(False)


COURSES = [
    ("p_at_magazine", "P(at magazine)"),
    ("p_head_in_magazine", "P(head in magazine)"),
    ("p_rotation_given_at", "P(rotation | at magazine)"),
    ("speed_cm_s", "whole-body speed (cm/s)"),
]


def course(ax: plt.Axes, curves: pd.DataFrame, column: str, ylabel: str) -> None:
    for cue in CUE_ORDER:
        c = curves[curves["trial_type"] == cue]
        if len(c):
            ax.plot(c["time_s"], c[column], color=CUE_COLORS[cue], lw=2, label=CUE_LABELS[cue])
    mark_epochs(ax)
    ax.set_xlim(-settings.COURSE_PRE_S, settings.COURSE_POST_S)
    ax.set_ylabel(ylabel)
    style(ax)


def session_figure(table: pd.DataFrame, curves: pd.DataFrame, conditions: pd.DataFrame,
                   title: str = "") -> plt.Figure:
    fig = plt.figure(figsize=(16, 8.5))
    grid = fig.add_gridspec(2, 4, hspace=0.38, wspace=0.32)
    for i, (column, label) in enumerate(COURSES):
        ax = fig.add_subplot(grid[0, i])
        course(ax, curves, column, label)
        ax.set_xlabel("time from cue onset (s)")
        if i == 0:
            ax.legend(frameon=False, loc="upper left")
    for i, cue in enumerate(CUE_ORDER):
        ax = fig.add_subplot(grid[1, i])
        raster(ax, table, cue)
        ax.set_xlabel("time from cue onset (s)")
    ax = fig.add_subplot(grid[1, 2:])
    measures = [("magazine_pct", "at magazine\n(% of 0-5 s)"),
                ("head_in_magazine_pct", "head in magazine\n(% of 0-5 s)"),
                ("rotation_pct_of_magazine", "rotation\n(% of magazine time)")]
    width = 0.38
    for j, cue in enumerate(CUE_ORDER):
        row = conditions[conditions["trial_type"] == cue]
        if row.empty:
            continue
        values = [float(row[m].iloc[0]) for m, _ in measures]
        x = np.arange(len(measures)) + (j - 0.5) * width
        ax.bar(x, values, width * 0.92, color=CUE_COLORS[cue], label=CUE_LABELS[cue])
        for xi, v in zip(x, values):
            ax.text(xi, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=TEXT)
    ax.set_xticks(np.arange(len(measures)), [l for _, l in measures])
    ax.set_ylabel("%")
    ax.set_ylim(0, 120)
    ax.set_yticks(range(0, 101, 20))
    style(ax)
    ax.legend(handles=[Patch(color=AWAY, label="away"), Patch(color=AT_MAGAZINE, label="at magazine"),
                       Patch(color=ROTATION, label="rotation bout")],
              frameon=False, loc="upper center", ncol=3, title="raster colours")
    if title:
        fig.suptitle(title, fontsize=13, x=0.06, ha="left")
    return fig
