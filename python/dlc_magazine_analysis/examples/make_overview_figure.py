"""Build analysis_overview.pdf: one poster-sized page, one box per analysis step,
each with a short explanation and a worked example from the example session
(Cohort01 R205, fixed ITI, Day 9).

Run from this folder after analyze_session.py and make_trial_video.py have
filled example_output/ (see README.md):

    python make_overview_figure.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from dlc_magazine import analyze_session, settings                      # noqa: E402
from dlc_magazine import trials as trial_tools                          # noqa: E402
from dlc_magazine.dlc_io import calibration_point, read_dlc, read_trials  # noqa: E402
from dlc_magazine.magazine import MagazineFrame, zone_outlines_cm, zone_outlines_px  # noqa: E402
from dlc_magazine.plots import (AT_MAGAZINE, AWAY, CUE_COLORS, CUE_LABELS, CUE_ORDER, HEAD_IN,  # noqa: E402
                                ROTATION, TEXT, TEXT_DIM, course, raster, style)

DATA = HERE / "example_data"
OUT = HERE / "example_output"
BODY = DATA / "R205_fITI_D09_body_DLC.csv.gz"
CALIB = DATA / "R205_fITI_D09_calibration_DLC.csv.gz"
TRIALS = DATA / "R205_fITI_D09_trials.csv"
CLIP = DATA / "R205_fITI_D09_trial008_video_from_frame21548.mp4"
CLIP_FIRST = 21548
EXAMPLE_TRIAL = 8          # 4 pellets, lots of rotation
HYSTERESIS_TRIAL = 22      # 1 pellet, time in the 7-12 cm band while still "at the magazine"
OVERLAY = OUT / "videos" / "trial008_Cue4p.mp4"

FPS = settings.FPS
BOX_EDGE = "#c9c9c4"
BOX_FILL = "#fbfbf9"
HEAD_FILL = "#2f3b4a"
plt.rcParams.update({"font.size": 15, "axes.titlesize": 15, "axes.labelsize": 14,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "font.family": "DejaVu Sans",
                     "axes.facecolor": "none"})

# Page layout (figure fractions)
FIG_W, FIG_H = 33.1, 23.4                    # A1 landscape, inches
LEFT, RIGHT, TOP, BOTTOM = 0.012, 0.988, 0.915, 0.075
N_COLS, N_ROWS = 4, 3
GAP_X, GAP_Y = 0.010, 0.016
BOX_W = (RIGHT - LEFT - (N_COLS - 1) * GAP_X) / N_COLS
BOX_H = (TOP - BOTTOM - (N_ROWS - 1) * GAP_Y) / N_ROWS
HEADER_H = 0.028
TEXT_H = 0.062


def box(fig, number: int, title: str, text: str):
    """Draw box `number` (1-12, row-major) and return its example area (x, y, w, h)."""
    row, col = divmod(number - 1, N_COLS)
    x = LEFT + col * (BOX_W + GAP_X)
    y = TOP - (row + 1) * BOX_H - row * GAP_Y
    fig.patches.append(FancyBboxPatch((x, y), BOX_W, BOX_H, boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, fc=BOX_FILL, ec=BOX_EDGE, lw=1.5,
                                      zorder=-10))
    fig.patches.append(FancyBboxPatch((x, y + BOX_H - HEADER_H), BOX_W, HEADER_H,
                                      boxstyle="round,pad=0,rounding_size=0.006",
                                      transform=fig.transFigure, fc=HEAD_FILL, ec=HEAD_FILL,
                                      zorder=-9))
    fig.text(x + 0.008, y + BOX_H - HEADER_H / 2, f"{number}", color="#9fd0ff", fontsize=22,
             fontweight="bold", va="center")
    fig.text(x + 0.026, y + BOX_H - HEADER_H / 2, title, color="white", fontsize=20,
             fontweight="bold", va="center")
    fig.text(x + 0.008, y + BOX_H - HEADER_H - 0.006, text, fontsize=13.5, color=TEXT,
             va="top", ha="left", linespacing=1.35, wrap=False)
    return x + 0.008, y + 0.010, BOX_W - 0.016, BOX_H - HEADER_H - TEXT_H - 0.018


def sub(fig, area, rect):
    """Axes at `rect` = (x, y, w, h) given as fractions of the example area."""
    x, y, w, h = area
    return fig.add_axes([x + rect[0] * w, y + rect[1] * h, rect[2] * w, rect[3] * h])


def video_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"could not read frame {index} of {path.name}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def shade(ax, t, on, color, alpha=0.25):
    on = np.asarray(on, bool)
    edges = np.diff(np.r_[0, on.astype(int), 0])
    for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        ax.axvspan(t[a], t[min(b, len(t) - 1)], color=color, alpha=alpha, lw=0)


def state_rows(ax, t, rows):
    """Horizontal on/off bars, one row per (label, values, colour)."""
    for k, (label, on, color) in enumerate(rows):
        y = len(rows) - 1 - k
        ax.add_patch(Rectangle((t[0], y + 0.15), t[-1] - t[0], 0.7, color="#ececea", lw=0))
        on = np.asarray(on, bool)
        edges = np.diff(np.r_[0, on.astype(int), 0])
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
            ax.add_patch(Rectangle((t[a], y + 0.15), t[min(b, len(t) - 1)] - t[a] + 1 / FPS, 0.7,
                                   color=color, lw=0))
    ax.set_yticks(np.arange(len(rows)) + 0.5, [r[0] for r in rows][::-1])
    ax.set_ylim(0, len(rows))
    ax.set_xlim(t[0], t[-1])
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)


def main() -> None:
    print("Computing the example session ...")
    frames, info = analyze_session(BODY, CALIB, CLIP)
    dlc = read_dlc(BODY)
    calib = read_dlc(CALIB, nrows=settings.CALIBRATION_NROWS)
    mag = MagazineFrame(CALIB, CLIP)
    zones_px = zone_outlines_px(mag)
    zones_cm = zone_outlines_cm(mag)
    trials = read_trials(TRIALS)
    table = trial_tools.trial_frames(frames, trials)
    curves = trial_tools.time_course(table)
    conditions = trial_tools.per_condition(table)

    onset = int(trials.loc[trials[settings.TRIAL_NUMBER_COLUMN] == EXAMPLE_TRIAL, "onset_frame"].iloc[0])
    still_frame = onset + int(round(3.17 * FPS))            # a rotation frame of trial 8
    img = video_frame(CLIP, still_frame - CLIP_FIRST)
    w30 = slice(onset - int(10 * FPS), onset + int(20 * FPS))   # -10 to +20 s around the cue
    seg = frames.iloc[w30]
    t30 = (seg["frame"].to_numpy() - onset) / FPS

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.text(LEFT, 0.972, "DLC magazine analysis: what each step does", fontsize=38, fontweight="bold",
             color=TEXT, va="center")
    fig.text(LEFT, 0.936,
             "Input: DeepLabCut tracking of the rat (9 body points) and of the box (calibration points), "
             "plus the trial table.   Output, per video frame: at the magazine · head in the magazine · "
             "whole-body speed · head-anchored rotation;\nthen per trial and per cue type.   "
             "Example used in every box: Cohort01 R205, fixed ITI, Day 9 (99 trials, 78 min of video).",
             fontsize=17, color=TEXT_DIM, va="center")

    # 1 ------------------------------------------------------------------ inputs
    area = box(fig, 1, "Inputs: video + two DLC files",
               "Body DLC ('BehSyl' model): snout, neck, back1-4, tailbase, paws, per frame.\n"
               "Calibration DLC ('NewMar5'): static box points, incl. mag1 / mag2 =\n"
               "the two ends of the magazine opening (median over confident frames).\n"
               "Trial table: video frame of each cue onset (TrialLightOn_FrameIndex).")
    ax = sub(fig, area, (0.0, 0.0, 0.5, 1.0))
    ax.imshow(img)
    f = still_frame
    for bp in settings.BODYPARTS:
        if dlc[(bp, "likelihood")].iat[f] >= 0.6:
            ax.plot(dlc[(bp, "x")].iat[f], dlc[(bp, "y")].iat[f], "o", ms=7, mec="white", mew=0.8,
                    color={"snout": "#ff5a5a", "neck": "#ffc850", "tailbase": "#50c8ff"}.get(bp, "#dddddd"))
    for bp in ("cornerA", "cornerB", "wall1", "wall2", "wall3", "wall4", "mag1", "mag2"):
        p = calibration_point(calib, bp)
        is_mag = bp.startswith("mag")
        ax.plot(*p, "s", ms=9 if is_mag else 7, color=HEAD_IN if is_mag else "#9fd0ff", mec="black", mew=0.8)
        if is_mag:
            ax.annotate(bp, p, xytext=(-50, 8 if bp == "mag1" else -18), textcoords="offset points",
                        color="white", fontsize=12, fontweight="bold")
    ax.set_axis_off()
    ax.set_title("video frame with both DLC models", fontsize=13, color=TEXT_DIM)
    ax = sub(fig, area, (0.56, 0.08, 0.44, 0.84))
    ax.axis("off")
    rows = [["file", "what"],
            ["*BehSyl*.csv", "rat, 9 points"],
            ["*NewMar5*.csv", "box points"],
            ["trials.csv", "cue-onset frames"],
            ["video.mp4", "optional (height,\noverlay videos)"]]
    tab = ax.table(cellText=rows, loc="center", cellLoc="left", colWidths=[0.48, 0.52])
    tab.auto_set_font_size(False)
    tab.set_fontsize(12.5)
    tab.scale(1, 2.6)
    for (r, c), cell in tab.get_celld().items():
        cell.set_edgecolor("#dddddd")
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#eef2f6")

    # 2 ------------------------------------------------------------------ coordinates
    area = box(fig, 2, "Pixels -> floor cm -> magazine frame",
               "Camera views the box at an angle: 1 px = 0.18 cm wide but 0.11 cm tall,\n"
               "so pixels go through a 2x2 matrix (PX_TO_CM), not one factor.\n"
               "Then everything is put in a frame centred on the magazine opening:\n"
               "x = cm into the box, y = cm along the opening.")
    ax = sub(fig, area, (0.0, 0.05, 0.42, 0.9))
    mat = settings.PX_TO_CM
    ax.axis("off")
    ax.text(0.0, 0.95, "cm = PX_TO_CM @ px", fontsize=16, fontweight="bold", transform=ax.transAxes)
    ax.text(0.0, 0.80, f"PX_TO_CM = [[{mat[0, 0]:.4f}, {mat[0, 1]:.4f}],\n"
                       f"            [{mat[1, 0]:.4f}, {mat[1, 1]:.4f}]]",
            family="monospace", fontsize=13, transform=ax.transAxes, va="top")
    ax.text(0.0, 0.50, f"this session\nopening length: {info['mag_length_cm']:.2f} cm\n"
                       f"box width check: {info['box_width_px']:.1f} px\n"
                       f"(reference {settings.BOX_WIDTH_REFERENCE_PX:.1f} px ± 3%: "
                       f"{'ok' if info['box_width_ok'] else 'CHECK'})",
            fontsize=13, transform=ax.transAxes, va="top", color=TEXT_DIM, linespacing=1.5)
    ax = sub(fig, area, (0.50, 0.08, 0.48, 0.88))
    ten = frames.iloc[onset - int(120 * FPS): onset + int(120 * FPS)]
    local = mag.transform(ten["centroid_x_px"].to_numpy(), ten["centroid_y_px"].to_numpy())
    ax.plot(local[:, 0], local[:, 1], color="#bbbbbb", lw=0.6)
    at = ten["at_magazine"].to_numpy()
    ax.plot(local[at, 0], local[at, 1], ".", ms=2, color=AT_MAGAZINE)
    for name, color, ls in (("at_7cm", AT_MAGAZINE, "-"), ("left_12cm", "#888888", "--"),
                            ("semicircle", HEAD_IN, ":"), ("snout_circle", "#2e9e4f", "-")):
        z = zones_cm[name]
        ax.plot(z[:, 0], z[:, 1], color=color, ls=ls, lw=1.6)
    ax.plot(*zones_cm["opening"].T, color=HEAD_IN, lw=5, solid_capstyle="butt")
    ax.set_aspect("equal")
    ax.set_xlabel("into the box (cm)")
    ax.set_ylabel("along the opening (cm)")
    ax.set_title("body centroid, 4 min (green = at magazine)", fontsize=13, color=TEXT_DIM)
    style(ax)

    # 3 ------------------------------------------------------------------ zones
    area = box(fig, 3, "The zones around the magazine",
               "In cm the zones are circles; on the angled image they become ellipses.\n"
               "Green circle (0.75 x opening) + orange semicircle (9.75 cm): head in magazine.\n"
               "Solid green line (7 cm) + dashed grey line (12 cm): body at the magazine.\n"
               "Distances are to the opening as a segment, not to its midpoint.")
    ax = sub(fig, area, (0.0, 0.0, 0.62, 1.0))
    ax.imshow(img)
    for name, color, ls, lw in (("left_12cm", "#cccccc", "--", 2), ("at_7cm", AT_MAGAZINE, "-", 2.5),
                                ("semicircle", HEAD_IN, ":", 2.5), ("snout_circle", "#7CFC9A", "-", 1.6)):
        z = zones_px[name]
        ax.plot(z[:, 0], z[:, 1], color=color, ls=ls, lw=lw)
    ax.plot(*zones_px["opening"].T, color=HEAD_IN, lw=5)
    xs = np.r_[zones_px["left_12cm"][:, 0], zones_px["opening"][:, 0]]
    ys = np.r_[zones_px["left_12cm"][:, 1], zones_px["opening"][:, 1]]
    ax.set_xlim(xs.min() - 60, xs.max() + 90)
    ax.set_ylim(ys.max() + 20, ys.min() - 20)
    ax.set_axis_off()
    ax = sub(fig, area, (0.64, 0.05, 0.36, 0.9))
    ax.axis("off")
    legend = [("opening (mag1-mag2)", HEAD_IN, "-", 5),
              (f"snout circle r = {mag.snout_radius_cm:.2f} cm", "#2e9e4f", "-", 2),
              (f"semicircle r = {settings.MAG_ENTRY_RADIUS_CM:.2f} cm", HEAD_IN, ":", 2.5),
              (f"arrive < {settings.AT_MAGAZINE_CM:.0f} cm", AT_MAGAZINE, "-", 2.5),
              (f"leave > {settings.LEFT_MAGAZINE_CM:.0f} cm", "#888888", "--", 2)]
    for k, (label, color, ls, lw) in enumerate(legend):
        y = 0.9 - k * 0.17
        ax.plot([0.0, 0.16], [y, y], color=color, ls=ls, lw=lw, transform=ax.transAxes)
        ax.text(0.2, y, label, va="center", fontsize=12.5, transform=ax.transAxes)

    # 4 ------------------------------------------------------------------ at magazine
    area = box(fig, 4, "At the magazine (body)",
               "Body centroid = mean of the 9 body points (low-confidence frames filled in).\n"
               "The rat ARRIVES when the centroid is < 7 cm from the opening and LEAVES\n"
               "only when it is > 12 cm away; in between it keeps its state (hysteresis),\n"
               "so wobbling around one radius makes no fake visits. -> 'time at magazine'")
    hold_onset = int(trials.loc[trials[settings.TRIAL_NUMBER_COLUMN] == HYSTERESIS_TRIAL, "onset_frame"].iloc[0])
    hseg = frames.iloc[hold_onset - int(10 * FPS): hold_onset + int(20 * FPS)]
    th = (hseg["frame"].to_numpy() - hold_onset) / FPS
    ax = sub(fig, area, (0.10, 0.30, 0.88, 0.68))
    d = hseg["distance_to_magazine_cm"].to_numpy()
    shade(ax, th, hseg["at_magazine"], AT_MAGAZINE, 0.28)
    ax.axhspan(settings.AT_MAGAZINE_CM, settings.LEFT_MAGAZINE_CM, color="#f1e7c9", alpha=0.7, lw=0)
    ax.plot(th, d, color=TEXT, lw=1.4)
    ax.axhline(settings.AT_MAGAZINE_CM, color=AT_MAGAZINE, lw=2)
    ax.axhline(settings.LEFT_MAGAZINE_CM, color="#888888", lw=2, ls="--")
    ax.text(15.2, settings.AT_MAGAZINE_CM - 0.4, "arrive < 7 cm", va="top", fontsize=12,
            color="#2f7d3a", fontweight="bold")
    ax.text(1.0, settings.LEFT_MAGAZINE_CM + 0.4, "leave > 12 cm", va="bottom", fontsize=12,
            color="#555555", fontweight="bold")
    ax.text(1.0, 0.5 * (settings.AT_MAGAZINE_CM + settings.LEFT_MAGAZINE_CM), "band: keep state",
            va="center", fontsize=11, color="#8a7a3c")
    ax.axvline(0, color=TEXT, ls="--", lw=1)
    ax.set_ylim(0, min(np.nanmax(d) + 2, 30))
    ax.set_xlim(th[0], th[-1])
    ax.set_ylabel("centroid distance\nto opening (cm)")
    ax.tick_params(labelbottom=False)
    style(ax)
    ax = sub(fig, area, (0.10, 0.08, 0.88, 0.13))
    state_rows(ax, th, [("at magazine", hseg["at_magazine"], AT_MAGAZINE)])
    ax.axvline(0, color=TEXT, ls="--", lw=1)
    ax.set_xlabel(f"time from cue onset, trial {HYSTERESIS_TRIAL} (1 pellet) (s)")

    # 5 ------------------------------------------------------------------ head in
    area = box(fig, 5, "Head in the magazine (entry rule)",
               "A frame is an entry when ANY of three rules holds (likelihood >= 0.8):\n"
               "  snout: snout inside the green circle\n"
               "  head: neck->snout segment touches it, and back1 in the semicircle\n"
               "  facing: back1 in the semicircle, back1->neck line points at the opening")
    ax = sub(fig, area, (0.17, 0.42, 0.81, 0.56))
    state_rows(ax, t30, [("rule: snout", seg["rule_snout"], "#7fbf8a"),
                         ("rule: head", seg["rule_head"], "#7fbf8a"),
                         ("rule: facing", seg["rule_facing"], "#7fbf8a"),
                         ("head in magazine", seg["head_in_magazine"], HEAD_IN),
                         ("at magazine (body)", seg["at_magazine"], AT_MAGAZINE)])
    ax.axvline(0, color=TEXT, ls="--", lw=1)
    ax.set_xlabel(f"time from cue onset, trial {EXAMPLE_TRIAL} (s)")
    ax = sub(fig, area, (0.17, 0.0, 0.81, 0.28))
    ax.axis("off")
    ax.text(0, 0.80, f"Whole session: head in magazine {info['head_in_magazine_pct_of_session']:.1f}% of frames,\n"
                     f"body at magazine {info['at_magazine_pct_of_session']:.1f}%. The body definition is broader\n"
                     "(the rat can stand there with its head out).\n"
                     "Rotation (boxes 7-8) uses the body definition.",
            fontsize=13, va="top", color=TEXT_DIM, transform=ax.transAxes, linespacing=1.45)

    # 6 ------------------------------------------------------------------ speed
    area = box(fig, 6, "Whole-body speed",
               "Per body point: frame-to-frame step (through PX_TO_CM) x 29.97 fps.\n"
               "Kept only if both frames have likelihood >= 0.8 and speed <= 150 cm/s\n"
               "(bigger jumps are DLC label swaps). Whole-body speed = mean of the usable\n"
               "points, if at least 3 are usable. Per trial: mean over -5-0 s and 0-5 s.")
    ax = sub(fig, area, (0.09, 0.62, 0.89, 0.38))
    shade(ax, t30, seg["at_magazine"], AT_MAGAZINE, 0.2)
    ax.plot(t30, seg["speed_cm_s"], color=TEXT, lw=0.9)
    ax.axvline(0, color=TEXT, ls="--", lw=1)
    ax.set_xlim(t30[0], t30[-1])
    ax.set_ylabel("speed (cm/s)")
    ax.set_xlabel(f"time from cue onset, trial {EXAMPLE_TRIAL} (s)   (green = at magazine)")
    style(ax)
    ax = sub(fig, area, (0.09, 0.0, 0.40, 0.34))
    pt = trial_tools.per_trial(frames, trials)
    for k, cue in enumerate(CUE_ORDER):
        p = pt[pt[settings.TRIAL_TYPE_COLUMN] == cue]
        for j, col in enumerate(("speed_pre_cm_s", "speed_post_cm_s")):
            x = j + (k - 0.5) * 0.36
            ax.bar(x, p[col].mean(), 0.33, color=CUE_COLORS[cue], alpha=0.55 if j == 0 else 1)
            ax.errorbar(x, p[col].mean(), p[col].sem(), color=TEXT, capsize=3, lw=1)
    ax.set_xticks([0, 1], ["-5 to 0 s", "0 to 5 s"])
    ax.set_ylabel("cm/s")
    ax.set_title("per trial, mean ± SEM", fontsize=12, color=TEXT_DIM)
    style(ax)
    ax = sub(fig, area, (0.60, 0.0, 0.38, 0.34))
    course(ax, curves, "speed_cm_s", "cm/s")
    ax.set_xlabel("time from cue (s)")
    ax.set_title("time course", fontsize=12, color=TEXT_DIM)
    ax.legend(frameon=False, fontsize=11, loc="upper left")

    # 7 ------------------------------------------------------------------ rotation
    area = box(fig, 7, "Head-anchored rotation (per frame)",
               "Head stays at the magazine, body swings around it. A frame counts when ALL hold:\n"
               f"  1 at the magazine (box 4)    2 neck->tailbase axis turns > {settings.ROTATION_CUT_DEG_S:.1f} deg/s\n"
               f"  3 pivot s* (axis point that moves least) in the neck quarter: |s*| <= {settings.PIVOT_HEAD_MAX}\n"
               "Cut calibrated on hand-annotated turning syllables (AUC 0.86). Anchor = neck.")
    ax = sub(fig, area, (0.0, 0.42, 0.40, 0.58))
    ax.set_xlim(-0.4, 3.0)
    ax.set_ylim(-1.25, 1.45)
    ax.axis("off")
    for k, (s, label) in enumerate(((0.0, "s* = 0\nhead anchored\n(counted)"), (0.5, "s* = 0.5\nspins on middle"),
                                    (1.0, "s* = 1\ntail anchored"))):
        y = 0.85 - k * 0.85
        ax.add_patch(Rectangle((0.0, y - 0.07), 0.25, 0.14, color=AT_MAGAZINE, alpha=0.35, lw=0))
        ax.plot([0, 1], [y, y], color=ROTATION if s == 0 else TEXT, lw=3)
        ax.plot(0, y, "o", color="#ffc850", mec=TEXT, ms=8)
        ax.plot(1, y, "o", color="#50c8ff", mec=TEXT, ms=8)
        for pos in (0.0, 1.0):
            v = (pos - s) * 0.55
            if abs(v) > 1e-6:
                ax.add_patch(FancyArrowPatch((pos, y), (pos, y + v), arrowstyle="-|>", mutation_scale=12,
                                             color=TEXT_DIM, lw=1.4))
        ax.plot(s, y, "D", color=ROTATION if s == 0 else "#f0b400", ms=9, mec=TEXT)
        ax.text(1.35, y, label, va="center", fontsize=11.5, color=TEXT)
    ax.text(0.0, 1.32, "neck", fontsize=11, ha="center", color=TEXT_DIM)
    ax.text(1.0, 1.32, "tailbase", fontsize=11, ha="center", color=TEXT_DIM)
    t6 = slice(onset - int(1 * FPS), onset + int(7 * FPS))
    s6 = frames.iloc[t6]
    tt = (s6["frame"].to_numpy() - onset) / FPS
    ax = sub(fig, area, (0.56, 0.70, 0.42, 0.29))
    shade(ax, tt, s6["at_magazine"], AT_MAGAZINE, 0.18)
    ax.plot(tt, s6["rotation_deg_s"], color=TEXT, lw=1)
    ax.axhline(settings.ROTATION_CUT_DEG_S, color=ROTATION, lw=1.8, ls="--")
    ax.set_ylabel("axis turn\n(deg/s)")
    ax.set_xlim(tt[0], tt[-1])
    ax.set_ylim(0, 300)
    ax.tick_params(labelbottom=False)
    style(ax)
    ax = sub(fig, area, (0.56, 0.37, 0.42, 0.29))
    ax.axhspan(-settings.PIVOT_HEAD_MAX, settings.PIVOT_HEAD_MAX, color=AT_MAGAZINE, alpha=0.3, lw=0)
    rot_fast = s6["rotation_deg_s"].to_numpy() > settings.ROTATION_CUT_DEG_S
    ax.plot(tt, s6["pivot"].clip(-1, 2), color="#bbbbbb", lw=0.8)
    ax.plot(tt[rot_fast], s6["pivot"].clip(-1, 2).to_numpy()[rot_fast], ".", color=TEXT, ms=4)
    ax.set_ylabel("pivot s*")
    ax.set_ylim(-1, 2)
    ax.set_xlim(tt[0], tt[-1])
    ax.tick_params(labelbottom=False)
    style(ax)
    ax = sub(fig, area, (0.56, 0.06, 0.42, 0.25))
    state_rows(ax, tt, [("at mag.", s6["at_magazine"], AT_MAGAZINE),
                        ("fast", rot_fast, "#999999"),
                        ("rotation", s6["rotation_per_frame"], ROTATION)])
    ax.set_xlabel(f"time from cue, trial {EXAMPLE_TRIAL} (s)")
    ax = sub(fig, area, (0.0, 0.0, 0.42, 0.38))
    ax.axis("off")
    ax.text(0, 1.0, "Right, top: axis turn speed;\npink dashed = the cut.\n"
                    "Middle: pivot s*; green band =\nneck quarter; black dots = fast\nframes.\n"
                    "Bottom: rotation = at magazine\n+ fast + dot inside the band.",
            fontsize=12.5, color=TEXT_DIM, va="top", transform=ax.transAxes, linespacing=1.35)

    # 8 ------------------------------------------------------------------ bouts
    area = box(fig, 8, "Rotation as bouts",
               "Single-frame labels flicker, so rotation is counted as bouts, made on the\n"
               f"whole session before trials are cut: 1 gaps <= {settings.BOUT_MERGE_GAP_FRAMES} frames (167 ms) are\n"
               f"closed; 2 bouts < {settings.BOUT_MIN_FRAMES} frames (167 ms) are dropped; 3 bout frames must\n"
               "be at the magazine. Every rotation number is based on bouts.")
    z = slice(onset + int(1.5 * FPS), onset + int(5.0 * FPS))
    sz = frames.iloc[z]
    tz = (sz["frame"].to_numpy() - onset) / FPS
    ax = sub(fig, area, (0.18, 0.50, 0.80, 0.46))
    state_rows(ax, tz, [("per frame", sz["rotation_per_frame"], "#e7a3c9"),
                        ("bouts", sz["rotation_bout"], ROTATION),
                        ("at magazine", sz["at_magazine"], AT_MAGAZINE)])
    for xv in tz:
        ax.axvline(xv, color="white", lw=0.25, alpha=0.5)
    ax.set_xlabel(f"time from cue onset, trial {EXAMPLE_TRIAL} (s)  (thin lines = single frames)")
    ax = sub(fig, area, (0.18, 0.0, 0.80, 0.32))
    ax.axis("off")
    pf = frames["rotation_per_frame"].to_numpy()
    bo = frames["rotation_bout"].to_numpy()
    atf = frames["at_magazine"].to_numpy()
    n_runs = int(np.sum(np.diff(np.r_[0, pf.astype(int)]) == 1))
    n_bouts = int(np.sum(np.diff(np.r_[0, bo.astype(int)]) == 1))
    ax.text(0, 0.9, f"Whole session: {n_runs:,} per-frame runs -> {n_bouts:,} bouts.\n"
                    f"Rotation share of magazine time: {100 * pf[atf].mean():.1f}% per frame, "
                    f"{100 * bo[atf].mean():.1f}% as bouts.\n"
                    f"High / low rotation rats: fITI bout share (0-5 s, mean of 1p and 4p)\n"
                    f"above / below {settings.ROTATION_GROUP_SPLIT_PCT:.0f}%. This rat: high.",
            fontsize=13, va="top", color=TEXT_DIM, transform=ax.transAxes, linespacing=1.45)

    # 9 ------------------------------------------------------------------ rasters
    area = box(fig, 9, "Cut trials around cue onset",
               "Every frame label is cut from -2 to +8 s around each cue onset (the trial\n"
               "table's video frame). Trials whose window leaves the video are dropped.\n"
               "One row per trial: grey = away, green = at magazine, pink = rotation bout.\n"
               "Dashed line: cue onset.  Dotted line: pellet (5 s).")
    ax = sub(fig, area, (0.08, 0.10, 0.40, 0.88))
    raster(ax, table, "Cue1p")
    ax.set_xlabel("time from cue (s)")
    ax = sub(fig, area, (0.59, 0.10, 0.40, 0.88))
    raster(ax, table, "Cue4p")
    ax.set_xlabel("time from cue (s)")

    # 10 ----------------------------------------------------------------- time courses
    area = box(fig, 10, "Time courses (1 pellet vs 4 pellets)",
               "Per frame offset, across trials: P(at magazine), P(head in magazine),\n"
               "P(rotation | at magazine) and mean speed. Numerator and denominator are\n"
               "Gaussian-smoothed separately (sigma 0.26 s), so frames with few trials at\n"
               "the magazine do not dominate.  Red = 1 pellet, blue = 4 pellets.")
    for k, (column, label) in enumerate((("p_at_magazine", "P(at magazine)"),
                                         ("p_head_in_magazine", "P(head in)"),
                                         ("p_rotation_given_at", "P(rotation | at)"),
                                         ("speed_cm_s", "speed (cm/s)"))):
        r, c = divmod(k, 2)
        ax = sub(fig, area, (0.10 + c * 0.50, 0.58 - r * 0.52, 0.38, 0.40))
        course(ax, curves, column, label)
        if r == 1:
            ax.set_xlabel("time from cue (s)")

    # 11 ----------------------------------------------------------------- numbers
    area = box(fig, 11, "Numbers per trial and per cue type",
               "conditions.csv: 0-5 s after cue onset, frames pooled over trials (one row\n"
               "per cue type; these are the per-rat values used in the group analyses).\n"
               "trials.csv: the same measures for every single trial, plus speed before\n"
               "and after the cue. frames.csv.gz: every label for every frame.")
    ax = sub(fig, area, (0.0, 0.30, 1.0, 0.66))
    ax.axis("off")
    show = [("n_trials", "trials", "{:.0f}"), ("magazine_pct", "at magazine (% of 0-5 s)", "{:.1f}"),
            ("head_in_magazine_pct", "head in magazine (% of 0-5 s)", "{:.1f}"),
            ("rotation_pct_of_magazine", "rotation (% of magazine time)", "{:.1f}"),
            ("rotation_pct_of_window", "rotation (% of 0-5 s)", "{:.1f}"),
            ("rotation_deg_per_magazine_s", "rotation (deg per s at magazine)", "{:.1f}"),
            ("speed_cm_s", "speed (cm/s)", "{:.1f}")]
    cell = [[label] + [fmt.format(float(conditions.loc[conditions["trial_type"] == cue, col].iloc[0]))
                       for cue in CUE_ORDER] for col, label, fmt in show]
    tab = ax.table(cellText=cell, colLabels=["measure (0-5 s after cue)", "1 pellet", "4 pellets"],
                   loc="center", cellLoc="center", colWidths=[0.58, 0.2, 0.2])
    tab.auto_set_font_size(False)
    tab.set_fontsize(13)
    tab.scale(1, 2.1)
    for (r, c), cl in tab.get_celld().items():
        cl.set_edgecolor("#dddddd")
        if c == 0:
            cl.set_text_props(ha="left")
            cl._loc = "left"
        if r == 0:
            cl.set_text_props(fontweight="bold")
            cl.set_facecolor("#eef2f6")
        elif c == 1:
            cl.set_text_props(color=CUE_COLORS["Cue1p"], fontweight="bold")
        elif c == 2:
            cl.set_text_props(color=CUE_COLORS["Cue4p"], fontweight="bold")
    ax = sub(fig, area, (0.0, 0.0, 1.0, 0.24))
    ax.axis("off")
    ax.text(0, 0.95, "Check: with the full-precision DLC file these per-cue values are identical to the\n"
                     "published table of the original analysis (e.g. 4 pellets: 78.5% at magazine,\n"
                     "35.9% rotation). The example file is rounded to 0.01 px, which moves 1p rotation\n"
                     "from 30.0 to 30.2%.",
            fontsize=12.5, va="top", color=TEXT_DIM, transform=ax.transAxes, linespacing=1.4)

    # 12 ----------------------------------------------------------------- videos
    area = box(fig, 12, "Overlay videos: check it by eye",
               "make_trial_video.py draws the zones, the DLC skeleton, the centroid (green =\n"
               "at magazine) and the body axis (pink = rotation bout; diamond = pivot s*) on\n"
               "each trial, with state badges and a timeline. Below: trial 8 (4 pellets),\n"
               "before the cue, approaching, and during a head-anchored rotation.")
    times = (-2.0, 1.0, 3.17)
    for k, tsec in enumerate(times):
        idx = int(round((5.0 + tsec) * FPS))
        frame_img = video_frame(OVERLAY, idx) if OVERLAY.exists() else np.zeros((10, 10, 3), np.uint8)
        ax = sub(fig, area, (k / 3 + 0.005, 0.0, 1 / 3 - 0.01, 1.0))
        ax.imshow(frame_img)
        ax.set_axis_off()
        ax.set_title(f"t = {tsec:+.1f} s", fontsize=13, color=TEXT_DIM)

    # footer -----------------------------------------------------------------
    fig.text(LEFT, 0.045,
             "Run it:   python analyze_session.py --body-dlc <rat DLC> --calibration-dlc <box DLC> "
             "--trials <trials.csv> --out <folder>      python make_trial_video.py ... --video <mp4> "
             "--trial 8 --out <folder>",
             family="monospace", fontsize=14.5, color=TEXT, va="center")
    fig.text(LEFT, 0.020,
             "Same code and numbers as the Dopamine-and-Timing analyses: on this session every per-frame "
             "label (distance, at magazine, head-entry rules, speed, axis turn, pivot, rotation frames "
             "and bouts) matches the original modules on all 140,055 frames.   All numbers live in "
             "dlc_magazine/settings.py.",
             fontsize=14.5, color=TEXT_DIM, va="center")

    out = HERE / "analysis_overview.pdf"
    fig.savefig(out)
    fig.savefig(HERE / "analysis_overview_preview.png", dpi=50)
    if len(sys.argv) > 1:                       # optional: a high-res png for checking the layout
        fig.savefig(sys.argv[1], dpi=110)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
