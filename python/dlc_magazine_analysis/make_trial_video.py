"""Overlay video of single trials: what the analysis "sees" frame by frame.

For each requested trial, a clip from -5 to +10 s around cue onset with:
  on the video   the magazine opening (orange line), the head-in-magazine zones
                 (green snout circle, orange semicircle), the at-magazine radii
                 (7 cm solid, 12 cm dashed), the DLC skeleton, the body centroid
                 (green when at the magazine), and the neck -> tailbase axis,
                 pink during a head-anchored rotation bout, with its pivot s*
                 (diamond) whenever the axis turns faster than the cut.
  header         rat / trial / cue, time from cue onset, state badges
  timeline       at magazine, head in magazine, rotation bout, and speed over
                 the clip, with a cursor; cue onset (0 s) and pellet (5 s) marked.

Usage
  python make_trial_video.py --body-dlc <body.csv> --calibration-dlc <calib.csv> ^
      --trials <trials.csv> --video <session.mp4> --trial 8 --out <folder>

  --video-first-frame N   the video is a clip that starts at session frame N
                          (the example clips are; the number is in their name)
  Without --video the overlay is drawn on a dark background.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd

from dlc_magazine import analyze_session, settings
from dlc_magazine.dlc_io import read_dlc, read_trials, video_height_px
from dlc_magazine.magazine import MagazineFrame, zone_outlines_px

PRE_S, POST_S = 5.0, 10.0
SCALE = 2                       # upscale the (352 x 480) video for legibility
HEADER_H, TIMELINE_H = 86, 250
REWARD_S = 5.0

# BGR colours
BG = (24, 20, 18)
TEXT = (236, 236, 236)
TEXT_DIM = (160, 160, 160)
GREEN = (106, 184, 95)          # at magazine
ORANGE = (26, 129, 232)         # head in magazine / magazine opening
PINK = (146, 58, 214)           # rotation
GREY = (150, 150, 150)
WHITE = (235, 235, 235)
CUE_BGR = {"Cue1p": (34, 34, 178), "Cue4p": (180, 120, 31)}
POINT_COLORS = {"snout": (90, 90, 255), "neck": (80, 200, 255), "tailbase": (255, 200, 80)}
SKELETON = [("snout", "neck"), ("neck", "back1"), ("back1", "back2"), ("back2", "back3"),
            ("back3", "back4"), ("back4", "tailbase")]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def px(points: np.ndarray) -> np.ndarray:
    return np.round(np.asarray(points) * SCALE).astype(np.int32)


def dashed(img, points: np.ndarray, color, dash: int = 8, thickness: int = 1) -> None:
    pts = px(points)
    for i in range(0, len(pts) - 1, 2):
        cv2.line(img, tuple(pts[i]), tuple(pts[i + 1]), color, thickness, cv2.LINE_AA)


def draw_zones(img, zones: Dict[str, np.ndarray]) -> None:
    dashed(img, zones["left_12cm"], GREY, thickness=2)
    cv2.polylines(img, [px(zones["at_7cm"])], False, GREEN, 2, cv2.LINE_AA)
    dashed(img, zones["semicircle"], ORANGE)
    cv2.polylines(img, [px(zones["snout_circle"])], True, (120, 220, 120), 1, cv2.LINE_AA)
    cv2.line(img, *map(tuple, px(zones["opening"])), ORANGE, 3, cv2.LINE_AA)


def draw_rat(img, dlc: pd.DataFrame, f: int, row: pd.Series) -> None:
    def point(bp):
        if dlc[(bp, "likelihood")].iat[f] < 0.6:
            return None
        return np.array([dlc[(bp, "x")].iat[f], dlc[(bp, "y")].iat[f]])

    for a, b in SKELETON:
        pa, pb = point(a), point(b)
        if pa is not None and pb is not None:
            cv2.line(img, tuple(px(pa)), tuple(px(pb)), (205, 205, 205), 1, cv2.LINE_AA)
    for bp in settings.BODYPARTS:
        p = point(bp)
        if p is not None:
            cv2.circle(img, tuple(px(p)), 4, POINT_COLORS.get(bp, (200, 200, 200)), -1, cv2.LINE_AA)
    neck, tail = point("neck"), point("tailbase")
    if neck is not None and tail is not None:
        bout = bool(row["rotation_bout"])
        cv2.line(img, tuple(px(neck)), tuple(px(tail)), PINK if bout else WHITE, 4 if bout else 2, cv2.LINE_AA)
        if np.isfinite(row["rotation_deg_s"]) and row["rotation_deg_s"] > settings.ROTATION_CUT_DEG_S \
                and np.isfinite(row["pivot"]):
            s = float(np.clip(row["pivot"], -0.5, 1.5))
            p = px(neck + s * (tail - neck))
            anchored = abs(row["pivot"]) <= settings.PIVOT_HEAD_MAX
            d = 7
            diamond = np.array([[p[0], p[1] - d], [p[0] + d, p[1]], [p[0], p[1] + d], [p[0] - d, p[1]]])
            cv2.fillPoly(img, [diamond], PINK if anchored else (0, 210, 255), cv2.LINE_AA)
    c = np.array([row["centroid_x_px"], row["centroid_y_px"]])
    if np.isfinite(c).all():
        cv2.circle(img, tuple(px(c)), 6, GREEN if row["at_magazine"] else GREY, 2, cv2.LINE_AA)


def badge(img, text: str, x_right: int, y: int, color, on: bool) -> int:
    (w, h), _ = cv2.getTextSize(text, FONT, 0.5, 1)
    x0 = x_right - w - 16
    if on:
        cv2.rectangle(img, (x0, y), (x_right, y + h + 12), color, -1)
        cv2.putText(img, text, (x0 + 8, y + h + 5), FONT, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    else:
        cv2.rectangle(img, (x0, y), (x_right, y + h + 12), (80, 80, 80), 1)
        cv2.putText(img, text, (x0 + 8, y + h + 5), FONT, 0.5, (110, 110, 110), 1, cv2.LINE_AA)
    return x0 - 8


def timeline_base(width: int, clip: pd.DataFrame, fps: float) -> np.ndarray:
    img = np.full((TIMELINE_H, width, 3), BG, np.uint8)
    n = len(clip)
    left, right = 150, width - 16
    xs = left + (np.arange(n) + 0.5) / n * (right - left)
    tracks = [("at magazine", "at_magazine", GREEN), ("head in magazine", "head_in_magazine", ORANGE),
              ("rotation bout", "rotation_bout", PINK)]
    for k, (label, column, color) in enumerate(tracks):
        y0 = 14 + k * 30
        cv2.putText(img, label, (10, y0 + 15), FONT, 0.45, TEXT_DIM, 1, cv2.LINE_AA)
        cv2.rectangle(img, (left, y0), (right, y0 + 20), (48, 44, 42), -1)
        on = clip[column].to_numpy(bool)
        for i in np.flatnonzero(on):
            x0 = int(left + i / n * (right - left))
            x1 = int(left + (i + 1) / n * (right - left)) + 1
            cv2.rectangle(img, (x0, y0), (x1, y0 + 20), color, -1)
    y_top, y_bot = 120, TIMELINE_H - 34
    cv2.putText(img, "speed", (10, y_top + 20), FONT, 0.45, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.putText(img, "(0-40 cm/s)", (10, y_top + 40), FONT, 0.4, TEXT_DIM, 1, cv2.LINE_AA)
    cv2.rectangle(img, (left, y_top), (right, y_bot), (48, 44, 42), 1)
    speed = np.clip(clip["speed_cm_s"].to_numpy(float), 0, 40)
    ys = y_bot - speed / 40 * (y_bot - y_top)
    pts = np.stack([xs, ys], axis=1)
    for i in range(n - 1):
        if np.isfinite(pts[i]).all() and np.isfinite(pts[i + 1]).all():
            cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), WHITE, 1, cv2.LINE_AA)
    t = clip["t_rel"].to_numpy()
    for value, label, style in ((0.0, "cue", TEXT), (REWARD_S, "pellet", TEXT_DIM)):
        x = int(left + np.searchsorted(t, value) / n * (right - left))
        cv2.line(img, (x, 10), (x, y_bot), style, 1, cv2.LINE_AA)
        cv2.putText(img, label, (x - 14, y_bot + 18), FONT, 0.45, style, 1, cv2.LINE_AA)
    for sec in range(int(-PRE_S), int(POST_S) + 1, 5):
        x = int(left + np.searchsorted(t, sec) / n * (right - left))
        cv2.putText(img, f"{sec:+d} s" if sec else "0 s", (min(x - 14, width - 46), TIMELINE_H - 4), FONT, 0.4,
                    TEXT_DIM, 1, cv2.LINE_AA)
    img[:, :1] = BG
    return img, left, right


def render_trial(trial: pd.Series, frames: pd.DataFrame, dlc: pd.DataFrame, zones, title: str,
                 video: Optional[Path], video_first_frame: int, out: Path, fps: float,
                 size: tuple) -> Path:
    onset = int(trial["onset_frame"])
    start = max(onset - int(round(PRE_S * fps)), 0)
    stop = min(onset + int(round(POST_S * fps)), len(frames))
    clip = frames.iloc[start:stop].copy()
    clip["t_rel"] = (clip["frame"] - onset) / fps
    w, h = size[0] * SCALE, size[1] * SCALE
    timeline, left, right = timeline_base(w, clip, fps)
    cue = str(trial[settings.TRIAL_TYPE_COLUMN])
    name = f"trial{int(trial[settings.TRIAL_NUMBER_COLUMN]):03d}_{cue}.mp4"
    path = out / name

    cap = None
    if video is not None:
        cap = cv2.VideoCapture(str(video))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start - video_first_frame)
    writer = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{w}x{h + HEADER_H + TIMELINE_H}", "-r", "30000/1001", "-i", "-",
         "-c:v", "libx264", "-crf", "23", "-pix_fmt", "yuv420p", str(path)],
        stdin=subprocess.PIPE)
    for i, (f, row) in enumerate(clip.iterrows()):
        if cap is not None:
            ok, img = cap.read()
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR) if ok else np.full((h, w, 3), BG, np.uint8)
        else:
            img = np.full((h, w, 3), BG, np.uint8)
        draw_zones(img, zones)
        draw_rat(img, dlc, int(f), row)

        head = np.full((HEADER_H, w, 3), BG, np.uint8)
        cv2.putText(head, title, (12, 26), FONT, 0.6, TEXT, 1, cv2.LINE_AA)
        cv2.putText(head, f"trial {int(trial[settings.TRIAL_NUMBER_COLUMN])}", (12, 54), FONT, 0.6, TEXT, 1, cv2.LINE_AA)
        cv2.putText(head, cue, (130, 54), FONT, 0.6, CUE_BGR.get(cue, TEXT), 2, cv2.LINE_AA)
        cv2.putText(head, f"t = {row['t_rel']:+5.2f} s from cue", (12, 78), FONT, 0.5, TEXT_DIM, 1, cv2.LINE_AA)
        x = badge(head, "ROTATION", w - 10, 12, PINK, bool(row["rotation_bout"]))
        x = badge(head, "HEAD IN", x, 12, ORANGE, bool(row["head_in_magazine"]))
        badge(head, "AT MAGAZINE", x, 12, GREEN, bool(row["at_magazine"]))
        cv2.putText(head, f"axis {row['rotation_deg_s']:5.0f} deg/s   s* {row['pivot']:+5.2f}"
                    if np.isfinite(row["rotation_deg_s"]) and np.isfinite(row["pivot"]) else "",
                    (w - 290, 70), FONT, 0.45, TEXT_DIM, 1, cv2.LINE_AA)

        tl = timeline.copy()
        xc = int(left + (i + 0.5) / len(clip) * (right - left))
        cv2.line(tl, (xc, 8), (xc, TIMELINE_H - 30), (0, 220, 255), 2)
        writer.stdin.write(np.vstack([head, img, tl]).tobytes())
    writer.stdin.close()
    writer.wait()
    if cap is not None:
        cap.release()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--body-dlc", type=Path, required=True)
    parser.add_argument("--calibration-dlc", type=Path, required=True)
    parser.add_argument("--trials", type=Path, required=True)
    parser.add_argument("--video", type=Path, help="session video (or a clip, see --video-first-frame)")
    parser.add_argument("--video-first-frame", type=int, default=0)
    parser.add_argument("--trial", type=int, nargs="+", required=True, help="trial number(s)")
    parser.add_argument("--title", default="")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=settings.FPS)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Computing per-frame labels ...")
    frames, _ = analyze_session(args.body_dlc, args.calibration_dlc, args.video, args.fps)
    dlc = read_dlc(args.body_dlc)
    frame = MagazineFrame(args.calibration_dlc, args.video)
    zones = zone_outlines_px(frame)
    trials = read_trials(args.trials)
    size = (352, int(video_height_px(args.video)))
    if args.video is not None:
        cap = cv2.VideoCapture(str(args.video))
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        cap.release()
    for number in args.trial:
        hit = trials[trials[settings.TRIAL_NUMBER_COLUMN] == number]
        if hit.empty:
            print(f"  trial {number}: not in the trial table, skipped")
            continue
        path = render_trial(hit.iloc[0], frames, dlc, zones, args.title, args.video,
                            args.video_first_frame, args.out, args.fps, size)
        print(f"  {path}")


if __name__ == "__main__":
    main()
