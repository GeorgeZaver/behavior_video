"""From per-frame labels to trials: cut windows around cue onset and summarise.

  trial_frames(frames, trials)   every frame from -2 to 8 s around each cue onset
  per_trial(frames, trials)      one row per trial (0-5 s measures, speed before/after)
  per_condition(trial_table)     one row per cue type, frames pooled over trials
                                 (the numbers reported per rat in the analyses)
  time_course(trial_table)       smoothed probability / mean per frame, -2 to 8 s

Only trials whose whole -2 to 8 s window lies inside the video are used
("usable"), as in the original analyses.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import settings

FRAME_COLUMNS = ["distance_to_magazine_cm", "at_magazine", "head_in_magazine",
                 "speed_cm_s", "rotation_deg_s", "pivot", "rotation_per_frame", "rotation_bout"]


def _frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def course_offsets(fps: float = settings.FPS) -> np.ndarray:
    """Frame offsets of the -2 to 8 s window (-60 ... 239 at 29.97 fps)."""
    return np.arange(-_frames(settings.COURSE_PRE_S, fps), _frames(settings.COURSE_POST_S, fps))


def usable_trials(trials: pd.DataFrame, n_frames: int, fps: float = settings.FPS) -> pd.DataFrame:
    offsets = course_offsets(fps)
    onset = trials["onset_frame"]
    return trials[(onset + offsets[0] >= 0) & (onset + offsets[-1] < n_frames)]


def trial_frames(frames: pd.DataFrame, trials: pd.DataFrame, fps: float = settings.FPS) -> pd.DataFrame:
    """Long table: one row per (trial, frame offset) from -2 to 8 s around cue onset."""
    usable = usable_trials(trials, len(frames), fps)
    offsets = course_offsets(fps)
    index = (usable["onset_frame"].to_numpy()[:, None] + offsets[None, :]).ravel()
    n = len(offsets)
    out = pd.DataFrame({
        "trial": np.repeat(usable[settings.TRIAL_NUMBER_COLUMN].to_numpy(), n),
        "trial_type": np.repeat(usable[settings.TRIAL_TYPE_COLUMN].to_numpy(), n),
        "frame": np.tile(offsets, len(usable)),
        "time_s": np.tile(offsets, len(usable)) / fps,
    })
    for column in FRAME_COLUMNS:
        out[column] = frames[column].to_numpy()[index]
    return out


def _cut(values: np.ndarray, onsets: np.ndarray, lo_s: float, hi_s: float, fps: float) -> np.ndarray:
    """trials x frames of values in [lo_s, hi_s) around each onset; NaN outside the video."""
    offsets = np.arange(-_frames(10.0, fps), _frames(10.0, fps) + 1)
    keep = offsets[(offsets / fps >= lo_s) & (offsets / fps < hi_s)]
    index = onsets[:, None] + keep[None, :]
    inside = (index >= 0) & (index < len(values))
    matrix = np.full(index.shape, np.nan)
    matrix[inside] = values[index[inside]]
    return matrix


def per_trial(frames: pd.DataFrame, trials: pd.DataFrame, fps: float = settings.FPS) -> pd.DataFrame:
    """One row per trial. The magazine / rotation columns use 0-5 s after cue onset."""
    window = _frames(settings.MEASURE_WINDOW_S, fps)
    onsets = trials["onset_frame"].to_numpy()
    usable = trials.index.isin(usable_trials(trials, len(frames), fps).index)
    out = trials[[settings.TRIAL_NUMBER_COLUMN, settings.TRIAL_TYPE_COLUMN, "onset_frame"]].copy()
    out["usable"] = usable

    def window_of(column):
        m = np.full((len(trials), window), np.nan)
        for i, onset in enumerate(onsets):
            if usable[i]:
                m[i] = frames[column].to_numpy(float)[onset:onset + window]
        return m

    at, head = window_of("at_magazine"), window_of("head_in_magazine")
    bout, rot = window_of("rotation_bout"), window_of("rotation_deg_s")
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        n_at = np.nansum(at, axis=1)
        out["at_magazine_pct"] = 100 * np.nanmean(at, axis=1)
        out["head_in_magazine_pct"] = 100 * np.nanmean(head, axis=1)
        out["rotation_pct_of_magazine"] = np.where(n_at > 0, 100 * np.nansum(bout * at, axis=1) / n_at, np.nan)
        out["rotation_pct_of_window"] = 100 * np.nanmean(bout, axis=1)
        out["rotation_deg_per_magazine_s"] = np.where(
            n_at > 0, np.nansum(np.where(bout == 1, rot, 0), axis=1) / n_at, np.nan)
        speed = frames["speed_cm_s"].to_numpy(float)
        pre = _cut(speed, onsets, *settings.SPEED_PRE_WINDOW, fps)
        post = _cut(speed, onsets, *settings.SPEED_POST_WINDOW, fps)
        out["speed_pre_cm_s"] = np.nanmean(pre, axis=1)
        out["speed_post_cm_s"] = np.nanmean(post, axis=1)
    out["speed_post_minus_pre_cm_s"] = out["speed_post_cm_s"] - out["speed_pre_cm_s"]
    out.loc[~usable, ["at_magazine_pct", "head_in_magazine_pct", "rotation_pct_of_magazine",
                      "rotation_pct_of_window", "rotation_deg_per_magazine_s"]] = np.nan
    return out


def per_condition(table: pd.DataFrame, fps: float = settings.FPS) -> pd.DataFrame:
    """One row per cue type over the 0-5 s window, frames pooled across trials.

    magazine_pct                 % of the window at the magazine
    rotation_pct_of_magazine     % of at-magazine frames in a rotation bout (headline)
    rotation_pct_of_window       the same frames as % of the whole window
    rotation_deg_per_magazine_s  degrees turned in rotation bouts per second at the magazine
    head_in_magazine_pct         % of the window with the head in the magazine
    speed_cm_s                   mean whole-body speed
    reliable                     >= 5 s at the magazine in total
    """
    window = table[table["frame"].between(0, _frames(settings.MEASURE_WINDOW_S, fps) - 1)]
    rows = []
    for cue, block in window.groupby("trial_type"):
        at = block[block["at_magazine"]]
        bouts = at[at["rotation_bout"]]
        magazine_s = len(at) / fps
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            speed = float(np.nanmean(block["speed_cm_s"]))
        rows.append({
            "trial_type": cue, "n_trials": block["trial"].nunique(),
            "magazine_pct": 100.0 * len(at) / len(block),
            "magazine_s": magazine_s,
            "rotation_pct_of_magazine": 100.0 * len(bouts) / len(at) if len(at) else np.nan,
            "rotation_pct_of_window": 100.0 * len(bouts) / len(block),
            "rotation_deg_per_magazine_s": (bouts["rotation_deg_s"].sum() / fps / magazine_s
                                            if magazine_s else np.nan),
            "head_in_magazine_pct": 100.0 * block["head_in_magazine"].mean(),
            "speed_cm_s": speed,
            "reliable": magazine_s >= settings.MIN_MAGAZINE_S,
        })
    return pd.DataFrame(rows)


def _smoothed_ratio(numerator: np.ndarray, denominator: np.ndarray, sigma: float) -> np.ndarray:
    """Smooth numerator and denominator separately, so frames with few trials do not dominate."""
    num = gaussian_filter1d(numerator.astype(float), sigma, mode="nearest")
    den = gaussian_filter1d(denominator.astype(float), sigma, mode="nearest")
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den >= 1.0, num / den, np.nan)


def time_course(table: pd.DataFrame, fps: float = settings.FPS) -> pd.DataFrame:
    """Per cue type and frame offset (-2 to 8 s):

    p_at_magazine          fraction of trials at the magazine
    p_head_in_magazine     fraction of trials with the head in the magazine
    p_rotation_given_at    P(rotation bout | at magazine)
    speed_cm_s             mean whole-body speed
    All Gaussian-smoothed (sigma 0.26 s).
    """
    sigma = settings.TIMECOURSE_SIGMA_S * fps
    offsets = course_offsets(fps)
    rows = []
    for cue, block in table.groupby("trial_type"):
        valid_speed = block["speed_cm_s"].notna()
        g = block.assign(speed0=block["speed_cm_s"].fillna(0.0), valid=valid_speed.astype(int),
                         one=1).groupby("frame")
        by = pd.DataFrame({
            "n": g["one"].sum(), "at": g["at_magazine"].sum(),
            "head": g["head_in_magazine"].sum(), "bout": g["rotation_bout"].sum(),
            "speed": g["speed0"].sum(), "valid": g["valid"].sum(),
        }).reindex(offsets, fill_value=0)
        rows.append(pd.DataFrame({
            "trial_type": cue, "frame": offsets, "time_s": offsets / fps,
            "p_at_magazine": _smoothed_ratio(by["at"].to_numpy(), by["n"].to_numpy(), sigma),
            "p_head_in_magazine": _smoothed_ratio(by["head"].to_numpy(), by["n"].to_numpy(), sigma),
            "p_rotation_given_at": _smoothed_ratio(by["bout"].to_numpy(), by["at"].to_numpy(), sigma),
            "speed_cm_s": _smoothed_ratio(by["speed"].to_numpy(), by["valid"].to_numpy(), sigma),
        }))
    return pd.concat(rows, ignore_index=True)
