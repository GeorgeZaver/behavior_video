"""Head-anchored body rotation: the rat keeps its head at the magazine and
swings its body around it.

A frame is a head-anchored rotation when all three hold:

  at the magazine    magazine.at_magazine (centroid < 7 cm, held to 12 cm).
  body rotating      the neck -> tailbase axis turns faster than
                     ROTATION_CUT_DEG_S (44.6 deg/s).
  pivot at the head  the point of the body axis that moves least lies in the
                     neck quarter of the axis. For a rigid axis the velocity
                     along it is linear between neck and tailbase, so the
                     still point is
                         s* = argmin_s |v_neck + s (v_tail - v_neck)|
                     s* = 0: neck still, tail swings (head-anchored);
                     s* = 1: tail still, head swings;
                     s* = 0.5: the rat spins on its middle.
                     Head-anchored = |s*| <= 0.25.

The neck, not the snout, is the anchor: the snout keeps sniffing and poking
while the head stays at the magazine.

The per-frame label flickers, so it is turned into bouts on the whole session
timeline (before any trial is cut): gaps of <= 5 frames are closed, bouts
shorter than 5 frames are dropped, and bout frames must be at the magazine.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import settings


def keypoint_cm(dlc: pd.DataFrame, bodypart: str) -> np.ndarray:
    """(n, 2) floor-plane cm of one bodypart; frames with likelihood < 0.6 are
    interpolated over when the gap is <= 5 frames, otherwise NaN."""
    xy = dlc[bodypart][["x", "y"]].to_numpy(float).copy()
    xy[dlc[bodypart]["likelihood"].to_numpy() < settings.ROTATION_LIKELIHOOD_MIN] = np.nan
    xy = (pd.DataFrame(xy).interpolate(limit=settings.ROTATION_GAP_FRAMES, limit_direction="both")
          .to_numpy())
    return (settings.PX_TO_CM @ xy.T).T


def _velocity(points_cm: np.ndarray, fps: float) -> np.ndarray:
    velocity = np.vstack([[np.nan, np.nan], np.diff(points_cm, axis=0)]) * fps
    return (pd.DataFrame(velocity).rolling(settings.SMOOTH_FRAMES, center=True, min_periods=3)
            .mean().to_numpy())


def body_rotation(dlc: pd.DataFrame, fps: float = settings.FPS) -> Tuple[np.ndarray, np.ndarray]:
    """Angular speed of the neck -> tailbase axis (deg/s) and the pivot location s*."""
    neck = keypoint_cm(dlc, "neck")
    tail = keypoint_cm(dlc, "tailbase")
    axis = neck - tail
    angle = np.arctan2(axis[:, 1], axis[:, 0])
    # Wrapped per-frame difference, so one missing frame does not poison the rest.
    step = np.abs(np.angle(np.exp(1j * np.diff(angle))))
    speed = np.r_[np.nan, step] * fps * 180.0 / np.pi
    speed = (pd.Series(speed).rolling(settings.SMOOTH_FRAMES, center=True, min_periods=3)
             .median().to_numpy())
    v_neck, v_tail = _velocity(neck, fps), _velocity(tail, fps)
    dv = v_tail - v_neck
    with np.errstate(invalid="ignore", divide="ignore"):
        pivot = -np.sum(v_neck * dv, axis=1) / np.sum(dv * dv, axis=1)
    return speed, pivot


def filter_bouts(rotating: np.ndarray, at: Optional[np.ndarray] = None,
                 merge_gap: int = settings.BOUT_MERGE_GAP_FRAMES,
                 min_frames: int = settings.BOUT_MIN_FRAMES) -> np.ndarray:
    """Per-frame label -> bout label: close gaps <= merge_gap, drop bouts < min_frames."""
    mask = np.asarray(rotating, dtype=bool)
    edges = np.diff(np.r_[0, mask.astype(np.int8), 0])
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    if not len(starts):
        return np.zeros_like(mask)
    separate = (starts[1:] - ends[:-1]) > merge_gap
    onsets = np.r_[starts[0], starts[1:][separate]]
    offsets = np.r_[ends[:-1][separate], ends[-1]]
    keep = (offsets - onsets) >= min_frames
    delta = np.zeros(len(mask) + 1, dtype=int)
    np.add.at(delta, onsets[keep], 1)
    np.add.at(delta, offsets[keep], -1)
    bouts = np.cumsum(delta[:-1]) > 0
    return bouts & at if at is not None else bouts


def rotation_labels(dlc: pd.DataFrame, at: np.ndarray,
                    cut: float = settings.ROTATION_CUT_DEG_S,
                    fps: float = settings.FPS) -> Dict[str, np.ndarray]:
    """Body rotation, pivot, and the head-anchored rotation label per frame and as bouts."""
    rotation, pivot = body_rotation(dlc, fps)
    with np.errstate(invalid="ignore"):
        rotating = rotation > cut
        anchored = np.abs(pivot) <= settings.PIVOT_HEAD_MAX
    per_frame = at & rotating & anchored
    return {
        "rotation_deg_s": rotation,
        "pivot": pivot,
        "rotation_per_frame": per_frame,
        "rotation_bout": filter_bouts(per_frame, at),
    }
