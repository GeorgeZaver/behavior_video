"""Where the magazine is, and when the rat is at it.

Two different questions, two different definitions:

  at_magazine      Is the rat's BODY near the magazine? Body centroid closer
                   than 7 cm to the magazine opening, held until it moves
                   further than 12 cm (hysteresis). Used for "time at the
                   magazine" and as the frames where rotation is measured.

  magazine_entry   Is the rat's HEAD in the magazine? Snout inside a circle
                   around the opening, or the head/body pointing into it from
                   close by. The rule that was validated by eye on videos.

Both are measured in one coordinate frame centred on the magazine opening
(MagazineFrame): column 0 = cm into the box, column 1 = cm along the opening.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import settings
from .dlc_io import calibration_point, read_dlc, video_height_px


# =============================================================================
# MAGAZINE-CENTRED COORDINATES
# =============================================================================

class MagazineFrame:
    """Per-video map from pixels to magazine-centred cm.

    mag1 and mag2 (the two ends of the magazine opening) come from the
    calibration DLC. The origin is their midpoint, +y runs along the opening
    (mag1 -> mag2) and +x points into the box (towards wall2).
    """

    def __init__(self, calibration_csv: Path, video: Optional[Path] = None):
        calib = read_dlc(calibration_csv, nrows=settings.CALIBRATION_NROWS)
        self.y_scale = settings.REFERENCE_VIDEO_HEIGHT_PX / video_height_px(video)
        self.mag1_px = calibration_point(calib, "mag1")
        self.mag2_px = calibration_point(calib, "mag2")
        mag1 = self._to_cm(self.mag1_px)
        mag2 = self._to_cm(self.mag2_px)
        interior = self._to_cm(calibration_point(calib, settings.INTERIOR_REFERENCE_BODYPART))

        self.center = 0.5 * (mag1 + mag2)
        along = mag2 - mag1
        self.mag_length_cm = float(np.linalg.norm(along))
        if not np.isfinite(self.mag_length_cm) or self.mag_length_cm < 1e-6:
            raise ValueError(f"mag1 and mag2 coincide in {Path(calibration_csv).name}")
        self.u = along / self.mag_length_cm                 # along the opening
        normal = np.array([-self.u[1], self.u[0]])
        self.n = normal if np.dot(interior - self.center, normal) >= 0 else -normal  # into box
        self.snout_radius_cm = settings.SNOUT_CIRCLE_RADIUS_PER_MAG_LENGTH * self.mag_length_cm

    def _to_cm(self, xy_px: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy_px, dtype=float)
        scaled = np.stack([xy[..., 0], xy[..., 1] * self.y_scale], axis=-1)
        return scaled @ settings.PX_TO_CM.T

    def transform(self, x_px: np.ndarray, y_px: np.ndarray) -> np.ndarray:
        """(n, 2) magazine-frame cm: column 0 = into the box, column 1 = along the opening."""
        rel = self._to_cm(np.stack([x_px, y_px], axis=-1)) - self.center
        return np.stack([rel @ self.n, rel @ self.u], axis=-1)

    def to_px(self, local_cm: np.ndarray) -> np.ndarray:
        """Inverse of transform: magazine-frame cm -> image px (for drawing)."""
        local = np.atleast_2d(np.asarray(local_cm, dtype=float))
        cm = self.center + local[:, :1] * self.n + local[:, 1:2] * self.u
        scaled = cm @ np.linalg.inv(settings.PX_TO_CM).T
        return np.stack([scaled[:, 0], scaled[:, 1] / self.y_scale], axis=-1)

    def summary(self) -> Dict[str, float]:
        return {"mag_length_cm": self.mag_length_cm,
                "snout_circle_radius_cm": self.snout_radius_cm,
                "video_y_scale": self.y_scale}


def zone_outlines_cm(frame: MagazineFrame, n: int = 60) -> Dict[str, np.ndarray]:
    """Outlines of every zone in magazine-frame cm (into box, along opening), for drawing.

    opening       the magazine opening, mag1 -> mag2
    snout_circle  head-in-magazine: snout inside this circle
    semicircle    head-in-magazine: back1 inside this half disc
    at_7cm        at magazine: the centroid arrives inside this line
    left_12cm     at magazine: ... and leaves outside this one
    """
    half = 0.5 * frame.mag_length_cm

    def half_stadium(r: float) -> np.ndarray:
        lower = np.linspace(-np.pi / 2, 0, n)
        upper = np.linspace(0, np.pi / 2, n)
        a = np.stack([r * np.cos(lower), -half + r * np.sin(lower)], axis=1)
        b = np.stack([r * np.cos(upper), half + r * np.sin(upper)], axis=1)
        return np.vstack([a, b])

    circle = np.linspace(0, 2 * np.pi, 2 * n)
    semi = np.linspace(-np.pi / 2, np.pi / 2, n)
    rs, rm = frame.snout_radius_cm, settings.MAG_ENTRY_RADIUS_CM
    return {
        "opening": np.array([[0.0, -half], [0.0, half]]),
        "snout_circle": np.stack([rs * np.cos(circle), rs * np.sin(circle)], axis=1),
        "semicircle": np.stack([rm * np.cos(semi), rm * np.sin(semi)], axis=1),
        "at_7cm": half_stadium(settings.AT_MAGAZINE_CM),
        "left_12cm": half_stadium(settings.LEFT_MAGAZINE_CM),
    }


def zone_outlines_px(frame: MagazineFrame) -> Dict[str, np.ndarray]:
    """zone_outlines_cm mapped back into image pixels."""
    return {name: frame.to_px(points) for name, points in zone_outlines_cm(frame).items()}


# =============================================================================
# AT THE MAGAZINE (body centroid + hysteresis)
# =============================================================================

def centroid_px(dlc: pd.DataFrame) -> np.ndarray:
    """(n, 2) body centroid in px: mean of the 9 body points, each with its
    low-confidence frames (likelihood < 0.5) interpolated over."""
    points = []
    for bodypart in settings.BODYPARTS:
        xy = dlc[bodypart][["x", "y"]].to_numpy(float).copy()
        xy[~(dlc[bodypart]["likelihood"].to_numpy() >= settings.CENTROID_LIKELIHOOD_MIN)] = np.nan
        points.append(pd.DataFrame(xy).interpolate(limit_direction="both").to_numpy())
    return np.nanmean(np.stack(points), axis=0)


def distance_to_magazine_cm(frame: MagazineFrame, points_px: np.ndarray) -> np.ndarray:
    """Distance (cm) from each point to the magazine opening.

    Distance is to the opening as a line SEGMENT (mag1..mag2), not to its
    midpoint, so standing beside the opening does not read as far away.
    """
    local = frame.transform(points_px[:, 0], points_px[:, 1])
    into_box, along = local[:, 0], local[:, 1]
    overhang = np.maximum(np.abs(along) - 0.5 * frame.mag_length_cm, 0.0)
    return np.hypot(into_box, overhang)


def at_magazine(distance_cm: np.ndarray) -> np.ndarray:
    """True while the rat is at the magazine: arrives under 7 cm, leaves over 12 cm."""
    state = np.full(len(distance_cm), np.nan)
    state[distance_cm < settings.AT_MAGAZINE_CM] = 1.0
    state[distance_cm > settings.LEFT_MAGAZINE_CM] = 0.0
    return pd.Series(state).ffill().to_numpy() == 1.0


# =============================================================================
# HEAD IN THE MAGAZINE (magazine-entry rule)
# =============================================================================

def bodypart_cm(dlc: pd.DataFrame, bodypart: str, frame: MagazineFrame) -> np.ndarray:
    """A bodypart in magazine-frame cm; frames with likelihood < 0.8 are NaN."""
    xy = frame.transform(dlc[(bodypart, "x")].to_numpy(float), dlc[(bodypart, "y")].to_numpy(float))
    xy[~(dlc[(bodypart, "likelihood")].to_numpy(float) >= settings.ENTRY_LIKELIHOOD_MIN)] = np.nan
    return xy


def fill_short_gaps(flags: np.ndarray, max_gap_frames: int) -> np.ndarray:
    """Set False runs of <= max_gap_frames to True when they have True on both sides."""
    flags = np.asarray(flags, dtype=bool).copy()
    if max_gap_frames < 1 or not flags.any():
        return flags
    edges = np.diff(np.r_[True, flags, True].astype(np.int8))
    starts, ends = np.flatnonzero(edges == -1), np.flatnonzero(edges == 1)
    for start, end in zip(starts, ends):
        if start > 0 and end < len(flags) and end - start <= max_gap_frames:
            flags[start:end] = True
    return flags


def segment_hits_circle(p1: np.ndarray, p2: np.ndarray, radius: float) -> np.ndarray:
    """Does each segment p1 -> p2 come within `radius` of the origin?"""
    seg = p2 - p1
    length_sq = np.einsum("ij,ij->i", seg, seg)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.clip(-np.einsum("ij,ij->i", p1, seg) / length_sq, 0.0, 1.0)
    t = np.where(length_sq > 1e-12, t, 0.0)
    closest = p1 + t[:, None] * seg
    return np.hypot(closest[:, 0], closest[:, 1]) <= radius


def magazine_entry(dlc: pd.DataFrame, frame: MagazineFrame,
                   fps: float = settings.FPS) -> Dict[str, np.ndarray]:
    """Per-frame "head in the magazine" flags for a whole session.

    A frame is an entry when any of these holds:
      rule_snout   the snout is inside the green circle (radius 0.75 x opening
                   length, centred on the opening);
      rule_head    the neck -> snout segment touches that circle while back1 is
                   inside the orange semicircle (9.75 cm, opening into the box);
      rule_facing  back1 is inside the orange semicircle and the back1 -> neck
                   direction points at the opening.
    Gaps <= 0.5 s are closed in the snout rule, then again in the combined rule.
    """
    snout = bodypart_cm(dlc, "snout", frame)
    neck = bodypart_cm(dlc, "neck", frame)
    back = bodypart_cm(dlc, "back1", frame)

    with np.errstate(invalid="ignore"):
        snout_in_circle = np.hypot(snout[:, 0], snout[:, 1]) <= frame.snout_radius_cm
        back_in_semicircle = ((np.hypot(back[:, 0], back[:, 1]) <= settings.MAG_ENTRY_RADIUS_CM)
                              & (back[:, 0] >= 0))

        head_valid = np.isfinite(snout).all(axis=1) & np.isfinite(neck).all(axis=1)
        head_rule = head_valid & back_in_semicircle & segment_hits_circle(
            neck, snout, frame.snout_radius_cm)

        direction = neck - back
        orientation_valid = (np.isfinite(direction).all(axis=1)
                             & (np.hypot(direction[:, 0], direction[:, 1]) > 1e-12))
        dx, dy = direction[:, 0], direction[:, 1]
        t_hit = np.where(np.abs(dx) > 1e-12, -back[:, 0] / dx, np.nan)   # where the ray meets x = 0
        y_hit = back[:, 1] + t_hit * dy
        facing = (t_hit >= 0) & (np.abs(y_hit) <= 0.5 * frame.mag_length_cm)
        facing_rule = orientation_valid & back_in_semicircle & facing

    snout_buffered = fill_short_gaps(snout_in_circle,
                                     int(round(settings.SNOUT_MAG_ENTRY_BREAK_BUFFER_S * fps)))
    raw = snout_buffered | head_rule | facing_rule
    in_magazine = fill_short_gaps(raw, int(round(settings.MAG_ENTRY_BREAK_BUFFER_S * fps)))
    return {
        "in_magazine": in_magazine,
        "classifiable": orientation_valid | in_magazine,
        "rule_snout": snout_buffered,
        "rule_head": head_rule,
        "rule_facing": facing_rule,
    }
