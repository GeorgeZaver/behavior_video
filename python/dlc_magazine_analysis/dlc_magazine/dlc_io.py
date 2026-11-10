"""Reading DeepLabCut output and the session trial table.

A DLC csv has three header rows (scorer / bodyparts / coords) and one row per
video frame. read_dlc returns a DataFrame with two-level columns, so
dlc[("snout", "x")] is the snout's x position in pixels for every frame.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd

from . import settings


def read_dlc(path: Path, nrows: Optional[int] = None) -> pd.DataFrame:
    """DLC csv (or csv.gz) -> DataFrame with (bodypart, x|y|likelihood) columns."""
    df = pd.read_csv(path, header=[1, 2], index_col=0, nrows=nrows)
    df.columns = pd.MultiIndex.from_tuples([(bp.strip(), c.strip()) for bp, c in df.columns])
    return df.apply(pd.to_numeric, errors="coerce")


def bodyparts(dlc: pd.DataFrame) -> list:
    return list(dict.fromkeys(dlc.columns.get_level_values(0)))


def calibration_point(calib: pd.DataFrame, bodypart: str) -> np.ndarray:
    """Median (x, y) in px of a static box point, over its confident frames."""
    x, y = calib[(bodypart, "x")], calib[(bodypart, "y")]
    use = ((calib[(bodypart, "likelihood")] >= settings.CALIBRATION_LIKELIHOOD_MIN)
           & x.notna() & y.notna())
    if use.sum() < settings.CALIBRATION_MIN_VALID_POINTS:
        use = x.notna() & y.notna()
    if not use.any():
        raise ValueError(f"no usable {bodypart} in the calibration file")
    return np.array([x[use].median(), y[use].median()])


def box_width_px(calibration_csv: Path) -> float:
    """cornerA-cornerB distance in px: a check that the camera geometry is as usual."""
    calib = read_dlc(calibration_csv, nrows=settings.BOX_CALIBRATION_NROWS)
    points = []
    for bp in ("cornerA", "cornerB"):
        ok = calib[(bp, "likelihood")] >= settings.BOX_CALIBRATION_LIKELIHOOD_MIN
        if ok.sum() < 10:
            raise ValueError(f"{bp} poorly detected in {Path(calibration_csv).name}")
        points.append(np.array([calib.loc[ok, (bp, "x")].median(),
                                calib.loc[ok, (bp, "y")].median()]))
    return float(np.linalg.norm(points[1] - points[0]))


def video_height_px(video: Optional[Path]) -> float:
    """Frame height of the video; the reference height when there is no video."""
    if video is None or not Path(video).exists():
        return float(settings.REFERENCE_VIDEO_HEIGHT_PX)
    cap = cv2.VideoCapture(str(video))
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap.release()
    return float(height) if height > 0 else float(settings.REFERENCE_VIDEO_HEIGHT_PX)


def read_trials(path: Path) -> pd.DataFrame:
    """Trial table: one row per trial with TrialNr, TrialType (Cue1p/Cue4p) and the
    cue-onset video frame (TrialLightOn_FrameIndex). Trials without an onset are dropped."""
    trials = pd.read_csv(path)
    missing = [c for c in (settings.ONSET_COLUMN, settings.TRIAL_TYPE_COLUMN,
                           settings.TRIAL_NUMBER_COLUMN) if c not in trials.columns]
    if missing:
        raise ValueError(f"{Path(path).name} is missing column(s): {', '.join(missing)}")
    trials = trials[pd.to_numeric(trials[settings.ONSET_COLUMN], errors="coerce").notna()].copy()
    trials["onset_frame"] = trials[settings.ONSET_COLUMN].astype(float).round().astype(int)
    return trials.reset_index(drop=True)
