"""Whole-body speed (cm/s) from the body DLC.

Per bodypart: frame-to-frame displacement, mapped to floor-plane cm with
PX_TO_CM, times the frame rate. A sample is kept only when both frames have
likelihood >= 0.8 and the speed is <= 150 cm/s (faster jumps are DLC label
swaps). Whole-body speed per frame = mean over the usable bodyparts, when at
least 3 are usable; otherwise NaN.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

from . import settings
from .dlc_io import bodyparts


def whole_body_speed(dlc: pd.DataFrame, fps: float = settings.FPS) -> Tuple[np.ndarray, np.ndarray]:
    """Per-frame mean speed (cm/s) over the usable bodyparts, and how many were usable."""
    parts = bodyparts(dlc)
    speeds = np.full((len(dlc), len(parts)), np.nan)
    for i, bp in enumerate(parts):
        x = dlc[(bp, "x")].to_numpy()
        y = dlc[(bp, "y")].to_numpy()
        valid = dlc[(bp, "likelihood")].to_numpy() >= settings.SPEED_LIKELIHOOD_MIN
        step_cm = settings.PX_TO_CM @ np.vstack([np.diff(x), np.diff(y)])
        step = np.hypot(step_cm[0], step_cm[1]) * fps
        keep = valid[1:] & valid[:-1] & (step <= settings.MAX_BODYPART_SPEED_CM_S)
        speeds[1:, i][keep] = step[keep]
    n_valid = np.isfinite(speeds).sum(axis=1)
    enough = np.where(n_valid[:, None] >= settings.MIN_VALID_BODYPARTS, speeds, np.nan)
    with np.errstate(invalid="ignore"):
        mean = np.full(len(dlc), np.nan)
        rows = np.isfinite(enough).any(axis=1)
        mean[rows] = np.nanmean(enough[rows], axis=1)
    return mean, n_valid
