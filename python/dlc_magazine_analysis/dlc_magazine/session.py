"""Run every per-frame measure on one session.

    frames, info = analyze_session(body_dlc_csv, calibration_dlc_csv, video_mp4)

frames has one row per video frame:

  frame, time_s
  centroid_x_px, centroid_y_px   body centroid in the image
  distance_to_magazine_cm        centroid to the magazine opening
  at_magazine                    body at the magazine (7 cm in / 12 cm out)
  head_in_magazine               magazine-entry rule (any of the three below)
  rule_snout, rule_head, rule_facing
  speed_cm_s                     whole-body speed
  rotation_deg_s                 angular speed of the neck -> tailbase axis
  pivot                          still point of the axis (0 = neck, 1 = tailbase)
  rotation_per_frame             head-anchored rotation, single frames
  rotation_bout                  head-anchored rotation as bouts (the one to use)

info holds the magazine geometry and the quality checks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from . import settings
from .dlc_io import box_width_px, read_dlc
from .magazine import MagazineFrame, at_magazine, centroid_px, distance_to_magazine_cm, magazine_entry
from .rotation import rotation_labels
from .speed import whole_body_speed


def analyze_session(body_dlc: Path, calibration_dlc: Path, video: Optional[Path] = None,
                    fps: float = settings.FPS,
                    rotation_cut: float = settings.ROTATION_CUT_DEG_S) -> Tuple[pd.DataFrame, Dict]:
    dlc = read_dlc(body_dlc)
    missing = [bp for bp in settings.BODYPARTS if bp not in dlc.columns.get_level_values(0)]
    if missing:
        raise ValueError(f"{Path(body_dlc).name} lacks bodypart(s) {missing}; "
                         "is this the BehSyl body model?")
    frame = MagazineFrame(calibration_dlc, video)

    centroid = centroid_px(dlc)
    distance = distance_to_magazine_cm(frame, centroid)
    at = at_magazine(distance)
    entry = magazine_entry(dlc, frame, fps)
    speed, _ = whole_body_speed(dlc, fps)
    rot = rotation_labels(dlc, at, rotation_cut, fps)

    frames = pd.DataFrame({
        "frame": np.arange(len(dlc)),
        "time_s": np.arange(len(dlc)) / fps,
        "centroid_x_px": centroid[:, 0], "centroid_y_px": centroid[:, 1],
        "distance_to_magazine_cm": distance,
        "at_magazine": at,
        "head_in_magazine": entry["in_magazine"],
        "rule_snout": entry["rule_snout"], "rule_head": entry["rule_head"],
        "rule_facing": entry["rule_facing"],
        "speed_cm_s": speed,
        "rotation_deg_s": rot["rotation_deg_s"], "pivot": rot["pivot"],
        "rotation_per_frame": rot["rotation_per_frame"],
        "rotation_bout": rot["rotation_bout"],
    })

    width = box_width_px(calibration_dlc)
    info = {
        "body_dlc": str(body_dlc), "calibration_dlc": str(calibration_dlc),
        "video": str(video) if video else "",
        "n_frames": len(dlc), "duration_min": len(dlc) / fps / 60,
        **frame.summary(),
        "mag1_x_px": frame.mag1_px[0], "mag1_y_px": frame.mag1_px[1],
        "mag2_x_px": frame.mag2_px[0], "mag2_y_px": frame.mag2_px[1],
        "box_width_px": width,
        "box_width_ok": abs(width / settings.BOX_WIDTH_REFERENCE_PX - 1) <= settings.BOX_WIDTH_TOLERANCE,
        "rotation_cut_deg_s": rotation_cut,
        "at_magazine_pct_of_session": 100 * at.mean(),
        "head_in_magazine_pct_of_session": 100 * entry["in_magazine"].mean(),
        "rotation_bout_pct_of_magazine_time": 100 * rot["rotation_bout"][at].mean() if at.any() else np.nan,
        "speed_valid_pct": 100 * np.isfinite(speed).mean(),
    }
    for bp in settings.BODYPARTS:
        info[f"likelihood_ok_pct_{bp.replace(' ', '_')}"] = \
            100 * (dlc[(bp, "likelihood")] >= settings.SPEED_LIKELIHOOD_MIN).mean()
    return frames, info
