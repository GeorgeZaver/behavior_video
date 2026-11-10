"""Analyse one session: magazine, rotation and speed from DLC tracking.

Inputs
  --body-dlc         DLC csv of the rat ("BehSyl" model, 9 bodyparts)
  --calibration-dlc  DLC csv of the box ("NewMar5" model: mag1, mag2, corners, walls)
  --trials           (optional) trial table with TrialNr, TrialType and
                     TrialLightOn_FrameIndex (video frame of cue onset)
  --video            (optional) the session video; only its height is read, to
                     scale the magazine geometry if it is not 480 px high

Outputs (in --out)
  frames.csv.gz      one row per video frame, every label (see dlc_magazine/session.py)
  session_info.csv   magazine geometry, box-size check, tracking quality
  and, with --trials:
  trials.csv         one row per trial (0-5 s after cue onset; speed -5-0 and 0-5 s)
  conditions.csv     one row per cue type, frames pooled over trials
  time_course.csv    P(at magazine), P(head in), P(rotation | at), speed, -2 to 8 s
  summary.png/.pdf   the figure of the above

Example
  python analyze_session.py ^
      --body-dlc examples/example_data/R205_fITI_D09_body_DLC.csv.gz ^
      --calibration-dlc examples/example_data/R205_fITI_D09_calibration_DLC.csv.gz ^
      --trials examples/example_data/R205_fITI_D09_trials.csv ^
      --out examples/example_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from dlc_magazine import analyze_session, settings
from dlc_magazine import trials as trial_tools
from dlc_magazine.dlc_io import read_trials
from dlc_magazine.plots import session_figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--body-dlc", type=Path, required=True)
    parser.add_argument("--calibration-dlc", type=Path, required=True)
    parser.add_argument("--trials", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--title", default="", help="figure title, e.g. the rat and session")
    parser.add_argument("--fps", type=float, default=settings.FPS)
    parser.add_argument("--rotation-cut", type=float, default=settings.ROTATION_CUT_DEG_S,
                        help="deg/s; default is the calibrated 44.6")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    print("Reading tracking and computing per-frame labels ...")
    frames, info = analyze_session(args.body_dlc, args.calibration_dlc, args.video,
                                   args.fps, args.rotation_cut)
    frames.to_csv(args.out / "frames.csv.gz", index=False, float_format="%.4f")
    pd.DataFrame([info]).T.rename(columns={0: "value"}).to_csv(args.out / "session_info.csv")
    print(f"  {info['n_frames']} frames ({info['duration_min']:.1f} min); "
          f"magazine opening {info['mag_length_cm']:.2f} cm; "
          f"box width {info['box_width_px']:.1f} px ({'ok' if info['box_width_ok'] else 'CHECK'})")
    print(f"  at magazine {info['at_magazine_pct_of_session']:.1f}% of the session, "
          f"head in magazine {info['head_in_magazine_pct_of_session']:.1f}%, "
          f"rotation {info['rotation_bout_pct_of_magazine_time']:.1f}% of magazine time")

    if args.trials is None:
        print(f"No trial table given; per-frame output only. Output: {args.out}")
        return

    trials = read_trials(args.trials)
    table = trial_tools.trial_frames(frames, trials, args.fps)
    per_trial = trial_tools.per_trial(frames, trials, args.fps)
    conditions = trial_tools.per_condition(table, args.fps)
    curves = trial_tools.time_course(table, args.fps)
    per_trial.to_csv(args.out / "trials.csv", index=False, float_format="%.4f")
    conditions.to_csv(args.out / "conditions.csv", index=False, float_format="%.4f")
    curves.to_csv(args.out / "time_course.csv", index=False, float_format="%.5f")
    n_dropped = (~per_trial["usable"]).sum()
    print(f"  {len(trials)} trials ({n_dropped} outside the video, left out)")
    print(conditions.round(1).to_string(index=False))

    fig = session_figure(table, curves, conditions, args.title)
    for suffix in ("png", "pdf"):
        fig.savefig(args.out / f"summary.{suffix}", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Done. Output: {args.out}")


if __name__ == "__main__":
    main()
