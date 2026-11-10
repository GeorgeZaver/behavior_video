# DLC magazine analysis: magazine, rotation and speed

This code takes DeepLabCut (DLC) tracking of a rat in an operant box and gives,
for every video frame:

| Measure | Question it answers |
|---|---|
| **At the magazine** | Is the rat's body near the reward magazine? |
| **Head in the magazine** | Is the rat's head in the magazine? |
| **Whole-body speed** | How fast is the rat moving (cm/s)? |
| **Head-anchored rotation** | Is the rat keeping its head at the magazine while swinging its body around it? |

It then cuts these around each cue onset and summarises them per trial and per
cue type (1 pellet / 4 pellets).

**Start here:** [examples/analysis_overview.pdf](examples/analysis_overview.pdf)
is a one-page poster with a box for each step and a worked example from a real
session. The [examples](examples/) folder has that session's data, so you can
run everything below straight away.

The code comes from the Day-9 analyses of the FSCV dopamine project
(Pavlovian 1p/4p task, fixed vs variable ITI; repository *Dopamine-and-Timing*).
It was split out of that repository and made self-contained. The algorithms and
every threshold are unchanged. On the example session, every per-frame label
matches the original code on all 140,055 frames, and the per-cue numbers match
the published table.

---

## Quick start

```bash
pip install -r requirements.txt     # numpy, pandas, scipy, matplotlib, opencv-python
                                    # ffmpeg must be on PATH for the videos

# 1. Analyse a session -> tables + summary figure
python analyze_session.py \
    --body-dlc        examples/example_data/R205_fITI_D09_body_DLC.csv.gz \
    --calibration-dlc examples/example_data/R205_fITI_D09_calibration_DLC.csv.gz \
    --trials          examples/example_data/R205_fITI_D09_trials.csv \
    --out             examples/example_output

# 2. Overlay video of one trial, to check by eye what the analysis sees
python make_trial_video.py \
    --body-dlc        examples/example_data/R205_fITI_D09_body_DLC.csv.gz \
    --calibration-dlc examples/example_data/R205_fITI_D09_calibration_DLC.csv.gz \
    --trials          examples/example_data/R205_fITI_D09_trials.csv \
    --video           examples/example_data/R205_fITI_D09_trial008_video_from_frame21548.mp4 \
    --video-first-frame 21548 --trial 8 \
    --out             examples/example_output/videos
```

For your own session, pass the full session video and leave out `--video-first-frame`.
On Windows, write each command on one line, or replace `\` with `^`.

---

## What you need per session

| File | DLC model | Contents |
|---|---|---|
| Body tracking `…DLC_resnet50_BehSyl projectDec22…csv` | BehSyl | 9 points on the rat: snout, neck, back1–4, tailbase, right paw, left paw |
| Calibration tracking `…DLC_resnet50_NewMar5…csv` | NewMar5 | Static box points: cornerA/B, wall1–4, light, **mag1 / mag2** (the two ends of the magazine opening) |
| Trial table (csv) | – | One row per trial: `TrialNr`, `TrialType` (`Cue1p` / `Cue4p`), `TrialLightOn_FrameIndex` (video frame of cue onset) |
| Video (mp4), optional | – | Only its height is read (geometry is scaled if it is not 480 px), and it is drawn under the overlay videos |

DLC files can be plain `.csv` or `.csv.gz`. Body and calibration tracking must
come from the same video, so the frame numbers line up.

---

## The steps (folder `dlc_magazine/`)

All numbers live in **[dlc_magazine/settings.py](dlc_magazine/settings.py)**, with a
comment on each one. Change them there, not inside the functions.

### 1. Pixels → floor cm → magazine frame  (`magazine.py`, `MagazineFrame`)
The camera views the box at an angle, so one pixel is ~0.18 cm wide but only
~0.11 cm tall. A 2×2 matrix (`PX_TO_CM`) maps pixel vectors to floor-plane cm.
It is the median fit between raw and angle-corrected tracking over 20 sessions.
The magazine opening (mag1, mag2; median over confident calibration frames)
then defines a frame centred on the opening: **x = cm into the box,
y = cm along the opening**. A box-size check compares the cornerA–cornerB
distance with the usual 135 px (±3%).

### 2. At the magazine  (`magazine.py`: `centroid_px`, `distance_to_magazine_cm`, `at_magazine`)
- Body centroid = mean of the 9 body points, with each point's low-confidence
  frames (likelihood < 0.5) interpolated.
- Distance = centroid to the magazine opening, treated as a line segment (so
  standing beside the opening still counts as close).
- **Hysteresis:** the rat *arrives* when the distance drops below **7 cm** and
  *leaves* only when it goes above **12 cm**. Between 7 and 12 cm it keeps its
  previous state, so wobbling around one radius does not create fake visits.

### 3. Head in the magazine  (`magazine.py`: `magazine_entry`)
This rule was validated by watching videos. A frame is an entry when any of these holds
(points need likelihood ≥ 0.8):
- **snout:** the snout is inside a circle around the opening (radius 0.75 × opening length);
- **head:** the neck→snout segment touches that circle, and back1 is inside a
  9.75 cm semicircle that opens into the box;
- **facing:** back1 is inside the semicircle and the back1→neck direction points at the opening.

Gaps of ≤ 0.5 s are closed (first in the snout rule, then in the combined rule).

### 4. Whole-body speed  (`speed.py`)
For each body point: the step from one frame to the next (through `PX_TO_CM`) × 29.97 fps.
A step is kept only when both frames have likelihood ≥ 0.8 and the speed is
≤ 150 cm/s (bigger jumps are DLC label swaps). Whole-body speed = mean over the
usable points, when at least 3 are usable.

### 5. Head-anchored rotation  (`rotation.py`)
A frame counts as a rotation when all three hold:
1. the rat is **at the magazine** (step 2);
2. the **neck→tailbase axis turns faster than 44.6 °/s**. This cut is the best
   threshold (Youden) against hand-annotated keypoint-MoSeq turning syllables
   (3/34/58), on 10 rats (AUC 0.86);
3. the **pivot is at the head**: s\*, the point on the axis that moves least,
   satisfies |s\*| ≤ 0.25 (s\* = 0 neck, 1 tailbase). For a rigid axis
   s\* = argmin_s |v_neck + s·(v_tail − v_neck)|.

The neck, not the snout, is the anchor, because the snout keeps sniffing while
the head stays at the magazine.

**Bouts:** the single-frame label flickers, so rotation is counted as bouts,
built on the whole session before trials are cut. Gaps ≤ 5 frames (167 ms) are
closed, bouts < 5 frames are dropped, and bout frames must be at the magazine.
**Every reported rotation number uses bouts** (`rotation_bout`). The per-frame
label is only kept for reference.

### 6. Trials  (`trials.py`)
- `trial_frames`: every frame from −2 to +8 s around each cue onset (trials
  whose window leaves the video are dropped).
- `per_trial`: one row per trial. Magazine and rotation measures cover 0–5 s
  after cue onset; speed is given for −5–0 s and 0–5 s.
- `per_condition`: one row per cue type, with **frames pooled over trials**.
  These are the per-rat values used in the group analyses:
  `magazine_pct`, `rotation_pct_of_magazine` (the headline rotation measure),
  `rotation_pct_of_window`, `rotation_deg_per_magazine_s`,
  `head_in_magazine_pct`, `speed_cm_s`, and `reliable` (≥ 5 s at the magazine).
- `time_course`: P(at magazine), P(head in), P(rotation | at magazine) and
  speed per frame offset. Numerator and denominator are smoothed separately
  (Gaussian, σ = 0.26 s).

`session.py` (`analyze_session`) runs steps 1–5 on one session; `plots.py`
draws the summary figure.

---

## Outputs of `analyze_session.py`

| File | Contents |
|---|---|
| `frames.csv.gz` | One row per video frame: centroid, distance, `at_magazine`, `head_in_magazine` (+ the three rules), `speed_cm_s`, `rotation_deg_s`, `pivot`, `rotation_per_frame`, `rotation_bout` |
| `session_info.csv` | Magazine geometry, box-size check, % of frames with good tracking per body point, session-level shares |
| `trials.csv` | One row per trial (see step 6) |
| `conditions.csv` | One row per cue type (see step 6) |
| `time_course.csv` | The smoothed curves, −2 to +8 s |
| `summary.png/.pdf` | Time courses, trial rasters (grey away / green at magazine / pink rotation), per-cue bars |

## Across rats

This folder works one session at a time. In the original project:
- rats were compared with their fITI and vITI sessions **kept separate**, never pooled;
- **high / low rotation rats** were split by their fITI bout share
  (`rotation_pct_of_magazine`, 0–5 s, mean of 1p and 4p) above / below **15%**
  (`ROTATION_GROUP_SPLIT_PCT`).

To build a group table, loop `analyze_session.py` over sessions and concatenate
the `conditions.csv` files with rat / ITI columns added.

## Notes and limits

- `PX_TO_CM`, the radii and the rotation cut were fitted for this camera set-up
  (352 × 480 px videos, 29.97 fps). In a different box or camera position,
  re-fit them or at least check them with the overlay videos.
- The rotation cut was calibrated against keypoint-MoSeq syllables. That
  calibration is not part of this folder; the resulting value is stored in `settings.py`.
- Cohort 3 has no angle-corrected tracking. That doesn't matter here, because
  everything is computed from the raw pixel tracking.
