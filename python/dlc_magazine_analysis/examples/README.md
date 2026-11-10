# Worked example: one session

**[analysis_overview.pdf](analysis_overview.pdf)** is a single poster-sized page
(A1) with one box per analysis step. Each box has a short explanation and the
step applied to the session below. A low-resolution copy is in
[analysis_overview_preview.png](analysis_overview_preview.png).

| Box | Step |
|---|---|
| 1 | Inputs: video + body DLC + calibration DLC + trial table |
| 2 | Pixels → floor cm → magazine-centred frame |
| 3 | The zones around the magazine |
| 4 | At the magazine (centroid, 7 cm in / 12 cm out) |
| 5 | Head in the magazine (the three entry rules) |
| 6 | Whole-body speed |
| 7 | Head-anchored rotation, per frame (axis turn speed, pivot s\*) |
| 8 | Rotation as bouts |
| 9 | Trials cut around cue onset (rasters) |
| 10 | Time courses, 1 pellet vs 4 pellets |
| 11 | Numbers per cue type (`conditions.csv`) |
| 12 | Overlay videos |

## The session

Cohort01, rat R205, Pavlovian 1p/4p, **fixed ITI, Day 9** (2024-08-15):
99 trials (79 × 1 pellet, 20 × 4 pellets), 78 min of video.
It is a high-rotation rat, so every step has something to show.

## example_data/

| File | What |
|---|---|
| `R205_fITI_D09_body_DLC.csv.gz` | Body DLC (BehSyl), the whole session |
| `R205_fITI_D09_calibration_DLC.csv.gz` | Calibration DLC (NewMar5), first 5,000 frames (all the code reads) |
| `R205_fITI_D09_trials.csv` | Trial table (from the corrected trial alignment) |
| `R205_fITI_D09_trial004_video_from_frame17287.mp4` | Raw video, trial 4 (1 pellet), −5 to +10 s |
| `R205_fITI_D09_trial008_video_from_frame21548.mp4` | Raw video, trial 8 (4 pellets), −5 to +10 s |

The number in each clip's name is the session frame it starts at; pass it as
`--video-first-frame`. The full video and DLC files are on the institute
server (`\\vs03\VS03-NandB-5\Georgios\FSCV_Rats_variable_vs_fixed_ITI\Data_collection\Videos\Cohort1\`
and `\\vs03\VS03-NandB-4\Allison\FSCV_Rat_Striatum_PAV1p4p-vITIfITI\Data_analysis\Daniela\New-Daniela-2026-03-05\videos-all\Cohort1\`).

To keep the repository small, the body DLC file is stored at 0.01 px precision
(likelihood truncated to 3 decimals, so no frame crosses a threshold). Compared
with the full-precision file, 28 of 140,055 rotation-bout frames differ, and
the 1-pellet rotation share moves from 30.0% to 30.2%. With the full-precision
file, every label and number matches the original analysis exactly.

## example_output/

Made with the commands in the main [README](../README.md#quick-start):

| File | What |
|---|---|
| `frames.csv.gz` | Every label for every frame |
| `trials.csv` | One row per trial |
| `conditions.csv` | One row per cue type: e.g. 4 pellets: 78.5% at magazine, 35.9% rotation, 13.4 cm/s |
| `time_course.csv` | The smoothed curves |
| `session_info.csv` | Geometry and tracking checks |
| `summary.png/.pdf` | The summary figure |
| `videos/trial004_Cue1p.mp4`, `videos/trial008_Cue4p.mp4` | Overlay videos |

## Rebuilding the poster

```bash
cd examples
python make_overview_figure.py
```

This needs `example_output/videos/trial008_Cue4p.mp4` (box 12), so run
`make_trial_video.py` first.
