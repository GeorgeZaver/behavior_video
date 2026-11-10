"""Every number the analysis uses, in one place.

The values are the ones used in the Dopamine-and-Timing analyses (Day-9 FSCV
rats, Pavlovian 1-pellet / 4-pellet task, fixed and variable ITI). Change them
here, not inside the functions.
"""

import numpy as np

# -----------------------------------------------------------------------------
# Video
# -----------------------------------------------------------------------------
FPS = 29.97                       # frames per second of the session videos

# The camera looks at the box at an angle, so one pixel is ~0.18 cm wide
# but only ~0.11 cm tall. A single cm/px factor would overestimate movement
# along y. This 2x2 matrix maps a pixel vector to a floor-plane cm vector:
#     cm = PX_TO_CM @ [x_px, y_px]
# It is the median linear part of an affine fit between raw pixel tracking
# and the lab's angle-corrected cm tracking over 20 Day-9 sessions
# (per-session SD <= 0.005).
PX_TO_CM = np.array([[0.1811, 0.0071],
                     [-0.0144, 0.1080]])

# PX_TO_CM was fitted on 480-px-high videos. Videos of another height have
# their y coordinates scaled to this height before the magazine geometry is
# built (magazine.MagazineFrame).
REFERENCE_VIDEO_HEIGHT_PX = 480

# -----------------------------------------------------------------------------
# DLC models and bodyparts
# -----------------------------------------------------------------------------
# Body model "BehSyl projectDec22": 9 points on the rat.
BODY_MODEL_TOKEN = "behsyl"
BODYPARTS = ["snout", "neck", "back1", "back2", "back3", "back4",
             "tailbase", "right paw", "left paw"]

# Calibration model "NewMar5": static points of the box (corners, walls,
# light, and the two ends of the magazine opening, mag1 and mag2).
CALIBRATION_MODEL_TOKEN = "newmar5"

# -----------------------------------------------------------------------------
# Magazine geometry (from the calibration DLC)
# -----------------------------------------------------------------------------
CALIBRATION_NROWS = 5000          # static points: the first 5000 frames are enough
CALIBRATION_LIKELIHOOD_MIN = 0.8  # frames used for the median position
CALIBRATION_MIN_VALID_POINTS = 20 # fewer confident frames -> use every frame
INTERIOR_REFERENCE_BODYPART = "wall2"   # tells which side of the magazine is the box

# Box-size check: cornerA-cornerB distance in px, compared with the median of
# the 37 Day-9 sessions (135.07 px). Sessions off by more than 3% are flagged.
BOX_CALIBRATION_NROWS = 3000
BOX_CALIBRATION_LIKELIHOOD_MIN = 0.95
BOX_WIDTH_REFERENCE_PX = 135.07
BOX_WIDTH_TOLERANCE = 0.03

# -----------------------------------------------------------------------------
# "At the magazine": body centroid near the magazine opening
# -----------------------------------------------------------------------------
# Centroid = mean of the 9 body points, each with its low-confidence frames
# interpolated over (the way keypoint-MoSeq initialises its centroid; agrees
# with KM's own centroid on 99.2% of frames).
CENTROID_LIKELIHOOD_MIN = 0.5

# Hysteresis: the rat ARRIVES when the centroid comes closer than 7 cm to the
# magazine opening, and LEAVES only when it goes further than 12 cm. Between
# 7 and 12 cm it keeps its previous state, so wobbling around one radius does
# not create fake visits.
AT_MAGAZINE_CM = 7.0
LEFT_MAGAZINE_CM = 12.0

# -----------------------------------------------------------------------------
# "Head in the magazine": magazine-entry rule (validated on videos)
# -----------------------------------------------------------------------------
ENTRY_LIKELIHOOD_MIN = 0.8
SNOUT_CIRCLE_RADIUS_PER_MAG_LENGTH = 0.75   # green circle radius = 0.75 x opening length
MAG_ENTRY_RADIUS_CM = 9.748724              # orange semicircle (into the box)
SNOUT_MAG_ENTRY_BREAK_BUFFER_S = 0.5        # gaps <= 0.5 s in the snout rule are closed
MAG_ENTRY_BREAK_BUFFER_S = 0.5              # and again in the combined rule

# -----------------------------------------------------------------------------
# Whole-body speed
# -----------------------------------------------------------------------------
SPEED_LIKELIHOOD_MIN = 0.8        # both frames must be confident
MAX_BODYPART_SPEED_CM_S = 150.0   # faster jumps are DLC label swaps
MIN_VALID_BODYPARTS = 3           # a frame needs >= 3 usable points

# -----------------------------------------------------------------------------
# Head-anchored rotation
# -----------------------------------------------------------------------------
ROTATION_LIKELIHOOD_MIN = 0.6     # neck / tailbase frames below this are gaps
ROTATION_GAP_FRAMES = 5           # gaps up to 5 frames are interpolated
SMOOTH_FRAMES = 5                 # centred rolling window for speeds

# Body axis = neck -> tailbase. A frame "rotates" when this axis turns faster
# than ROTATION_CUT_DEG_S. The cut is the Youden-optimal threshold against
# hand-annotated turning syllables (keypoint-MoSeq syllables 3/34/58) on
# at-magazine frames of 10 rats (AUC 0.86, TPR 0.79, FPR 0.21).
ROTATION_CUT_DEG_S = 44.61381065158484

# The pivot s* is the point of the body axis that moves least
# (0 = neck, 1 = tailbase). "Anchored at the head" = |s*| <= 0.25.
PIVOT_HEAD_MAX = 0.25

# Per-frame labels flicker, so rotation is counted as bouts:
#   1. gaps of <= 5 frames (167 ms) between rotation frames are closed;
#   2. bouts shorter than 5 frames (167 ms) after merging are dropped.
BOUT_MERGE_GAP_FRAMES = 5
BOUT_MIN_FRAMES = 5

# High / low rotation rats: fITI rotation (% of magazine time, 0-5 s after
# cue onset, mean of the 1p and 4p values) above 15% = "high".
ROTATION_GROUP_SPLIT_PCT = 15.0

# -----------------------------------------------------------------------------
# Trial windows (seconds relative to cue onset)
# -----------------------------------------------------------------------------
MEASURE_WINDOW_S = 5.0            # per-trial measures: 0 to 5 s after cue onset
COURSE_PRE_S, COURSE_POST_S = 2.0, 8.0   # time courses: -2 to 8 s
SPEED_PRE_WINDOW = (-5.0, 0.0)    # speed before the cue
SPEED_POST_WINDOW = (0.0, 5.0)    # speed during the cue
TIMECOURSE_SIGMA_S = 0.26         # Gaussian smoothing of the time courses
MIN_MAGAZINE_S = 5.0              # < 5 s at the magazine in a condition -> unreliable

# Trial-table columns
ONSET_COLUMN = "TrialLightOn_FrameIndex"   # video frame of cue-light onset
TRIAL_TYPE_COLUMN = "TrialType"            # Cue1p / Cue4p
TRIAL_NUMBER_COLUMN = "TrialNr"
