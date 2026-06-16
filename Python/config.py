"""configuration and constants for detection system."""

import os
import numpy as np

BASE_DIR   = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "runs", "detect", "CV", "weights", "best.pt")

# ---------------------------------------------------------------------------
# FUSION THRESHOLDS
# ---------------------------------------------------------------------------
YOLO_BASE_THRESHOLD = 0.50       # minimum YOLO conf to even consider
CV_BASE_THRESHOLD   = 0.50

# CV score needed to accept WITHOUT any AI corroboration
CV_DIRECT_ACCEPT_THRESHOLD = 0.65

# ---------------------------------------------------------------------------
# AI CONFIDENCE TIERS  (drives _accept_ai_cv_pair in fusion_engine)
# ---------------------------------------------------------------------------
# HIGH  → trust the AI box outright; CV is optional but boosts confidence
# MED   → trust unless CV actively disagrees (CV score < MED_CV_VETO)
# LOW   → only accept if CV corroborates (CV score >= LOW_CV_CONFIRM)
# below LOW → send to zoom recheck, never accept raw

AI_TRUST_HIGH         = 0.75   # was HIGH_AI_CONFIDENCE
AI_TRUST_MED          = 0.55
AI_TRUST_LOW          = 0.40   # was LOW_AI_CONFIDENCE

AI_TRUST_MED_CV_VETO   = 0.25  # CV score BELOW this vetoes a MED AI box
AI_TRUST_LOW_CV_CONFIRM = 0.45  # CV score needed to confirm a LOW AI box

# Fusion weights when both detectors contribute
AI_DOMINANT_FUSION_WEIGHT  = 0.70   # was YOLO_FUSION_WEIGHT
CV_MINORITY_FUSION_WEIGHT  = 0.30   # was CV_FUSION_WEIGHT

# legacy aliases so web_server / other modules don't break
YOLO_FUSION_WEIGHT  = AI_DOMINANT_FUSION_WEIGHT
CV_FUSION_WEIGHT    = CV_MINORITY_FUSION_WEIGHT
HIGH_AI_CONFIDENCE  = AI_TRUST_HIGH
LOW_AI_CONFIDENCE   = AI_TRUST_LOW

# ---------------------------------------------------------------------------
# KALMAN TRACKER  (KalmanBoxTrack in fusion_engine)
# ---------------------------------------------------------------------------
# How large the process noise is for position vs velocity states.
# Larger = filter trusts measurements more (snappier but noisier).
KALMAN_PROCESS_NOISE_POS   = 1.0
KALMAN_PROCESS_NOISE_VEL   = 10.0

# Measurement noise per detector type.
# Smaller = filter trusts that detector's bbox more.
KALMAN_MEASUREMENT_NOISE_AI = 25.0   # AI boxes are usually precise
KALMAN_MEASUREMENT_NOISE_CV = 60.0   # CV boxes are noisier

# A track is "confirmed" after this many REAL updates AND this much wall-time
TRACK_CONFIRM_MIN_UPDATES = 3
TRACK_CONFIRM_TIME_S      = 0.20   # seconds since track creation

# A track is "lost" (removed) after coasting this long without a real update
TRACK_COAST_MAX_S         = 0.60   # seconds

# Association gating: a detection must pass AT LEAST ONE of these tests
TRACK_MATCH_IOU_MIN           = 0.15   # relaxed from 0.30 — helps fast movers
TRACK_MATCH_CENTER_DIST_MAX   = 1.50   # max normalized center distance (diagonal units)

# ---------------------------------------------------------------------------
# FRAME-AGE DEBOUNCE  (addresses real camera latency)
# ---------------------------------------------------------------------------
# Detections older than this (wall-clock seconds since frame capture) are
# excluded from driving the robot.  Tune to ≈ your observed camera latency.
MAX_ACCEPTABLE_FRAME_AGE_S = 1.5   # seconds  — start conservative, tune down

# ---------------------------------------------------------------------------
# ZOOM RECHECK  (per-track, time-based cooldown)
# ---------------------------------------------------------------------------
MAX_RECHECKS              = 2    # legacy alias kept for web_server schema
ZOOM_RECHECK_MAX_PER_TRACK = 3
ZOOM_RECHECK_COOLDOWN_S    = 1.5  # seconds between recheck requests for one track
ZOOM_SCALE_FACTOR          = 2.0
RECHECK_AI_CONF            = 0.55
RECHECK_CV_CONF            = 0.45

# ---------------------------------------------------------------------------
# POSSIBLE-HIT FALLBACK
# ---------------------------------------------------------------------------
POSSIBLE_TARGET_FALLBACK_ENABLED = True
POSSIBLE_TARGET_MIN_CONF         = 0.50

# (legacy keys used by web_server schema — kept so the dashboard doesn't break)
PERSISTENCE_REQUIRED             = 3
PERSISTENCE_REQUIRED_CV_ONLY     = 4
PERSISTENCE_DECAY                = 0.85
IOU_MATCH_THRESHOLD              = 0.30
POSSIBLE_HIT_MIN_CONF            = 0.50
POSSIBLE_HIT_MIN_SEEN            = 1
POSSIBLE_CV_ONLY_MIN_CONF        = 0.50
POSSIBLE_CV_ONLY_MIN_SEEN        = 1
POSSIBLE_AI_ONLY_MIN_CONF        = 0.60
POSSIBLE_AI_ONLY_MIN_SEEN        = 1
POSSIBLE_AI_CONF_WEIGHT          = 0.50

# ---------------------------------------------------------------------------
# CV COLOUR RANGES  (HSV, OpenCV convention)
# ---------------------------------------------------------------------------
RED_LOWER1, RED_UPPER1 = np.array([0,   100, 135]), np.array([10,  255, 255])
RED_LOWER2, RED_UPPER2 = np.array([170, 100, 135]), np.array([179, 255, 255])

# Morphology
MORPH_OPEN_ITER  = 3
MORPH_CLOSE_ITER = 5

# Contour filtering
MIN_CONTOUR_AREA    = 150
CONVEXITY_MIN_AREA  = 3000
MERGE_OVERLAP_RATIO = 0.45

# CV scoring weights (must sum to 1.0)
CV_WEIGHT_REDNESS     = 0.50
CV_WEIGHT_CIRCULARITY = 0.25
CV_WEIGHT_SIZE        = 0.15
CV_WEIGHT_TEXTURE     = 0.10
CV_WEIGHT_TEMPORAL    = 0.05   # unused, kept for schema compat

# Size scoring reference values (px² in inference frame)
BERRY_SIZE_IDEAL = 15_000
BERRY_SIZE_MIN   = 200
BERRY_SIZE_MAX   = 500_000

# ---------------------------------------------------------------------------
# GRIPPER
# ---------------------------------------------------------------------------
GRIPPER_BB_WIDTH          = 500
GRIPPER_BB_HEIGHT         = 500
GRIPPER_CONTAINMENT_FRAMES = 3
GRIPPER_AUTO_GRIP_ENABLED  = True
MIN_GRAB_AREA_RATIO        = 0.1
GRAB_CENTER_TOLERANCE_X    = 15
GRAB_CENTER_TOLERANCE_Y    = 15

# ---------------------------------------------------------------------------
# DISPLAY / MISC
# ---------------------------------------------------------------------------
SHOW_DEBUG_WINDOWS = False
COLOR_AI       = (0,   255,   0)
COLOR_CV       = (255,  80,   0)
COLOR_FUSED    = (0,   255, 255)
COLOR_ZOOMED   = (255, 255,   0)
COLOR_POSSIBLE = (180, 100, 255)

AUTO_MODE_ALLOW_MOVE = True