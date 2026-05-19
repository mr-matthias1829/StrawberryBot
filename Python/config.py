"""Configuration and constants for strawberry detection system."""

import os

import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "runs", "detect", "superv2", "weights", "best.pt")

# =============================================================================
# FUSION THRESHOLDS
# =============================================================================
YOLO_BASE_THRESHOLD = 0.5      # Minimum YOLO conf to consider
CV_BASE_THRESHOLD = 0.6        # Minimum CV conf to consider
CV_DIRECT_ACCEPT_THRESHOLD = 0.72  # CV-only accept gate before zoom fallback
HIGH_AI_CONFIDENCE = 0.85      # If YOLO > this, trust even if CV says no
LOW_AI_CONFIDENCE = 0.4        # If YOLO < this, trigger zoom recheck

# Fusion weights when both agree
YOLO_FUSION_WEIGHT = 0.5
CV_FUSION_WEIGHT = 0.5

# =============================================================================
# CV PIPELINE — CONTOUR + CONVEXITY DEFECT
# =============================================================================
RED_LOWER1, RED_UPPER1 = np.array([0,   100, 135]),  np.array([10,  255, 255])
RED_LOWER2, RED_UPPER2 = np.array([170, 100, 135]),  np.array([179, 255, 255])

# Morphology
MORPH_OPEN_ITER = 3
MORPH_CLOSE_ITER = 5

# Contour filtering
MIN_CONTOUR_AREA = 200          # Very small — let fusion handle false positives
CONVEXITY_MIN_AREA = 3000       # Area threshold for trying to split clusters
MERGE_OVERLAP_RATIO = 0.45

# CV scoring weights (for cv_score_crop)
CV_WEIGHT_REDNESS = 0.35
CV_WEIGHT_CIRCULARITY = 0.25
CV_WEIGHT_SIZE = 0.20
CV_WEIGHT_TEXTURE = 0.15
CV_WEIGHT_TEMPORAL = 0.05       # Not used in single-frame, placeholder

# Size scoring parameters (adaptive — these are reference values)
BERRY_SIZE_IDEAL = 5000         # px² — ideal berry size in frame
BERRY_SIZE_MIN = 4             # px² — below this, size score decays
BERRY_SIZE_MAX = 25000          # px² — above this, size score decays

# =============================================================================
# ZOOM RECHECK (fallback refinement)
# =============================================================================
MAX_RECHECKS = 2
ZOOM_SCALE_FACTOR = 2.0
RECHECK_AI_CONF = 0.65
RECHECK_CV_CONF = 0.55

# =============================================================================
# TEMPORAL MEMORY
# =============================================================================
PERSISTENCE_REQUIRED = 2
PERSISTENCE_REQUIRED_CV_ONLY = 3
PERSISTENCE_DECAY = 0.7
IOU_MATCH_THRESHOLD = 0.4

# Possible-hit lane (kept separate from confirmed hits)
POSSIBLE_HIT_MIN_CONF = 0.50
POSSIBLE_HIT_MIN_SEEN = 1

# Source-aware possible-hit tuning: CV-only is easier to keep as possible,
# AI-only is stricter and down-weighted due to known false positives.
POSSIBLE_CV_ONLY_MIN_CONF = 0.6
POSSIBLE_CV_ONLY_MIN_SEEN = 1
POSSIBLE_AI_ONLY_MIN_CONF = 0.6
POSSIBLE_AI_ONLY_MIN_SEEN = 1
POSSIBLE_AI_CONF_WEIGHT = 0.5

# If no confirmed detections exist, optionally steer toward strong possible hits.
POSSIBLE_TARGET_FALLBACK_ENABLED = True
POSSIBLE_TARGET_MIN_CONF = 0.60

# =============================================================================
# DISPLAY
# =============================================================================
SHOW_DEBUG_WINDOWS = True
COLOR_AI = (0, 255, 0)
COLOR_CV = (255, 80, 0)      # Orange-blue from your script
COLOR_FUSED = (0, 255, 255)
COLOR_ZOOMED = (255, 255, 0)  # Cyan for zoom-rechecked boxes
COLOR_POSSIBLE = (180, 100, 255)
