"""Configuration and constants for strawberry detection system."""

import os

import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "runs", "detect", "CV", "weights", "best.pt")

# =============================================================================
# FUSION THRESHOLDS
# =============================================================================
YOLO_BASE_THRESHOLD = 0.5       # Minimum YOLO conf to consider

# Lowered from 0.60 → 0.50: CV scores are more conservative now that the
# redness denominator is tighter; more candidates should reach fusion.
CV_BASE_THRESHOLD = 0.50

# Lowered from 0.72 → 0.65: previous value was hard to reach with the old
# scoring — a well-lit berry with good shape now reliably clears 0.65.
CV_DIRECT_ACCEPT_THRESHOLD = 0.5

HIGH_AI_CONFIDENCE = 0.85       # If YOLO > this, trust even without CV
LOW_AI_CONFIDENCE  = 0.40       # If YOLO < this, trigger zoom recheck

# Fusion weights when both detectors agree on the same berry
YOLO_FUSION_WEIGHT = 0.65       # AI slightly preferred when both agree
CV_FUSION_WEIGHT   = 0.35

# =============================================================================
# CV PIPELINE — HSV COLOUR RANGES
# =============================================================================
# Lower bound sat 100, val cv1 135 keeps pale / dark non-berry reds out.
# If distant berries are being missed, try lowering sat to 80 and val cv1 to 100.
RED_LOWER1, RED_UPPER1 = np.array([0,   100, 135]), np.array([10,  255, 255])
RED_LOWER2, RED_UPPER2 = np.array([170, 100, 135]), np.array([179, 255, 255])

# Morphology
MORPH_OPEN_ITER  = 3
MORPH_CLOSE_ITER = 5

# Contour filtering
MIN_CONTOUR_AREA    = 150       # Slightly lower than 200 — distant berries are small
CONVEXITY_MIN_AREA  = 3000      # Area at which watershed cluster-splitting kicks in
MERGE_OVERLAP_RATIO = 0.45      # (legacy, kept for zoom-recheck path)

# CV scoring weights — REDNESS + CIRCULARITY + SIZE + TEXTURE must sum to 1.0.
# TEMPORAL is a placeholder kept for future use; it is NOT included in scoring.
#
# Redness:     0.40 — most reliable single-frame strawberry signal.
# Circularity: 0.40 — shape quality; low values already pre-filtered at contour
#                     stage so this mainly rewards well-formed berries.
# Size:        0.15 — penalises implausibly tiny or huge blobs.
# Texture:     0.05 — Laplacian variance is a weak tiebreaker ONLY. Keeping it
#                     low because it also rewards grass, fabric, and hair; a high
#                     weight here produces false positives on textured backgrounds.
CV_WEIGHT_REDNESS     = 0.40
CV_WEIGHT_CIRCULARITY = 0.40
CV_WEIGHT_SIZE        = 0.15
CV_WEIGHT_TEXTURE     = 0.05
CV_WEIGHT_TEMPORAL    = 0.05    # Placeholder — not used in single-frame scoring

# Size scoring reference values (px² in the inference frame)
BERRY_SIZE_IDEAL = 15_000         # px² — ideal berry area
BERRY_SIZE_MIN   = 200            # px² — below this, size score decays toward 0
BERRY_SIZE_MAX   = 500_000        # px² — above this, size score decays toward 0

# =============================================================================
# ZOOM RECHECK (fallback refinement)
# =============================================================================
MAX_RECHECKS      = 2
ZOOM_SCALE_FACTOR = 2.0
RECHECK_AI_CONF   = 0.65
RECHECK_CV_CONF   = 0.50        # Lowered from 0.55 — zoom crops can be tight/partial

# =============================================================================
# TEMPORAL MEMORY / TRACKING
# =============================================================================
PERSISTENCE_REQUIRED        = 2   # Frames needed to confirm (AI / fused)
PERSISTENCE_REQUIRED_CV_ONLY = 3  # Frames needed to confirm (CV-only source)
PERSISTENCE_DECAY           = 0.7
IOU_MATCH_THRESHOLD         = 0.40  # Standard IoU gate (containment check is separate)

# ---------------------------------------------------------------------------
# Possible-hit lane
# ---------------------------------------------------------------------------
# Generic fallback (fused / unknown source)
POSSIBLE_HIT_MIN_CONF = 0.50
POSSIBLE_HIT_MIN_SEEN = 1

# CV-only possible: lowered from 0.60 → 0.50 to account for the more
# conservative CV scoring after the redness denominator tightening.
POSSIBLE_CV_ONLY_MIN_CONF = 0.50
POSSIBLE_CV_ONLY_MIN_SEEN = 1

# AI-only possible: kept strict — AI false-positives are more common.
POSSIBLE_AI_ONLY_MIN_CONF = 0.60
POSSIBLE_AI_ONLY_MIN_SEEN = 1
POSSIBLE_AI_CONF_WEIGHT   = 0.50   # Down-weight AI conf for possible classification

# Possible → target fallback
POSSIBLE_TARGET_FALLBACK_ENABLED = True
POSSIBLE_TARGET_MIN_CONF         = 0.50  # Lowered from 0.60 to match CV_ONLY floor

# =============================================================================
# DISPLAY
# =============================================================================
SHOW_DEBUG_WINDOWS = True
COLOR_AI       = (0, 255, 0)
COLOR_CV       = (255, 80, 0)
COLOR_FUSED    = (0, 255, 255)
COLOR_ZOOMED   = (255, 255, 0)
COLOR_POSSIBLE = (180, 100, 255)

# =============================================================================
# GRIPPER
# =============================================================================
# Bounding box around the gripper point (in camera coordinates)
# Size of the box: [-WIDTH/2, +WIDTH/2] on X-axis, [-HEIGHT/2, +HEIGHT/2] on Y-axis
GRIPPER_BB_WIDTH  = 500   # px — horizontal extent
GRIPPER_BB_HEIGHT = 500   # px — vertical extent

# How many consecutive frames a strawberry must be fully contained within
# the gripper bounding box before we auto-grip it
GRIPPER_CONTAINMENT_FRAMES = 3  # frames required for auto-grip

# Auto-grip enabled (if False, gripper stays manual-only)
GRIPPER_AUTO_GRIP_ENABLED = True

# =============================================================================
# GRIP DECISION
# =============================================================================

MIN_GRAB_AREA_RATIO = 0.0

GRAB_CENTER_TOLERANCE_X = 15
GRAB_CENTER_TOLERANCE_Y = 15
