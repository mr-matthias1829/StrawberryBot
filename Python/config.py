# =============================================================================
# config.py — single source of truth for all thresholds and weights
# Swap V6 model in by changing MODEL_PATH. Tune thresholds here only.
# =============================================================================

import os

# ── Model ─────────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "runs", "detect", "superv2", "weights", "best.pt")

# ── YOLO ──────────────────────────────────────────────────────────────────────
YOLO_CONF_THRESHOLD  = 0.7    # minimum confidence to even consider a box
YOLO_ZOOM_THRESHOLD  = 0.95   # above this: accept without CV check
YOLO_UNSURE_LOW      = 0.60   # below YOLO_CONF_THRESHOLD but worth a zoom recheck
YOLO_UNSURE_HIGH     = 0.9   # upper bound of the "unsure" zone
MAX_RECHECKS         = 2      # zoom-and-recheck recursion cap

# ── OpenCV confidence scorer weights (must sum to 1.0) ───────────────────────
CV_WEIGHT_REDNESS      = 0.35
CV_WEIGHT_CIRCULARITY  = 0.25
CV_WEIGHT_SIZE         = 0.20
CV_WEIGHT_TEXTURE      = 0.15
CV_WEIGHT_TEMPORAL     = 0.05

CV_CONF_THRESHOLD      = 0.7  # minimum CV score to "pass" the sanity check

# ── Expected berry size (pixels²) — tune for your camera distance ─────────────
BERRY_SIZE_MIN =   200
BERRY_SIZE_MAX = 40_000
BERRY_SIZE_IDEAL = 8_000   # score peaks here

# ── Fusion ────────────────────────────────────────────────────────────────────
FUSION_YOLO_WEIGHT = 0.7   # weight of YOLO conf in final fused score
FUSION_CV_WEIGHT   = 0.3   # weight of CV conf in final fused score
FUSION_THRESHOLD   = 0.55  # minimum fused score to confirm a berry

# ── IoU matching ──────────────────────────────────────────────────────────────
IOU_MATCH_THRESHOLD = 0.40  # boxes with IoU above this are "the same berry"

# ── Temporal tracker ──────────────────────────────────────────────────────────
TRACKER_CONFIRM_COUNT = 3   # frames a box must persist before CONFIRMED
TRACKER_MISS_COUNT    = 5   # frames missing before track is dropped
TRACKER_IOU_LINK      = 0.30  # IoU to link a new box to an existing track

# ── CV pipeline — HSV red range ───────────────────────────────────────────────
RED_LOWER1 = (  0, 100, 135)
RED_UPPER1 = ( 10, 255, 255)
RED_LOWER2 = (170, 100, 135)
RED_UPPER2 = (179, 255, 255)

# ── CV pipeline — morphology ──────────────────────────────────────────────────
MORPH_OPEN_ITER  = 3
MORPH_CLOSE_ITER = 5

# ── CV pipeline — watershed tuning ───────────────────────────────────────────
PRIMARY_PEAK_KERNEL       = 50
PRIMARY_PEAK_THRESHOLD    = 12
TRANSITION_PERCENTILE     = 80.0
TRANSITION_PEAK_KERNEL    = 250
TRANSITION_PEAK_THRESHOLD = 15
FALLBACK_BLOB_MIN_AREA    = 10_000
FALLBACK_PEAK_KERNEL      = 11
FALLBACK_PEAK_THRESHOLD   = 5
SEED_DILATION             = 25
MERGE_OVERLAP_RATIO       = 0.45
MIN_BOX_AREA              = 800
ADAPTIVE_SCALE            = 0.55