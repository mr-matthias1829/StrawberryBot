"""configuration and constants for detection system."""

import os

import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "..", "runs", "detect", "CV", "weights", "best.pt")

# FUSION THRESHOLDS
YOLO_BASE_THRESHOLD = 0.50       # minimum YOLO conf to consider
CV_BASE_THRESHOLD = 0.50


# a well-lit berry with good shape reliably clears ~0.65
CV_DIRECT_ACCEPT_THRESHOLD = 0.65

HIGH_AI_CONFIDENCE = 0.85       # if YOLO > this, trust even without CV
LOW_AI_CONFIDENCE  = 0.40       # if YOLO < this, trigger zoom recheck

# fusion weights when both detectors agree on the same berry
YOLO_FUSION_WEIGHT = 0.65       # AI slightly preferred when both agree
CV_FUSION_WEIGHT   = 0.35


# CSV COLOUR RANGES
# if distant berries are being missed, try lowering sat to 80 and val cv1 to 100.
# covers both sides of the red spectrum
RED_LOWER1, RED_UPPER1 = np.array([0,   100, 135]), np.array([10,  255, 255])
RED_LOWER2, RED_UPPER2 = np.array([170, 100, 135]), np.array([179, 255, 255])

# morphology
MORPH_OPEN_ITER  = 3
MORPH_CLOSE_ITER = 5

# contour filtering
MIN_CONTOUR_AREA    = 150       # slightly lower than 200, distant berries are small
CONVEXITY_MIN_AREA  = 3000      # area at which watershed cluster-splitting kicks in
MERGE_OVERLAP_RATIO = 0.45      # legacy, kept for zoom-recheck path

# CV scoring weights: REDNESS + CIRCULARITY + SIZE + TEXTURE must sum to 1.0.
CV_WEIGHT_REDNESS     = 0.40 # this name is actually a lie: it's just weight of the color itself, whatever color its set to
CV_WEIGHT_CIRCULARITY = 0.40
CV_WEIGHT_SIZE        = 0.15 # penalises implausibly tiny or huge blobs. might break far/short distances
CV_WEIGHT_TEXTURE     = 0.05 # laplacian variance is a weak tiebreaker ONLY.
CV_WEIGHT_TEMPORAL    = 0.05 # no longer in use

# size scoring reference values (px2 in the inference frame)
# high range and values since it tracks pixels, not real world values like CM
BERRY_SIZE_IDEAL = 15_000         # ideal berry area
BERRY_SIZE_MIN   = 200            # below this, size score decays toward 0
BERRY_SIZE_MAX   = 500_000        # above this, size score decays toward 0

# zoom recheck
MAX_RECHECKS      = 2
ZOOM_SCALE_FACTOR = 2.0
RECHECK_AI_CONF   = 0.65
RECHECK_CV_CONF   = 0.50        # lowered from 0.55 — zoom crops can be tight/partial

# temporal memory
PERSISTENCE_REQUIRED        = 3 # higher = less flickering?
PERSISTENCE_REQUIRED_CV_ONLY = 4 # cv loves to flicker more than ai, so its a little higher
PERSISTENCE_DECAY           = 0.85
IOU_MATCH_THRESHOLD         = 0.30

# possible-hits
# generic fallback (fused / unknown source)
POSSIBLE_HIT_MIN_CONF = 0.50
POSSIBLE_HIT_MIN_SEEN = 1

POSSIBLE_CV_ONLY_MIN_CONF = 0.50
POSSIBLE_CV_ONLY_MIN_SEEN = 1

# AI-only possible: kept strict — AI false-positives are more common.
# when we get a better ai: adjust these values
POSSIBLE_AI_ONLY_MIN_CONF = 0.60
POSSIBLE_AI_ONLY_MIN_SEEN = 1
POSSIBLE_AI_CONF_WEIGHT   = 0.50   # down-weight AI conf for possible classification

# possible → target fallback
POSSIBLE_TARGET_FALLBACK_ENABLED = True
POSSIBLE_TARGET_MIN_CONF         = 0.50  # lowered from 0.60 to match CV_ONLY floor

# display
SHOW_DEBUG_WINDOWS = True # outdated and does nothing, always false now
COLOR_AI       = (0, 255, 0) #unused
COLOR_CV       = (255, 80, 0) #unused
COLOR_FUSED    = (0, 255, 255)
COLOR_ZOOMED   = (255, 255, 0) #unused
COLOR_POSSIBLE = (180, 100, 255)

# GRIPPER
# bounding box around the gripper point (in camera coordinates)
GRIPPER_BB_WIDTH  = 500   # px — horizontal extent
GRIPPER_BB_HEIGHT = 500   # px — vertical extent

# how many consecutive frames a strawberry must be fully contained within
# the gripper bounding box before we auto-grip it
GRIPPER_CONTAINMENT_FRAMES = 3  # frames required for auto-grip

# auto-grip enabled (if False, auto mode never grips shut when it otherwise would)
GRIPPER_AUTO_GRIP_ENABLED = True

# GRIPPER DECISION
MIN_GRAB_AREA_RATIO = 0.1

GRAB_CENTER_TOLERANCE_X = 15 #unused
GRAB_CENTER_TOLERANCE_Y = 15 #unused

# auto-mode control: when False, arm won't extend AND gripper won't auto-grip
AUTO_MODE_ALLOW_MOVE = True
