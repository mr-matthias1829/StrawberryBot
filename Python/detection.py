"""detection pipelines: YOLO AI and contour-based OpenCV.

Key changes vs. previous version
---------------------------------
* Every Detection carries a ``timestamp`` (time.monotonic() of the frame it
  was produced from).  FusionEngine uses this to gate stale detections.
* AIDetector.detect() accepts an optional ``timestamp`` kwarg.
* CVDetector.detect() accepts an optional ``timestamp`` kwarg.
* CV pipeline improvements:
  - Saturation + value gate applied before hue mask.
  - Per-contour circularity and aspect-ratio pre-filter (no non-circular
    blobs reach scoring or splitting).
  - Distance-transform watershed for cluster splitting (replaces convexity
    defects).
  - IoU NMS as final deduplication.
  - cv_score_crop() uses configurable CVConfig sampled thread-safely.
* CVConfig is a hot-swappable dataclass; web dashboard can update it live.
"""

import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
from config import CLASS_NAMES, CLASS_COLORS, TARGETABLE_CLASS_IDS

import config


# ---------------------------------------------------------------------------
# Detection dataclass
# ---------------------------------------------------------------------------

@dataclass
class Detection:
    """Unified detection object from either pipeline."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    source: str                  # "ai", "cv", "fused", "zoomed_*", …
    label: Optional[str] = "strawberry"
    class_id: int = 0
    timestamp: float = 0.0       # time.monotonic() of the originating frame

    def age(self) -> float:
        """Seconds since the frame this detection came from was captured."""
        return time.monotonic() - self.timestamp

    @property
    def is_targetable(self) -> bool:
        return self.class_id in TARGETABLE_CLASS_IDS

    @property
    def color(self) -> Tuple[int, int, int]:
        return CLASS_COLORS.get(self.class_id, (200, 200, 200))

    @property
    def class_name(self) -> str:
        return CLASS_NAMES.get(self.class_id, "unknown")

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2

    @property
    def center_x(self) -> int:
        return (self.x1 + self.x2) // 2

    @property
    def center_y(self) -> int:
        return (self.y1 + self.y2) // 2


# ---------------------------------------------------------------------------
# CVConfig — hot-swappable colour parameters
# ---------------------------------------------------------------------------

@dataclass
class CVConfig:
    """Runtime-editable parameters for the CV detector."""

    h1_low:  int = 0
    h1_high: int = 10
    h2_low:  int = 160
    h2_high: int = 179

    sat_min: int = 100   # raised from 80 — rejects washed-out colours better
    val_min: int = 50
    val_max: int = 240

    contour_min_circularity: float = 0.50   # relaxed slightly — distant berries are less round
    max_aspect_ratio:        float = 1.8    # wider tolerance for partial occlusion
    watershed_fg_thresh:     float = 0.35
    nms_iou_threshold:       float = 0.35

    def to_hsv_arrays(self):
        lower1 = np.array([self.h1_low,  self.sat_min, self.val_min])
        upper1 = np.array([self.h1_high, 255,          self.val_max])
        lower2 = np.array([self.h2_low,  self.sat_min, self.val_min])
        upper2 = np.array([self.h2_high, 255,          self.val_max])
        return lower1, upper1, lower2, upper2

    def sv_bounds(self):
        return (
            np.array([0,   self.sat_min, self.val_min]),
            np.array([180, 255,          self.val_max]),
        )

    def to_dict(self) -> Dict:
        return {
            "h1_low":  self.h1_low,  "h1_high": self.h1_high,
            "h2_low":  self.h2_low,  "h2_high": self.h2_high,
            "sat_min": self.sat_min,
            "val_min": self.val_min, "val_max":  self.val_max,
            "contour_min_circularity": self.contour_min_circularity,
            "max_aspect_ratio":        self.max_aspect_ratio,
            "watershed_fg_thresh":     self.watershed_fg_thresh,
            "nms_iou_threshold":       self.nms_iou_threshold,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "CVConfig":
        obj = cls()
        int_keys   = {"h1_low", "h1_high", "h2_low", "h2_high", "sat_min", "val_min", "val_max"}
        float_keys = {"contour_min_circularity", "max_aspect_ratio",
                      "watershed_fg_thresh", "nms_iou_threshold"}
        for k, v in d.items():
            if k in int_keys:
                setattr(obj, k, int(np.clip(int(v), 0, 255)))
            elif k in float_keys:
                setattr(obj, k, float(v))
        return obj


_cv_config_lock = threading.Lock()
_cv_config = CVConfig(
    h1_low  = int(config.RED_LOWER1[0]),
    h1_high = int(config.RED_UPPER1[0]),
    h2_low  = int(config.RED_LOWER2[0]),
    h2_high = int(config.RED_UPPER2[0]),
)


def get_cv_config() -> CVConfig:
    with _cv_config_lock:
        import copy
        return copy.copy(_cv_config)


def set_cv_config(new_cfg: CVConfig) -> None:
    global _cv_config
    with _cv_config_lock:
        _cv_config = new_cfg


# ---------------------------------------------------------------------------
# AI Detector
# ---------------------------------------------------------------------------

class AIDetector:
    """YOLO-based multi-class detector."""

    def __init__(self, model_path: str = config.MODEL_PATH) -> None:
        self.model = YOLO(model_path)
        print(f"Loaded AI model: {model_path}  (classes: {CLASS_NAMES})")

    def detect(
        self,
        frame: np.ndarray,
        conf_threshold: float = None,
        timestamp: float = 0.0,
    ) -> List[Detection]:
        if conf_threshold is None:
            conf_threshold = config.YOLO_BASE_THRESHOLD

        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections: List[Detection] = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf     = float(box.conf[0])
                class_id = int(box.cls[0]) if box.cls is not None else 0
                label    = CLASS_NAMES.get(class_id, "unknown")
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    source="ai",
                    label=label,
                    class_id=class_id,
                    timestamp=timestamp,
                ))

        return detections


# ---------------------------------------------------------------------------
# CV Detector
# ---------------------------------------------------------------------------

_KERNEL3 = np.ones((3, 3), np.uint8)


class CVDectector:
    """
    OpenCV contour + watershed detector.

    All detections are class_id=0 (strawberry) — CV is purely colour-based.

    Pipeline
    --------
    1. Snapshot CVConfig (thread-safe)
    2. Gaussian blur → HSV → hue + saturation/value gate → morphology
    3. Find external contours; reject tiny, non-circular, and elongated blobs
    4. Split large multi-berry clusters with distance-transform watershed
    5. Score each candidate box (redness, circularity, size, texture)
    6. IoU-based NMS
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_red_mask(
        self, frame: np.ndarray, cfg: CVConfig
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (hsv, binary_mask)."""
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv     = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        lower1, upper1, lower2, upper2 = cfg.to_hsv_arrays()
        sv_lower, sv_upper             = cfg.sv_bounds()

        hue_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2),
        )
        sv_mask = cv2.inRange(hsv, sv_lower, sv_upper)
        mask    = cv2.bitwise_and(hue_mask, sv_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL3, iterations=config.MORPH_OPEN_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL3, iterations=config.MORPH_CLOSE_ITER)

        # Adaptive close scaled to median blob radius
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            areas    = stats[1:, cv2.CC_STAT_AREA]
            median_r = float(np.sqrt(np.median(areas) / np.pi))
            k = int(np.clip(median_r * 0.20, 3, 31))
            k = k if k % 2 == 1 else k + 1
            mask = cv2.morphologyEx(
                mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2
            )

        return hsv, mask

    @staticmethod
    def _contour_circularity(cnt: np.ndarray) -> float:
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt, True)
        if peri == 0:
            return 0.0
        return float(4.0 * np.pi * area / (peri ** 2))

    @staticmethod
    def _contour_aspect_ratio(cnt: np.ndarray) -> float:
        _, _, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0:
            return 999.0
        return max(w, h) / max(min(w, h), 1)

    def _split_by_watershed(
        self,
        full_mask: np.ndarray,
        contour: np.ndarray,
        fg_thresh: float,
    ) -> List[Tuple[int, int, int, int]]:
        x, y, w, h = cv2.boundingRect(contour)
        roi = full_mask[y:y + h, x:x + w].copy()
        if roi.size == 0:
            return [(x, y, x + w, y + h)]

        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return [(x, y, x + w, y + h)]
        dist_norm = dist / dist.max()

        fg      = (dist_norm >= fg_thresh).astype(np.uint8)
        n_labels, markers = cv2.connectedComponents(fg)
        if n_labels <= 1:
            return [(x, y, x + w, y + h)]

        sure_bg = cv2.dilate(roi, np.ones((3, 3), np.uint8), iterations=3)
        unknown = cv2.subtract(sure_bg, fg)
        markers = markers + 1
        markers[unknown == 255] = 0

        mock_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        cv2.watershed(mock_bgr, markers)

        boxes: List[Tuple[int, int, int, int]] = []
        min_area = max(100, config.BERRY_SIZE_MIN // 4)
        for label in range(2, n_labels + 1):
            region = np.zeros_like(roi)
            region[markers == label] = 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnts[0])
            if rw * rh >= min_area:
                pad  = max(2, int(min(rw, rh) * 0.10))
                bx1  = max(0, rx - pad)
                by1  = max(0, ry - pad)
                bx2  = min(w, rx + rw + pad)
                by2  = min(h, ry + rh + pad)
                boxes.append((x + bx1, y + by1, x + bx2, y + by2))

        return boxes if boxes else [(x, y, x + w, y + h)]

    @staticmethod
    def _nms(
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        iou_threshold: float,
    ) -> List[int]:
        if not boxes:
            return []
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        kept: List[int] = []
        while order:
            best = order.pop(0)
            kept.append(best)
            bx1, by1, bx2, by2 = boxes[best]
            b_area = max(0, bx2 - bx1) * max(0, by2 - by1)
            filtered = []
            for idx in order:
                ox1, oy1, ox2, oy2 = boxes[idx]
                ix  = max(0, min(bx2, ox2) - max(bx1, ox1))
                iy  = max(0, min(by2, oy2) - max(by1, oy1))
                inter = ix * iy
                if inter == 0:
                    filtered.append(idx)
                    continue
                o_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
                union  = b_area + o_area - inter
                if union <= 0 or (inter / union) < iou_threshold:
                    filtered.append(idx)
            order = filtered
        return kept

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cv_score_crop(
        self, frame: np.ndarray, box: Tuple[int, int, int, int]
    ) -> Dict:
        """Score a single crop: redness, circularity, size, texture → weighted total."""
        cfg = get_cv_config()

        h_fr, w_fr = frame.shape[:2]
        x1 = max(0, box[0]);  y1 = max(0, box[1])
        x2 = min(w_fr, box[2]); y2 = min(h_fr, box[3])
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {"redness": 0.0, "circularity": 0.0, "size": 0.0,
                    "texture": 0.0, "total": 0.0}

        lower1, upper1, lower2, upper2 = cfg.to_hsv_arrays()
        sv_lower, sv_upper             = cfg.sv_bounds()

        hsv      = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hue_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower1, upper1),
            cv2.inRange(hsv, lower2, upper2),
        )
        sv_gate  = cv2.inRange(hsv, sv_lower, sv_upper)
        red_mask = cv2.bitwise_and(hue_mask, sv_gate)

        total_pixels = crop.shape[0] * crop.shape[1]
        redness = min(1.0, cv2.countNonZero(red_mask) / max(total_pixels * 0.20, 1))

        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        if contours:
            cnt  = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circularity = min(1.0, (4 * np.pi * area) / (peri ** 2))
            if circularity < 0.18:   # hard gate — almost certainly not a berry
                return {"redness": 0, "circularity": 0, "size": 0,
                        "texture": 0, "total": 0.0}

        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= config.BERRY_SIZE_MIN:
            size_score = box_area / max(config.BERRY_SIZE_MIN, 1)
        elif box_area >= config.BERRY_SIZE_MAX:
            size_score = config.BERRY_SIZE_MAX / box_area
        elif box_area <= config.BERRY_SIZE_IDEAL:
            size_score = (box_area - config.BERRY_SIZE_MIN) / max(
                config.BERRY_SIZE_IDEAL - config.BERRY_SIZE_MIN, 1)
        else:
            size_score = (config.BERRY_SIZE_MAX - box_area) / max(
                config.BERRY_SIZE_MAX - config.BERRY_SIZE_IDEAL, 1)
        size_score = float(np.clip(size_score, 0.0, 1.0))

        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture = float(np.clip(lap_var / 500.0, 0.0, 1.0))

        total = (
            config.CV_WEIGHT_REDNESS     * redness
            + config.CV_WEIGHT_CIRCULARITY * circularity
            + config.CV_WEIGHT_SIZE        * size_score
            + config.CV_WEIGHT_TEXTURE     * texture
        )
        return {
            "redness":     round(redness,     3),
            "circularity": round(circularity, 3),
            "size":        round(size_score,  3),
            "texture":     round(texture,     3),
            "total":       round(total,       3),
        }

    def detect(
        self,
        frame: np.ndarray,
        timestamp: float = 0.0,
    ) -> Tuple[List[Detection], np.ndarray]:
        """Run CV contour detection. Returns (detections, mask)."""
        cfg = get_cv_config()
        _, mask = self._build_red_mask(frame, cfg)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_boxes: List[Tuple[int, int, int, int]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CONTOUR_AREA:
                continue
            if self._contour_circularity(cnt) < cfg.contour_min_circularity:
                continue
            if self._contour_aspect_ratio(cnt) > cfg.max_aspect_ratio:
                continue

            if area >= config.CONVEXITY_MIN_AREA:
                raw_boxes.extend(self._split_by_watershed(mask, cnt, cfg.watershed_fg_thresh))
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                raw_boxes.append((x, y, x + w, y + h))

        if not raw_boxes:
            return [], mask

        scores      = [self.cv_score_crop(frame, box)["total"] for box in raw_boxes]
        kept_idx    = self._nms(raw_boxes, scores, cfg.nms_iou_threshold)

        detections: List[Detection] = []
        for idx in kept_idx:
            conf = scores[idx]
            if conf >= config.CV_BASE_THRESHOLD:
                x1, y1, x2, y2 = raw_boxes[idx]
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf,
                    source="cv",
                    label="Strawberry",
                    class_id=0,
                    timestamp=timestamp,
                ))

        return detections, mask


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def iou(box1: Detection, box2: Detection) -> float:
    """Intersection-over-Union between two Detection objects."""
    x1 = max(box1.x1, box2.x1);  y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2);  y2 = min(box1.y2, box2.y2)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0
    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0