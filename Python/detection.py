"""Detection pipelines: YOLO AI and contour-based OpenCV."""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

import config


@dataclass
class Detection:
    """Unified detection object from either pipeline."""
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float
    source: str  # "ai", "cv", "ai_zoomed", "cv_zoomed", "fused"
    label: Optional[str] = "strawberry"


class AIDetector:
    """YOLO-based detector."""

    def __init__(self, model_path: str = config.MODEL_PATH):
        self.model = YOLO(model_path)
        print(f"Loaded AI model")

    def detect(self, frame: np.ndarray, conf_threshold: float = None) -> List[Detection]:
        """Run YOLO detection. Returns list of Detection objects."""
        if conf_threshold is None:
            conf_threshold = config.YOLO_BASE_THRESHOLD

        results = self.model(frame, conf=conf_threshold, verbose=False)
        detections = []

        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf, source="ai"
                ))

        return detections


# ---------------------------------------------------------------------------
# Strawberry-specific HSV gate applied on top of the red ranges in config.
# Rejects pale, dark, and low-saturation reds that are not strawberry-like.
# ---------------------------------------------------------------------------
_SAT_MIN   = 80    # reject washed-out / pastel reds
_VAL_MIN   = 50    # reject very dark reds / shadows
_VAL_MAX   = 240   # reject near-white highlights

# Circularity floor at the contour stage — drops thin/elongated blobs early.
# Strawberries are roughly circular; wires, clothing seams, etc. are not.
_CONTOUR_MIN_CIRCULARITY = 0.55

# Aspect-ratio gate: width/height (or height/width) must be within this.
# Rejects very elongated objects before scoring.
_MAX_ASPECT_RATIO = 1.6

# Watershed: minimum fraction of the bounding-box diagonal that a foreground
# peak must be from the edge to count as a distinct berry centre.
_WATERSHED_FG_THRESH = 0.35   # higher → fewer splits (less over-splitting)

# NMS IoU threshold for final deduplication.
_NMS_IOU_THRESHOLD = 0.35

# Reuse tiny kernels / thresholds instead of recreating them every frame.
_KERNEL3 = np.ones((3, 3), np.uint8)
_SV_LOWER = np.array([0, _SAT_MIN, _VAL_MIN])
_SV_UPPER = np.array([180, 255, _VAL_MAX])


class CVDectector:
    """
    OpenCV contour + watershed detector.

    Improvements over the previous version
    ---------------------------------------
    1. Saturation + value gate on the red mask so pale / dark non-berry
       reds are rejected at the very first stage.
    2. Per-contour circularity and aspect-ratio pre-filter so elongated
       blobs never reach splitting or scoring.
    3. Watershed-based cluster splitting instead of convexity-defect
       heuristics — produces at most one box per berry centre, no more.
    4. IoU-based NMS (non-maximum suppression) as the final deduplication
       step, replacing the iterative overlap-merge which was creating
       duplicate boxes.
    5. The CV confidence score now requires a higher redness *and* a
       non-trivial circularity before a detection is accepted, so vaguely
       red blobs with poor shape are filtered out automatically.
    """

    def __init__(self):
        pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_red_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (hsv, binary_mask) of strawberry-red regions."""
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

        # Hue-range masks (from config)
        hue_mask = cv2.bitwise_or(
            cv2.inRange(hsv, config.RED_LOWER1, config.RED_UPPER1),
            cv2.inRange(hsv, config.RED_LOWER2, config.RED_UPPER2),
        )

        # Saturation / value gate: strawberries are vivid and not pitch-dark
        sv_mask = cv2.inRange(hsv, _SV_LOWER, _SV_UPPER)
        mask = cv2.bitwise_and(hue_mask, sv_mask)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  _KERNEL3, iterations=config.MORPH_OPEN_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, _KERNEL3, iterations=config.MORPH_CLOSE_ITER)

        # Adaptive close scaled to median blob radius (kept from original)
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            median_r = float(np.sqrt(np.median(areas) / np.pi))
            k = int(np.clip(median_r * 0.20, 3, 31))
            k = k if k % 2 == 1 else k + 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((k, k), np.uint8), iterations=2)
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
        """Returns the longer side divided by the shorter side (≥ 1)."""
        _, _, w, h = cv2.boundingRect(cnt)
        if h == 0 or w == 0:
            return 999.0
        return max(w, h) / max(min(w, h), 1)

    def _split_by_watershed(
        self,
        full_mask: np.ndarray,
        contour: np.ndarray,
    ) -> List[Tuple[int, int, int, int]]:
        """
        Split a potentially multi-berry contour into individual bounding boxes
        using distance-transform watershed.

        Returns one box per detected berry centre, or the whole bounding rect
        if no clean split is found.
        """
        x, y, w, h = cv2.boundingRect(contour)
        roi = full_mask[y:y + h, x:x + w].copy()

        if roi.size == 0:
            return [(x, y, x + w, y + h)]

        # Distance transform
        dist = cv2.distanceTransform(roi, cv2.DIST_L2, 5)
        if dist.max() == 0:
            return [(x, y, x + w, y + h)]
        dist_norm = dist / dist.max()

        # Foreground = definite berry centres (high distance from edge)
        fg = (dist_norm >= _WATERSHED_FG_THRESH).astype(np.uint8)
        n_labels, markers = cv2.connectedComponents(fg)

        if n_labels <= 1:
            # No foreground peaks found → single berry
            return [(x, y, x + w, y + h)]

        # Dilate fg slightly to build unknown border
        sure_bg = cv2.dilate(roi, np.ones((3, 3), np.uint8), iterations=3)
        unknown = cv2.subtract(sure_bg, fg)

        markers = markers + 1           # background becomes label 1
        markers[unknown == 255] = 0     # border region is unknown

        # Watershed needs a 3-channel image; use a mock BGR from the mask
        mock_bgr = cv2.cvtColor(roi, cv2.COLOR_GRAY2BGR)
        cv2.watershed(mock_bgr, markers)

        boxes: List[Tuple[int, int, int, int]] = []
        min_area = max(100, config.BERRY_SIZE_MIN // 4)  # lenient for distance

        for label in range(2, n_labels + 1):  # skip bg (1) and border (-1)
            region = np.zeros_like(roi)
            region[markers == label] = 255
            cnts, _ = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            rx, ry, rw, rh = cv2.boundingRect(cnts[0])
            if rw * rh >= min_area:
                # Add a small padding so tight watershed regions aren't too small
                pad = max(2, int(min(rw, rh) * 0.10))
                bx1 = max(0, rx - pad)
                by1 = max(0, ry - pad)
                bx2 = min(w, rx + rw + pad)
                by2 = min(h, ry + rh + pad)
                boxes.append((x + bx1, y + by1, x + bx2, y + by2))

        return boxes if boxes else [(x, y, x + w, y + h)]

    @staticmethod
    def _nms(
        boxes: List[Tuple[int, int, int, int]],
        scores: List[float],
        iou_threshold: float = _NMS_IOU_THRESHOLD,
    ) -> List[int]:
        """
        Standard IoU-based NMS. Returns indices of kept boxes, sorted by score.
        """
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
                ix = max(0, min(bx2, ox2) - max(bx1, ox1))
                iy = max(0, min(by2, oy2) - max(by1, oy1))
                inter = ix * iy
                if inter == 0:
                    filtered.append(idx)
                    continue
                o_area = max(0, ox2 - ox1) * max(0, oy2 - oy1)
                union = b_area + o_area - inter
                if union <= 0 or (inter / union) < iou_threshold:
                    filtered.append(idx)
            order = filtered

        return kept

    # ------------------------------------------------------------------
    # Public scoring + detection
    # ------------------------------------------------------------------

    def cv_score_crop(self, frame: np.ndarray, box: Tuple[int, int, int, int]) -> Dict:
        """
        Score a single crop: redness, circularity, size, texture → weighted total.
        Returns dict with individual scores and 'total'.
        """
        h_fr, w_fr = frame.shape[:2]
        x1 = max(0, box[0])
        y1 = max(0, box[1])
        x2 = min(w_fr, box[2])
        y2 = min(h_fr, box[3])
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return {"redness": 0.0, "circularity": 0.0, "size": 0.0,
                    "texture": 0.0, "total": 0.0}

        # Redness score (with saturation gate applied to crop as well).
        # Denominator is 20% of total pixels — a well-cropped berry commonly
        # covers 25-50% of its bounding box, so this gives realistic scores.
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, config.RED_LOWER1, config.RED_UPPER1)
        red2 = cv2.inRange(hsv, config.RED_LOWER2, config.RED_UPPER2)
        red_hue = cv2.bitwise_or(red1, red2)
        sv_gate = cv2.inRange(hsv, _SV_LOWER, _SV_UPPER)
        red_mask = cv2.bitwise_and(red_hue, sv_gate)
        total_pixels = crop.shape[0] * crop.shape[1]
        redness = min(1.0, cv2.countNonZero(red_mask) / max(total_pixels * 0.20, 1))

        # Circularity score.
        # No hard zero gate here — low circularity just contributes a low
        # weighted term.  The contour pre-filter in detect() already removed
        # clearly non-circular blobs before we get to scoring.
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circularity = min(1.0, (4 * np.pi * area) / (peri ** 2))

            # HARD GATE: If it fails the roundness requirement,
            # it's a poster block or a background artifact. Nuke it entirely.
            if circularity < 0.20:
                return {"redness": 0, "circularity": 0, "size": 0,
                        "texture": 0, "total": 0.0}

        # Size score (adaptive — penalise both too-small and too-large)
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

        # Texture score (Laplacian variance — berries have seedy texture)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture = float(np.clip(lap_var / 500.0, 0.0, 1.0))

        total = (
            config.CV_WEIGHT_REDNESS     * redness +
            config.CV_WEIGHT_CIRCULARITY * circularity +
            config.CV_WEIGHT_SIZE        * size_score +
            config.CV_WEIGHT_TEXTURE     * texture
        )

        return {
            "redness":     round(redness,     3),
            "circularity": round(circularity, 3),
            "size":        round(size_score,  3),
            "texture":     round(texture,     3),
            "total":       round(total,       3),
        }

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """
        Run CV contour detection.

        Pipeline
        --------
        1. Build red mask with saturation/value gate.
        2. Find external contours; reject tiny, non-circular, and elongated ones.
        3. Split large multi-berry blobs with distance-transform watershed.
        4. Score each candidate box.
        5. Run NMS to eliminate duplicate boxes.

        Returns
        -------
        (detections, mask)
        """
        _, mask = self._build_red_mask(frame)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes: List[Tuple[int, int, int, int]] = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CONTOUR_AREA:
                continue

            # Shape pre-filter — reject non-circular or elongated blobs
            if self._contour_circularity(cnt) < _CONTOUR_MIN_CIRCULARITY:
                continue
            if self._contour_aspect_ratio(cnt) > _MAX_ASPECT_RATIO:
                continue

            if area >= config.CONVEXITY_MIN_AREA:
                # Potentially a cluster — try watershed split
                raw_boxes.extend(self._split_by_watershed(mask, cnt))
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                raw_boxes.append((x, y, x + w, y + h))

        if not raw_boxes:
            return [], mask

        # Score every candidate box
        scores: List[float] = []
        for box in raw_boxes:
            s = self.cv_score_crop(frame, box)
            scores.append(s["total"])

        # NMS: removes duplicate boxes around the same berry
        kept_indices = self._nms(raw_boxes, scores)

        # Build Detection objects, applying the acceptance threshold
        min_confidence = config.CV_BASE_THRESHOLD
        detections: List[Detection] = []
        for idx in kept_indices:
            conf = scores[idx]
            if conf >= min_confidence:
                x1, y1, x2, y2 = raw_boxes[idx]
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=conf, source="cv",
                ))

        return detections, mask


def iou(box1: Detection, box2: Detection) -> float:
    """IoU between two Detection objects."""
    x1 = max(box1.x1, box2.x1)
    y1 = max(box1.y1, box2.y1)
    x2 = min(box1.x2, box2.x2)
    y2 = min(box1.y2, box2.y2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection == 0:
        return 0.0

    area1 = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
    area2 = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0