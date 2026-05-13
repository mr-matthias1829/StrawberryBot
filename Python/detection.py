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


class CVDectector:
    """OpenCV contour + convexity-defect detector."""

    def __init__(self):
        pass

    def _build_red_mask(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (hsv, binary_mask) of red regions."""
        blurred = cv2.GaussianBlur(frame, (7, 7), 0)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, config.RED_LOWER1, config.RED_UPPER1),
            cv2.inRange(hsv, config.RED_LOWER2, config.RED_UPPER2),
        )
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=config.MORPH_OPEN_ITER)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=config.MORPH_CLOSE_ITER)

        # Adaptive close: scale kernel to median blob radius
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            median_r = float(np.sqrt(np.median(areas) / np.pi))
            k = int(np.clip(median_r * 0.20, 3, 31))
            k = k if k % 2 == 1 else k + 1
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                    np.ones((k, k), np.uint8), iterations=2)
        return hsv, mask

    def _split_cluster(self, contour: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Split convex blob into individual berry bounding boxes using convexity defects."""
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = None
        try:
            defects = cv2.convexityDefects(contour, hull)
        except cv2.error:
            pass

        if defects is None or len(defects) < 2:
            x, y, w, h = cv2.boundingRect(contour)
            return [(x, y, x + w, y + h)]

        # Find deep defects (gaps between berries)
        deep = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            depth = d / 256.0
            if depth > 8:  # px depth threshold
                deep.append(tuple(contour[f][0]))

        if len(deep) < 2:
            x, y, w, h = cv2.boundingRect(contour)
            return [(x, y, x + w, y + h)]

        # Partition based on defect positions
        xs = [p[0] for p in deep]
        ys = [p[1] for p in deep]
        x, y, w, h = cv2.boundingRect(contour)

        # Decide split axis: vertical if defects spread more horizontally
        if max(xs) - min(xs) >= max(ys) - min(ys):
            split_xs = sorted(set(xs))
            boxes = []
            prev_x = x
            for sx in split_xs:
                if sx - prev_x > 10:
                    boxes.append((prev_x, y, sx, y + h))
                    prev_x = sx
            boxes.append((prev_x, y, x + w, y + h))
        else:
            split_ys = sorted(set(ys))
            boxes = []
            prev_y = y
            for sy in split_ys:
                if sy - prev_y > 10:
                    boxes.append((x, prev_y, x + w, sy))
                    prev_y = sy
            boxes.append((x, prev_y, x + w, y + h))

        # Filter by minimum area (adaptive: smaller for distant berries)
        min_area = max(100, config.BERRY_SIZE_MIN // 2)  # Adaptive: allow smaller
        return [(bx1, by1, bx2, by2) for bx1, by1, bx2, by2 in boxes
                if (bx2 - bx1) * (by2 - by1) >= min_area]

    def _merge_overlapping(self, boxes: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Iteratively merge boxes with high overlap."""
        def overlap(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix = max(0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0, min(ay2, by2) - max(ay1, by1))
            inter = ix * iy
            if inter == 0:
                return 0.0
            area_a = (ax2 - ax1) * (ay2 - ay1)
            area_b = (bx2 - bx1) * (by2 - by1)
            return inter / min(area_a, area_b)

        merged = True
        while merged:
            merged = False
            kept = []
            absorbed = [False] * len(boxes)
            for i, a in enumerate(boxes):
                if absorbed[i]:
                    continue
                cur = a
                for j, b in enumerate(boxes):
                    if i == j or absorbed[j]:
                        continue
                    if overlap(cur, b) >= config.MERGE_OVERLAP_RATIO:
                        cur = (min(cur[0], b[0]), min(cur[1], b[1]),
                               max(cur[2], b[2]), max(cur[3], b[3]))
                        absorbed[j] = True
                        merged = True
                kept.append(cur)
            boxes = kept
        return boxes

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

        # Redness score
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        red1 = cv2.inRange(hsv, config.RED_LOWER1, config.RED_UPPER1)
        red2 = cv2.inRange(hsv, config.RED_LOWER2, config.RED_UPPER2)
        red_mask = cv2.bitwise_or(red1, red2)
        total_pixels = crop.shape[0] * crop.shape[1]
        redness = min(1.0, cv2.countNonZero(red_mask) / max(total_pixels * 0.3, 1))

        # Circularity score
        contours, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circularity = 0.0
        if contours:
            cnt = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(cnt)
            peri = cv2.arcLength(cnt, True)
            if peri > 0:
                circularity = min(1.0, (4 * np.pi * area) / (peri ** 2))

        # Size score (adaptive - penalize both too small and too large)
        box_area = (x2 - x1) * (y2 - y1)
        if box_area <= config.BERRY_SIZE_MIN:
            size_score = box_area / max(config.BERRY_SIZE_MIN, 1)
        elif box_area >= config.BERRY_SIZE_MAX:
            size_score = config.BERRY_SIZE_MAX / box_area
        elif box_area <= config.BERRY_SIZE_IDEAL:
            size_score = (box_area - config.BERRY_SIZE_MIN) / max(config.BERRY_SIZE_IDEAL - config.BERRY_SIZE_MIN, 1)
        else:
            size_score = (config.BERRY_SIZE_MAX - box_area) / max(config.BERRY_SIZE_MAX - config.BERRY_SIZE_IDEAL, 1)
        size_score = float(np.clip(size_score, 0.0, 1.0))

        # Texture score (Laplacian variance - higher = more texture)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        texture = float(np.clip(lap_var / 500.0, 0.0, 1.0))

        # Weighted total
        total = (
            config.CV_WEIGHT_REDNESS * redness +
            config.CV_WEIGHT_CIRCULARITY * circularity +
            config.CV_WEIGHT_SIZE * size_score +
            config.CV_WEIGHT_TEXTURE * texture
        )

        return {
            "redness": round(redness, 3),
            "circularity": round(circularity, 3),
            "size": round(size_score, 3),
            "texture": round(texture, 3),
            "total": round(total, 3)
        }

    def detect(self, frame: np.ndarray) -> Tuple[List[Detection], np.ndarray]:
        """
        Run CV contour detection.

        Returns:
            (detections, mask)
        """
        hsv, mask = self._build_red_mask(frame)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        raw_boxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < config.MIN_CONTOUR_AREA:
                continue
            if area >= config.CONVEXITY_MIN_AREA:
                raw_boxes.extend(self._split_cluster(cnt))
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                raw_boxes.append((x, y, x + w, y + h))

        boxes = self._merge_overlapping(raw_boxes)

        # Convert to Detection objects with CV confidence score
        detections = []
        for x1, y1, x2, y2 in boxes:
            scores = self.cv_score_crop(frame, (x1, y1, x2, y2))
            confidence = scores["total"]

            # Only keep if above threshold (but threshold is low - fusion will decide)
            if confidence >= config.CV_BASE_THRESHOLD * 0.7:  # Lower threshold, fusion handles
                detections.append(Detection(
                    x1=x1, y1=y1, x2=x2, y2=y2,
                    confidence=confidence, source="cv"
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