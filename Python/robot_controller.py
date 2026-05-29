import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from detection import Detection

import config


X_THRESHOLD     = 25
Y_THRESHOLD     = 25
PRIORITIZE_Y    = True
LOCK_HYSTERESIS = 100  # new target must be this many pixels closer to gripper to steal the lock

# =========================
# DEPTH ESTIMATION CONFIG
# =========================
# Depth score is 1.0 at ideal berry size, falls off in both directions.
# Tune DEPTH_FALLOFF to control how steeply score drops with size deviation.
DEPTH_FALLOFF  = 1.5   # higher = more aggressive falloff from ideal
DEPTH_CONF_WEIGHT = 0.2  # how much detection confidence softens the score (0 = ignore)


@dataclass
class RobotTarget:
    detection: Detection
    center_x:  int
    center_y:  int
    distance:  float
    depth_score: float = 0.0   # 0.0 = far/unknown, 1.0 = at ideal picking distance


class RobotController:
    """Select target strawberries and produce simple movement commands."""

    def __init__(self) -> None:
        self.current_target: Optional[RobotTarget] = None

    @staticmethod
    def get_box_center(det: Detection) -> Tuple[int, int]:
        return (det.x1 + det.x2) // 2, (det.y1 + det.y2) // 2

    @staticmethod
    def _distance_to(gripper_x: int, gripper_y: int, x: int, y: int) -> float:
        return math.hypot(x - gripper_x, y - gripper_y)

    @staticmethod
    def estimate_depth(det: Detection) -> float:
        """
        Estimate relative depth from apparent bounding-box size.

        Returns a depth_score in [0.0, 1.0]:
            1.0  →  berry is close (at or above ideal size)
            0.0  →  berry is far away (much smaller than ideal)

        The curve is intentionally asymmetric:
            - Smaller than ideal → score drops toward 0 (far)
            - Larger than ideal  → score stays near 1.0 (close, just very close)
            - At ideal           → score = 1.0

        This means you can actually distinguish "too small = far" from
        "too large = very close", unlike a symmetric Gaussian which maps
        both to the same score.
        """
        area = max(1, (det.x2 - det.x1) * (det.y2 - det.y1))
        ideal = max(1, config.BERRY_SIZE_IDEAL)

        ratio = area / ideal  # <1.0 = smaller than ideal (far), >1.0 = larger (close)

        if ratio >= 1.0:
            # Berry is at or above ideal size → close.
            # Gentle falloff for extremely large boxes (noise / partial occlusion).
            raw_score = math.exp(-((ratio - 1.0) ** 2) / (2 * 1.5 ** 2))
        else:
            # Berry is smaller than ideal → farther away.
            # Steeper falloff: half-ideal size (ratio=0.5) scores ~0.57,
            # quarter-ideal (ratio=0.25) scores ~0.10.
            raw_score = math.exp(-((ratio - 1.0) ** 2) / (2 * 0.35 ** 2))

        conf = float(det.confidence or 0.0)
        score = raw_score * (1.0 - DEPTH_CONF_WEIGHT) + raw_score * conf * DEPTH_CONF_WEIGHT

        return round(min(1.0, max(0.0, score)), 3)

    @staticmethod
    def depth_label(depth_score: float) -> str:
        """Human-readable depth bucket for logging."""
        if depth_score >= 0.80:
            return "CLOSE"
        if depth_score >= 0.50:
            return "MEDIUM"
        if depth_score >= 0.25:
            return "FAR"
        return "VERY FAR"

    def _target_still_exists(self, detections: List[Detection]) -> bool:
        if self.current_target is None:
            return False

        target = self.current_target.detection
        for det in detections:
            if self._simple_iou(target, det) > 0.3:
                cx, cy = self.get_box_center(det)
                self.current_target.detection  = det
                self.current_target.center_x   = cx
                self.current_target.center_y   = cy
                self.current_target.distance   = 0.0
                self.current_target.depth_score = self.estimate_depth(det)
                return True
        return False

    @staticmethod
    def _simple_iou(a: Detection, b: Detection) -> float:
        x_a = max(a.x1, b.x1)
        y_a = max(a.y1, b.y1)
        x_b = min(a.x2, b.x2)
        y_b = min(a.y2, b.y2)
        inter  = max(0, x_b - x_a) * max(0, y_b - y_a)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def choose_target(self, detections: List[Detection], gripper_x: int, gripper_y: int) -> Optional[RobotTarget]:
        if not detections:
            self.current_target = None
            return None

        # Find the closest candidate across all detections
        closest: Optional[RobotTarget] = None
        for det in detections:
            cx, cy  = self.get_box_center(det)
            dist    = self._distance_to(gripper_x, gripper_y, cx, cy)
            depth   = self.estimate_depth(det)
            candidate = RobotTarget(det, cx, cy, dist, depth)
            if closest is None or dist < closest.distance:
                closest = candidate

        # If current target still exists in the frame, keep it unless the
        # best candidate is significantly closer (beats hysteresis threshold).
        if self._target_still_exists(detections):
            current_dist = self._distance_to(
                gripper_x, gripper_y,
                self.current_target.center_x,
                self.current_target.center_y
            )
            if closest.distance < current_dist - LOCK_HYSTERESIS:
                self.current_target = closest
            return self.current_target

        # No existing target — take the closest
        self.current_target = closest
        return self.current_target

    def generate_dx(self, gripper_x: int) -> int:
        if self.current_target is None:
            return 0
        return self.current_target.center_x - gripper_x

    def generate_dy(self, gripper_y: int) -> int:
        if self.current_target is None:
            return 0
        return self.current_target.center_y - gripper_y

    def generate_movementstring(self, gripper_x: int, gripper_y: int) -> str:
        if self.current_target is None:
            return "NO TARGET"

        tx = self.current_target.center_x
        ty = self.current_target.center_y

        dx = tx - gripper_x
        dy = ty - gripper_y

        if PRIORITIZE_Y:
            if abs(dy) > Y_THRESHOLD:
                return "ARM GO DOWN" if dy > 0 else "ARM GO UP"
        if abs(dx) > X_THRESHOLD:
            return "MOVE RIGHT" if dx > 0 else "MOVE LEFT"

        return "TARGET LOCKED"

    def generate_depthstring(self) -> str:
        """One-liner depth summary for the current target, ready to log."""
        if self.current_target is None:
            return "DEPTH: no target"
        t = self.current_target
        area = (t.detection.x2 - t.detection.x1) * (t.detection.y2 - t.detection.y1)
        return (
            f"DEPTH: {self.depth_label(t.depth_score)} "
            f"(score={t.depth_score:.2f}, area={area}px²)"
        )