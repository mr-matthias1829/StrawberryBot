# robot_controller.py

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from detection import Detection


# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# How close the strawberry center must be to the gripper center
# before movement stops.
X_THRESHOLD = 25
Y_THRESHOLD = 25

# Y has priority over X
PRIORITIZE_Y = True


# ─────────────────────────────────────────────────────────────
# DATA TYPES
# ─────────────────────────────────────────────────────────────

@dataclass
class RobotTarget:
    detection: Detection
    center_x: int
    center_y: int
    distance: float


# ─────────────────────────────────────────────────────────────
# CONTROLLER
# ─────────────────────────────────────────────────────────────

class RobotController:
    """
    Handles:
    - selecting target strawberry
    - keeping lock on target
    - generating movement instructions
    """

    def __init__(self) -> None:
        self.current_target: Optional[RobotTarget] = None

    # ─────────────────────────────────────────────────────────

    @staticmethod
    def get_box_center(det: Detection) -> Tuple[int, int]:
        cx = (det.x1 + det.x2) // 2
        cy = (det.y1 + det.y2) // 2
        return cx, cy

    # ─────────────────────────────────────────────────────────

    def _distance_to_gripper(
        self,
        det: Detection,
        gripper_x: int,
        gripper_y: int
    ) -> float:
        cx, cy = self.get_box_center(det)

        return math.sqrt(
            (cx - gripper_x) ** 2 +
            (cy - gripper_y) ** 2
        )

    # ─────────────────────────────────────────────────────────

    def _target_still_exists(
        self,
        detections: List[Detection]
    ) -> bool:

        if self.current_target is None:
            return False

        target = self.current_target.detection

        for det in detections:
            iou_score = self._simple_iou(target, det)

            if iou_score > 0.3:
                return True

        return False

    # ─────────────────────────────────────────────────────────

    @staticmethod
    def _simple_iou(a: Detection, b: Detection) -> float:

        xA = max(a.x1, b.x1)
        yA = max(a.y1, b.y1)
        xB = min(a.x2, b.x2)
        yB = min(a.y2, b.y2)

        inter_w = max(0, xB - xA)
        inter_h = max(0, yB - yA)

        inter = inter_w * inter_h

        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)

        union = area_a + area_b - inter

        if union <= 0:
            return 0.0

        return inter / union

    # ─────────────────────────────────────────────────────────

    def choose_target(
        self,
        detections: List[Detection],
        gripper_x: int,
        gripper_y: int
    ) -> Optional[RobotTarget]:

        # Keep current target if still visible
        if self._target_still_exists(detections):
            return self.current_target

        # Lost target
        self.current_target = None

        if not detections:
            return None

        closest: Optional[RobotTarget] = None

        for det in detections:

            cx, cy = self.get_box_center(det)

            dist = self._distance_to_gripper(
                det,
                gripper_x,
                gripper_y
            )

            target = RobotTarget(
                detection=det,
                center_x=cx,
                center_y=cy,
                distance=dist
            )

            if closest is None or dist < closest.distance:
                closest = target

        self.current_target = closest

        return closest

    # ─────────────────────────────────────────────────────────

    def generate_movement(
        self,
        gripper_x: int,
        gripper_y: int
    ) -> str:

        if self.current_target is None:
            return "NO TARGET"

        tx = self.current_target.center_x
        ty = self.current_target.center_y

        dx = tx - gripper_x
        dy = ty - gripper_y

        # ─────────────────────────────────
        # PRIORITY: HEIGHT (Y)
        # ─────────────────────────────────

        if PRIORITIZE_Y:

            if abs(dy) > Y_THRESHOLD:

                if dy > 0:
                    return "ARM GO DOWN"

                return "ARM GO UP"

        # ─────────────────────────────────
        # LEFT / RIGHT
        # ─────────────────────────────────

        if abs(dx) > X_THRESHOLD:

            if dx > 0:
                return "MOVE RIGHT"

            return "MOVE LEFT"

        # ─────────────────────────────────
        # CENTERED
        # ─────────────────────────────────

        return "TARGET LOCKED"