import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from detection import Detection


X_THRESHOLD = 25
Y_THRESHOLD = 25
PRIORITIZE_Y = True


@dataclass
class RobotTarget:
    detection: Detection
    center_x: int
    center_y: int
    distance: float

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

    def _target_still_exists(self, detections: List[Detection]) -> bool:
        if self.current_target is None:
            return False

        target = self.current_target.detection
        for det in detections:
            if self._simple_iou(target, det) > 0.3:
                cx, cy = self.get_box_center(det)
                self.current_target.detection = det
                self.current_target.center_x = cx
                self.current_target.center_y = cy
                self.current_target.distance = 0.0
                return True
        return False

    @staticmethod
    def _simple_iou(a: Detection, b: Detection) -> float:
        x_a = max(a.x1, b.x1)
        y_a = max(a.y1, b.y1)
        x_b = min(a.x2, b.x2)
        y_b = min(a.y2, b.y2)
        inter = max(0, x_b - x_a) * max(0, y_b - y_a)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def choose_target(self, detections: List[Detection], gripper_x: int, gripper_y: int) -> Optional[RobotTarget]:
        if self._target_still_exists(detections):
            return self.current_target
        self.current_target = None
        if not detections:
            return None

        closest: Optional[RobotTarget] = None
        for det in detections:
            cx, cy = self.get_box_center(det)
            dist = self._distance_to(gripper_x, gripper_y, cx, cy)
            target = RobotTarget(det, cx, cy, dist)
            if closest is None or dist < closest.distance:
                closest = target

        self.current_target = closest
        return closest

    def generate_movement(self, gripper_x: int, gripper_y: int) -> str:

        if self.current_target is None:
            return "NO TARGET"

        tx = self.current_target.center_x
        ty = self.current_target.center_y

        dx = tx - gripper_x
        dy = ty - gripper_y
        if PRIORITIZE_Y:
            if abs(dy) > Y_THRESHOLD:
                if dy > 0:
                    return "ARM GO DOWN"
                return "ARM GO UP"
        if abs(dx) > X_THRESHOLD:
            if dx > 0:
                return "MOVE RIGHT"
            return "MOVE LEFT"


        return "TARGET LOCKED"