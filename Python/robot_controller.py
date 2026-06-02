"""
robot_controller.py
===================

Selects the best strawberry target and produces movement commands.
Also dispatches hardware commands to the servo submodules.

Hardware calls are fire-and-forget.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from detection import Detection
import config


# =============================================================================
# HARDWARE IMPORTS
# =============================================================================

try:
    import turntable as _turntable
    _HAS_TURNTABLE = True
except ImportError:
    _HAS_TURNTABLE = False

try:
    import lift as _lift
    _HAS_LIFT = True
except ImportError:
    _HAS_LIFT = False

try:
    import gripper as _gripper
    _HAS_GRIPPER = True
except ImportError:
    _HAS_GRIPPER = False


# =============================================================================
# CONFIG
# =============================================================================

X_THRESHOLD = 25
Y_THRESHOLD = 25

PRIORITIZE_Y = True
LOCK_HYSTERESIS = 100

DEPTH_CONF_WEIGHT = 0.2

HW_LOG_EVERY = 5


# =============================================================================
# DATA
# =============================================================================

@dataclass
class RobotTarget:
    detection: Detection
    center_x: int
    center_y: int
    distance: float
    depth_score: float = 0.0


# =============================================================================
# CONTROLLER
# =============================================================================

class RobotController:

    def __init__(self) -> None:

        self.current_target: Optional[RobotTarget] = None

        self._hw_log_counter = 0

        self._gripper_containment_frames = 0
        self._last_target_id = None

    # =========================================================================
    # GEOMETRY
    # =========================================================================

    @staticmethod
    def get_box_center(det: Detection) -> Tuple[int, int]:
        return (
            (det.x1 + det.x2) // 2,
            (det.y1 + det.y2) // 2,
        )

    @staticmethod
    def _distance_to(
        gx: int,
        gy: int,
        x: int,
        y: int,
    ) -> float:
        return math.hypot(x - gx, y - gy)

    # =========================================================================
    # DEPTH
    # =========================================================================

    @staticmethod
    def estimate_depth(det: Detection) -> float:

        area = max(
            1,
            (det.x2 - det.x1) * (det.y2 - det.y1),
        )

        ideal = max(
            1,
            config.BERRY_SIZE_IDEAL,
        )

        ratio = area / ideal

        if ratio >= 1.0:
            raw = math.exp(
                -((ratio - 1.0) ** 2) / (2 * 1.5 ** 2)
            )
        else:
            raw = math.exp(
                -((ratio - 1.0) ** 2) / (2 * 0.35 ** 2)
            )

        conf = float(det.confidence or 0.0)

        score = (
            raw * (1.0 - DEPTH_CONF_WEIGHT)
            + raw * conf * DEPTH_CONF_WEIGHT
        )

        return round(
            min(1.0, max(0.0, score)),
            3,
        )

    @staticmethod
    def depth_label(score: float) -> str:

        if score >= 0.80:
            return "CLOSE"

        if score >= 0.50:
            return "MEDIUM"

        if score >= 0.25:
            return "FAR"

        return "VERY FAR"

    # =========================================================================
    # TARGET SELECTION
    # =========================================================================

    @staticmethod
    def _simple_iou(
        a: Detection,
        b: Detection,
    ) -> float:

        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)

        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)

        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)

        union = area_a + area_b - inter

        if union <= 0:
            return 0.0

        return inter / union

    def _target_still_exists(
        self,
        detections: List[Detection],
    ) -> bool:

        if self.current_target is None:
            return False

        target = self.current_target.detection

        for det in detections:

            if self._simple_iou(target, det) > 0.3:

                cx, cy = self.get_box_center(det)

                self.current_target.detection = det
                self.current_target.center_x = cx
                self.current_target.center_y = cy
                self.current_target.depth_score = self.estimate_depth(det)

                return True

        return False

    def choose_target(
        self,
        detections: List[Detection],
        gripper_x: int,
        gripper_y: int,
    ) -> Optional[RobotTarget]:

        if not detections:
            self.current_target = None
            return None

        closest = None

        for det in detections:

            cx, cy = self.get_box_center(det)

            dist = self._distance_to(
                gripper_x,
                gripper_y,
                cx,
                cy,
            )

            depth = self.estimate_depth(det)

            candidate = RobotTarget(
                det,
                cx,
                cy,
                dist,
                depth,
            )

            if closest is None or dist < closest.distance:
                closest = candidate

        if self._target_still_exists(detections):

            current_distance = self._distance_to(
                gripper_x,
                gripper_y,
                self.current_target.center_x,
                self.current_target.center_y,
            )

            if closest.distance < current_distance - LOCK_HYSTERESIS:
                self.current_target = closest

            return self.current_target

        self.current_target = closest
        return closest

    # =========================================================================
    # GRIPPER
    # =========================================================================

    @staticmethod
    def get_gripper_bbox(
        gripper_x: int,
        gripper_y: int,
    ) -> Tuple[int, int, int, int]:

        half_w = config.GRIPPER_BB_WIDTH // 2
        half_h = config.GRIPPER_BB_HEIGHT // 2

        return (
            gripper_x - half_w,
            gripper_y - half_h,
            gripper_x + half_w,
            gripper_y + half_h,
        )

    @staticmethod
    def is_detection_fully_contained(
        det: Detection,
        bbox: Tuple[int, int, int, int],
    ) -> bool:

        x1, y1, x2, y2 = bbox

        return (
            det.x1 >= x1
            and det.y1 >= y1
            and det.x2 <= x2
            and det.y2 <= y2
        )

    def object_fill_ratio(self) -> float:

        if self.current_target is None:
            return 0.0

        det = self.current_target.detection

        target_area = (
            (det.x2 - det.x1)
            * (det.y2 - det.y1)
        )

        gripper_area = (
            config.GRIPPER_BB_WIDTH
            * config.GRIPPER_BB_HEIGHT
        )

        return target_area / max(1, gripper_area)

    def ready_to_grab(
        self,
        gripper_x: int,
        gripper_y: int,
    ) -> bool:

        if self.current_target is None:
            return False

        det = self.current_target.detection

        bbox = self.get_gripper_bbox(
            gripper_x,
            gripper_y,
        )

        contained = self.is_detection_fully_contained(
            det,
            bbox,
        )

        ratio = self.object_fill_ratio()

        dx = self.generate_dx(gripper_x)
        dy = self.generate_dy(gripper_y)

        centered = (
            abs(dx) <= config.GRAB_CENTER_TOLERANCE_X
            and abs(dy) <= config.GRAB_CENTER_TOLERANCE_Y
        )

        return (
            contained
            and centered
            and ratio >= config.MIN_GRAB_AREA_RATIO
        )

    def update_gripper_containment(
        self,
        gripper_x: int,
        gripper_y: int,
    ) -> None:

        if self.current_target is None:
            self._gripper_containment_frames = 0
            self._last_target_id = None
            return

        target_id = id(self.current_target.detection)

        if target_id != self._last_target_id:
            self._gripper_containment_frames = 0
            self._last_target_id = target_id

        bbox = self.get_gripper_bbox(
            gripper_x,
            gripper_y,
        )

        if self.is_detection_fully_contained(
            self.current_target.detection,
            bbox,
        ):
            self._gripper_containment_frames += 1
        else:
            self._gripper_containment_frames = 0

    # =========================================================================
    # OFFSETS
    # =========================================================================

    def generate_dx(self, gripper_x: int) -> int:

        if self.current_target is None:
            return 0

        return self.current_target.center_x - gripper_x

    def generate_dy(self, gripper_y: int) -> int:

        if self.current_target is None:
            return 0

        return self.current_target.center_y - gripper_y

    # =========================================================================
    # HUD
    # =========================================================================

    def generate_movementstring(
        self,
        gripper_x: int,
        gripper_y: int,
    ) -> str:

        if self.current_target is None:
            return "NO TARGET"

        dx = self.generate_dx(gripper_x)
        dy = self.generate_dy(gripper_y)

        if PRIORITIZE_Y and abs(dy) > Y_THRESHOLD:
            return "ARM GO DOWN" if dy > 0 else "ARM GO UP"

        if abs(dx) > X_THRESHOLD:
            return "MOVE RIGHT" if dx > 0 else "MOVE LEFT"

        return "TARGET LOCKED"

    def generate_depthstring(self) -> str:

        if self.current_target is None:
            return "DEPTH: no target"

        t = self.current_target

        area = (
            (t.detection.x2 - t.detection.x1)
            * (t.detection.y2 - t.detection.y1)
        )

        return (
            f"DEPTH: {self.depth_label(t.depth_score)} "
            f"(score={t.depth_score:.2f}, area={area}px²)"
        )

    def generate_gripperstring(self) -> str:

        if not _HAS_GRIPPER:
            return "GRIPPER: SIMULATED"

        state = _gripper.get_state()

        if self.current_target is None:
            return f"GRIPPER: {state} (no target)"

        required = config.GRIPPER_CONTAINMENT_FRAMES

        return (
            f"GRIPPER: {state} "
            f"({self._gripper_containment_frames}/{required})"
        )

    # =========================================================================
    # HARDWARE
    # =========================================================================

    def drive_hardware(
        self,
        gripper_x: int,
        gripper_y: int,
    ) -> None:

        self._hw_log_counter += 1

        do_log = (
            self._hw_log_counter % HW_LOG_EVERY == 0
        )

        if self.current_target is None:

            if _HAS_TURNTABLE:
                _turntable.stop()

            if _HAS_LIFT:
                _lift.stop()

            return

        dx = self.generate_dx(gripper_x)
        dy = self.generate_dy(gripper_y)

        if _HAS_TURNTABLE:
            tt_msg = _turntable.update(dx)
        else:
            tt_msg = f"TURNTABLE dx={dx}"

        if _HAS_LIFT:
            lift_msg = _lift.update(dy)
        else:
            lift_msg = f"LIFT dy={dy}"

        if _HAS_GRIPPER:

            self.update_gripper_containment(
                gripper_x,
                gripper_y,
            )

            gripper_msg = self.generate_gripperstring()

            if config.GRIPPER_AUTO_GRIP_ENABLED:

                if (
                    self._gripper_containment_frames
                    >= config.GRIPPER_CONTAINMENT_FRAMES
                    and self.ready_to_grab(
                        gripper_x,
                        gripper_y,
                    )
                ):

                    ratio = self.object_fill_ratio()

                    print(
                        f"[GRAB CHECK] "
                        f"ratio={ratio:.2f} "
                        f"frames={self._gripper_containment_frames}"
                    )

                    result = _gripper.grip()

                    if result["status"] == "ok":
                        gripper_msg = "GRIPPER: GRIPPING"

        else:
            gripper_msg = "GRIPPER SIMULATED"

        if do_log:
            print(
                f"[HW] "
                f"{tt_msg} | "
                f"{lift_msg} | "
                f"{gripper_msg}"
            )