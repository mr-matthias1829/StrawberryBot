"""
robot_controller.py
===================
Selects the best strawberry target and produces movement commands.
Also dispatches hardware commands to the servo submodules (turntable, lift, etc.).

Hardware calls are fire-and-forget: each submodule owns a background thread
so no serial write ever blocks the vision pipeline.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from detection import Detection
import config

# ---------------------------------------------------------------------------
# Hardware submodule imports — all guarded so the code runs fine on a dev
# machine that has no Pi / no serial port.
# ---------------------------------------------------------------------------
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

# Future submodules — uncomment when ready:
# try:
#     import arm as _arm
#     _HAS_ARM = True
# except ImportError:
#     _HAS_ARM = False


# =============================================================================
# CONFIG
# =============================================================================

X_THRESHOLD     = 25    # px — horizontal dead zone
Y_THRESHOLD     = 25    # px — vertical   dead zone
PRIORITIZE_Y    = True  # move arm up/down before turntable when both are off
LOCK_HYSTERESIS = 100   # px — new target must beat current by this to steal lock

DEPTH_CONF_WEIGHT = 0.2  # how much detection confidence softens depth score

# How often to print the hardware log line (every N detection cycles).
# Set to 1 to log every detection, higher to reduce terminal spam.
HW_LOG_EVERY = 5


# =============================================================================
# DATA
# =============================================================================

@dataclass
class RobotTarget:
    detection:   Detection
    center_x:    int
    center_y:    int
    distance:    float
    depth_score: float = 0.0   # 0.0 = far, 1.0 = at ideal picking distance


# =============================================================================
# CONTROLLER
# =============================================================================

class RobotController:
    """Select target strawberries and drive hardware submodules."""

    def __init__(self) -> None:
        self.current_target: Optional[RobotTarget] = None
        self._hw_log_counter: int = 0

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    @staticmethod
    def get_box_center(det: Detection) -> Tuple[int, int]:
        return (det.x1 + det.x2) // 2, (det.y1 + det.y2) // 2

    @staticmethod
    def _distance_to(gx: int, gy: int, x: int, y: int) -> float:
        return math.hypot(x - gx, y - gy)

    # ------------------------------------------------------------------
    # Depth estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_depth(det: Detection) -> float:
        """
        Estimate relative depth from bounding-box area.
        Returns score in [0.0, 1.0]:  1.0 = close, 0.0 = far.
        Asymmetric curve: too-small penalised more than too-large.
        """
        area  = max(1, (det.x2 - det.x1) * (det.y2 - det.y1))
        ideal = max(1, config.BERRY_SIZE_IDEAL)
        ratio = area / ideal

        if ratio >= 1.0:
            raw = math.exp(-((ratio - 1.0) ** 2) / (2 * 1.5 ** 2))
        else:
            raw = math.exp(-((ratio - 1.0) ** 2) / (2 * 0.35 ** 2))

        conf  = float(det.confidence or 0.0)
        score = raw * (1.0 - DEPTH_CONF_WEIGHT) + raw * conf * DEPTH_CONF_WEIGHT
        return round(min(1.0, max(0.0, score)), 3)

    @staticmethod
    def depth_label(score: float) -> str:
        if score >= 0.80: return "CLOSE"
        if score >= 0.50: return "MEDIUM"
        if score >= 0.25: return "FAR"
        return "VERY FAR"

    # ------------------------------------------------------------------
    # Target selection
    # ------------------------------------------------------------------

    @staticmethod
    def _simple_iou(a: Detection, b: Detection) -> float:
        ix1 = max(a.x1, b.x1); iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2); iy2 = min(a.y2, b.y2)
        inter  = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _target_still_exists(self, detections: List[Detection]) -> bool:
        if self.current_target is None:
            return False
        target = self.current_target.detection
        for det in detections:
            if self._simple_iou(target, det) > 0.3:
                cx, cy = self.get_box_center(det)
                self.current_target.detection   = det
                self.current_target.center_x    = cx
                self.current_target.center_y    = cy
                self.current_target.distance    = 0.0
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

        # Find closest candidate
        closest: Optional[RobotTarget] = None
        for det in detections:
            cx, cy = self.get_box_center(det)
            dist   = self._distance_to(gripper_x, gripper_y, cx, cy)
            depth  = self.estimate_depth(det)
            cand   = RobotTarget(det, cx, cy, dist, depth)
            if closest is None or dist < closest.distance:
                closest = cand

        # Keep locked target unless a much-closer one exists
        if self._target_still_exists(detections):
            cur_dist = self._distance_to(
                gripper_x, gripper_y,
                self.current_target.center_x,
                self.current_target.center_y,
            )
            if closest.distance < cur_dist - LOCK_HYSTERESIS:
                self.current_target = closest
            return self.current_target

        self.current_target = closest
        return self.current_target

    # ------------------------------------------------------------------
    # Offset generators
    # ------------------------------------------------------------------

    def generate_dx(self, gripper_x: int) -> int:
        return 0 if self.current_target is None else self.current_target.center_x - gripper_x

    def generate_dy(self, gripper_y: int) -> int:
        return 0 if self.current_target is None else self.current_target.center_y - gripper_y

    # ------------------------------------------------------------------
    # HUD strings
    # ------------------------------------------------------------------

    def generate_movementstring(self, gripper_x: int, gripper_y: int) -> str:
        if self.current_target is None:
            return "NO TARGET"
        dx = self.current_target.center_x - gripper_x
        dy = self.current_target.center_y - gripper_y
        if PRIORITIZE_Y and abs(dy) > Y_THRESHOLD:
            return "ARM GO DOWN" if dy > 0 else "ARM GO UP"
        if abs(dx) > X_THRESHOLD:
            return "MOVE RIGHT" if dx > 0 else "MOVE LEFT"
        return "TARGET LOCKED"

    def generate_depthstring(self) -> str:
        if self.current_target is None:
            return "DEPTH: no target"
        t    = self.current_target
        area = (t.detection.x2 - t.detection.x1) * (t.detection.y2 - t.detection.y1)
        return (
            f"DEPTH: {self.depth_label(t.depth_score)} "
            f"(score={t.depth_score:.2f}, area={area}px²)"
        )

    # ------------------------------------------------------------------
    # Hardware dispatch  ← called once per detection cycle, never per raw frame
    # ------------------------------------------------------------------

    def drive_hardware(self, gripper_x: int, gripper_y: int) -> None:
        """
        Fire-and-forget hardware commands for the current frame.
        All serial writes happen in each submodule's background thread —
        this method returns immediately and never blocks the pipeline.

        Log output is throttled to every HW_LOG_EVERY detection cycles.
        """
        self._hw_log_counter += 1
        do_log = (self._hw_log_counter % HW_LOG_EVERY == 0)

        if self.current_target is None:
            if _HAS_TURNTABLE:
                _turntable.stop()
            if _HAS_LIFT:
                _lift.stop()
            if do_log:
                print("[HW] turntable=STOP  lift=STOP  (no target)")
            return

        dx = self.generate_dx(gripper_x)
        dy = self.generate_dy(gripper_y)

        # ── Turntable (X axis) ────────────────────────────────────────────────
        if _HAS_TURNTABLE:
            tt_msg = _turntable.update(dx)
        else:
            if   dx >  X_THRESHOLD: tt_msg = f"TURNTABLE SIMULATED RIGHT (dx={dx:+d})"
            elif dx < -X_THRESHOLD: tt_msg = f"TURNTABLE SIMULATED LEFT  (dx={dx:+d})"
            else:                   tt_msg = f"TURNTABLE SIMULATED STOP  (dx={dx:+d})"

        # ── Lift (Y axis) ─────────────────────────────────────────────────────
        if _HAS_LIFT:
            lift_msg = _lift.update(dy)
        else:
            if   dy >  Y_THRESHOLD: lift_msg = f"LIFT SIMULATED DOWN (dy={dy:+d})"
            elif dy < -Y_THRESHOLD: lift_msg = f"LIFT SIMULATED UP   (dy={dy:+d})"
            else:                   lift_msg = f"LIFT SIMULATED STOP (dy={dy:+d})"

        # ── Future: arm (depth / Z axis) ──────────────────────────────────────
        # arm_msg = _arm.update(self.current_target.depth_score) if _HAS_ARM else "ARM SIM"

        if do_log:
            print(f"[HW] {tt_msg} | {lift_msg}")