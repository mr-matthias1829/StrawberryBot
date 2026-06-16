"""
robot_controller.py
===================

Selects the best strawberry target and produces movement commands.
Also dispatches hardware commands to the servo submodules.

Hardware calls are fire-and-forget.

Arm logic (autonomous mode)
----------------------------
The arm extends forward whenever the current target is fully contained
inside the gripper bounding box, regardless of apparent depth.
Once the target leaves the gripper area the arm stops.

Grip logic
----------
The gripper fires when:
  1. The berry is fully contained inside the gripper bbox.
  2. The berry's area fills at least config.MIN_GRAB_AREA_RATIO of the
     gripper bbox area — i.e. the berry is close enough.
     Example: 0.4 = berry bbox is 40% of the 500x500 gripper area.
  3. The above has been true for config.GRIPPER_CONTAINMENT_FRAMES
     consecutive frames.

Post-grip sequence
------------------
After a successful grip, drive_hardware() freezes:
  - all servos (turntable, lift, arm) stop
  - _gripped is set True, blocking all subsequent drive_hardware() calls
  - bin.place_berry() runs on a daemon worker thread
  - when place_berry() finishes, _gripped clears and tracking resumes

Smoothing
---------
Raw dx/dy offsets go through an EMA filter before reaching the servos.
Absorbs single-frame detection blinks and prevents micro-jitter on the
turntable and lift.

Dead-zone gate zeroes out any smoothed offset still below the X/Y
threshold so servos hold position instead of hunting around centre.

Tune EMA_ALPHA:
  0.4  — responsive, mild smoothing
  0.25 — balanced (default)
  0.1  — very smooth, noticeable lag on fast moves

Action debounce
---------------
Because the RTSP camera feed has latency, detections we receive now
correspond to what the camera saw ACTION_DEBOUNCE_S seconds ago.
All servo output is suppressed until the current target has been
continuously visible for at least ACTION_DEBOUNCE_S seconds.

  ACTION_DEBOUNCE_S = 0.0  — disables debounce
  ACTION_DEBOUNCE_S = 0.25 — good starting point for ~200-300ms feed lag
"""

import math
import time
import threading
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
    import arm as _arm
    _HAS_ARM = True
except ImportError:
    _HAS_ARM = False

try:
    import gripper as _gripper
    _HAS_GRIPPER = True
except ImportError:
    _HAS_GRIPPER = False

try:
    import bin as _bin
    _HAS_BIN = True
except ImportError:
    _HAS_BIN = False


X_THRESHOLD = 25
Y_THRESHOLD = 25

PRIORITIZE_Y = True

# EMA smoothing
EMA_ALPHA = 0.25

# lock hysteresis
LOCK_HYSTERESIS_BASE  = 40
LOCK_HYSTERESIS_DEPTH = 30

# ghost-lock grace period
LOCK_GHOST_FRAMES = 2

# candidate scoring weights
WEIGHT_DIST  = 0.5
WEIGHT_DEPTH = 0.5

# depth units
DEPTH_UNITS_MIN = 0.2
DEPTH_UNITS_MAX = 1000

# depth estimation
DEPTH_CONF_WEIGHT = 0.35

HW_LOG_EVERY = 5

# servo output is suppressed until the target has been continuously visible
# for this many seconds (accounts for RTSP feed latency). set to 0.0 to disable.
ACTION_DEBOUNCE_S = 0.3

@dataclass
class RobotTarget:
    detection:   Detection
    center_x:    int
    center_y:    int
    distance:    float
    depth_score: float = 0.0
    depth_units: float = 1.0


class RobotController:
    """select target strawberries and produce simple movement commands."""

    def __init__(self) -> None:
        self.current_target: Optional[RobotTarget] = None
        self._ghost_frames: int = 0
        self._hw_log_counter = 0
        self._gripper_containment_frames = 0
        self._last_target_id = None
        self._arm_extending: bool = False

        self._smooth_dx: float = 0.0
        self._smooth_dy: float = 0.0

        self._stable_since: Optional[float] = None   # monotonic timestamp
        self._debounce_logged: bool = False           # suppress repeat log spam

        # True while bin.place_berry() is running
        self._gripped: bool = False


    @staticmethod
    def get_box_center(det: Detection) -> Tuple[int, int]:
        return (det.x1 + det.x2) // 2, (det.y1 + det.y2) // 2

    @staticmethod
    def _distance_to(gripper_x: int, gripper_y: int, x: int, y: int) -> float:
        return math.hypot(x - gripper_x, y - gripper_y)


    @staticmethod
    def estimate_depth(det: Detection) -> float:
        area  = max(1, (det.x2 - det.x1) * (det.y2 - det.y1))
        ideal = max(1, config.BERRY_SIZE_IDEAL)
        ratio = area / ideal

        if ratio >= 1.0:
            raw_score = math.exp(-((ratio - 1.0) ** 2) / (2 * 1.5 ** 2))
        else:
            raw_score = math.exp(-((ratio - 1.0) ** 2) / (2 * 0.35 ** 2))

        conf  = float(det.confidence or 0.0)
        score = raw_score * (1.0 - DEPTH_CONF_WEIGHT) + raw_score * conf * DEPTH_CONF_WEIGHT
        return round(min(1.0, max(0.0, score)), 3)

    @staticmethod
    def estimate_depth_units(det: Detection) -> float:
        area  = max(1, (det.x2 - det.x1) * (det.y2 - det.y1))
        ideal = max(1, config.BERRY_SIZE_IDEAL)
        units = math.sqrt(ideal / area)
        return round(min(DEPTH_UNITS_MAX, max(DEPTH_UNITS_MIN, units)), 3)

    @staticmethod
    def depth_label(depth_score: float) -> str:
        if depth_score >= 0.80:
            return "CLOSE"
        if depth_score >= 0.50:
            return "MEDIUM"
        if depth_score >= 0.25:
            return "FAR"
        if depth_score > 0.0:
            return "VERY FAR"
        return "UNKNOWN"


    def _candidate_score(self, dist: float, depth_score: float, max_dist: float) -> float:
        dist_norm = dist / max(max_dist, 1.0)
        return dist_norm * WEIGHT_DIST + (1.0 - depth_score) * WEIGHT_DEPTH


    @staticmethod
    def _simple_iou(a: Detection, b: Detection) -> float:
        x_a    = max(a.x1, b.x1)
        y_a    = max(a.y1, b.y1)
        x_b    = min(a.x2, b.x2)
        y_b    = min(a.y2, b.y2)
        inter  = max(0, x_b - x_a) * max(0, y_b - y_a)
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
                self.current_target.depth_units = self.estimate_depth_units(det)
                self._ghost_frames = 0
                return True

        self._ghost_frames += 1
        if self._ghost_frames <= LOCK_GHOST_FRAMES:
            return True

        self._ghost_frames = 0
        return False

    def choose_target(
        self,
        detections: List[Detection],
        gripper_x: int,
        gripper_y: int,
    ) -> Optional[RobotTarget]:

        if not detections:
            self.current_target   = None
            self._ghost_frames    = 0
            self._stable_since    = None
            self._debounce_logged = False
            return None

        candidates: List[RobotTarget] = []
        for det in detections:
            cx, cy = self.get_box_center(det)
            dist   = self._distance_to(gripper_x, gripper_y, cx, cy)
            depth  = self.estimate_depth(det)
            units  = self.estimate_depth_units(det)
            candidates.append(RobotTarget(det, cx, cy, dist, depth, units))

        max_dist = max(c.distance for c in candidates)

        best = min(
            candidates,
            key=lambda c: self._candidate_score(c.distance, c.depth_score, max_dist),
        )

        target_alive = self._target_still_exists(detections)

        if target_alive:
            cur      = self.current_target
            cur_dist = self._distance_to(gripper_x, gripper_y, cur.center_x, cur.center_y)
            cur_score  = self._candidate_score(cur_dist, cur.depth_score, max_dist)
            best_score = self._candidate_score(best.distance, best.depth_score, max_dist)

            margin = (
                (LOCK_HYSTERESIS_BASE + LOCK_HYSTERESIS_DEPTH * cur.depth_score)
                / max(max_dist, 1.0)
            ) * WEIGHT_DIST

            if best_score < cur_score - margin:
                # switched to a better target — reset debounce
                self._ghost_frames    = 0
                self.current_target   = best
                self._stable_since    = None
                self._debounce_logged = False

            return self.current_target

        # new target acquired — start debounce clock
        self._ghost_frames    = 0
        self.current_target   = best
        self._stable_since    = time.monotonic()
        self._debounce_logged = False
        return self.current_target


    def _debounce_ready(self) -> bool:
        """returns True when the current target has been stable long enough to act on."""
        if ACTION_DEBOUNCE_S <= 0.0:
            return True
        if self._stable_since is None:
            self._stable_since = time.monotonic()
        elapsed = time.monotonic() - self._stable_since
        if elapsed >= ACTION_DEBOUNCE_S:
            return True
        if not self._debounce_logged:
            print(f"[DEBOUNCE] waiting {ACTION_DEBOUNCE_S - elapsed:.2f}s before acting")
            self._debounce_logged = True
        return False


    @staticmethod
    def get_gripper_bbox(gripper_x: int, gripper_y: int) -> Tuple[int, int, int, int]:
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
        det          = self.current_target.detection
        target_area  = (det.x2 - det.x1) * (det.y2 - det.y1)
        gripper_area = config.GRIPPER_BB_WIDTH * config.GRIPPER_BB_HEIGHT
        return target_area / max(1, gripper_area)

    def ready_to_grab(self, gripper_x: int, gripper_y: int) -> bool:
        if self.current_target is None:
            return False
        bbox      = self.get_gripper_bbox(gripper_x, gripper_y)
        contained = self.is_detection_fully_contained(self.current_target.detection, bbox)
        if not contained:
            return False
        return self.object_fill_ratio() >= config.MIN_GRAB_AREA_RATIO

    def update_gripper_containment(self, gripper_x: int, gripper_y: int) -> None:
        if self.current_target is None:
            self._gripper_containment_frames = 0
            self._last_target_id = None
            return

        target_id = id(self.current_target)
        if target_id != self._last_target_id:
            self._gripper_containment_frames = 0
            self._last_target_id = target_id

        if self.ready_to_grab(gripper_x, gripper_y):
            self._gripper_containment_frames += 1
        else:
            self._gripper_containment_frames = 0


    def _stop_all_servos(self) -> None:
        """hard-stop every moving axis immediately."""
        if _HAS_TURNTABLE:
            _turntable.stop()
        if _HAS_LIFT:
            _lift.stop()
        if _HAS_ARM:
            _arm.stop()
        self._arm_extending = False

    def _run_place_berry(self) -> None:
        """
        worker thread: run the full bin-placement sequence,
        then release the gripped lock so tracking resumes.
        """
        try:
            print("[RC] Starting bin placement sequence…")
            if _HAS_BIN:
                _bin.place_berry()
            else:
                print("[RC][SIM] bin not available — skipping place_berry()")
        except Exception as e:
            print(f"[RC] ERROR in place_berry thread: {e}")
        finally:
            self._gripped = False
            self._reset_ema()
            print("[RC] Bin placement done — resuming tracking")


    def generate_dx(self, gripper_x: int) -> int:
        if self.current_target is None:
            return 0
        return self.current_target.center_x - gripper_x

    def generate_dy(self, gripper_y: int) -> int:
        if self.current_target is None:
            return 0
        return self.current_target.center_y - gripper_y


    def _update_ema(self, raw_dx: int, raw_dy: int) -> Tuple[int, int]:
        self._smooth_dx = EMA_ALPHA * raw_dx + (1.0 - EMA_ALPHA) * self._smooth_dx
        self._smooth_dy = EMA_ALPHA * raw_dy + (1.0 - EMA_ALPHA) * self._smooth_dy
        dx = int(self._smooth_dx) if abs(self._smooth_dx) >= X_THRESHOLD else 0
        dy = int(self._smooth_dy) if abs(self._smooth_dy) >= Y_THRESHOLD else 0
        return dx, dy

    def _reset_ema(self) -> None:
        self._smooth_dx = 0.0
        self._smooth_dy = 0.0


    def generate_movementstring(self, gripper_x: int, gripper_y: int) -> str:
        if self._gripped:
            return "PLACING BERRY…"

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
        t    = self.current_target
        area = (t.detection.x2 - t.detection.x1) * (t.detection.y2 - t.detection.y1)
        return (
            f"DEPTH: {self.depth_label(t.depth_score)} "
            f"(score={t.depth_score:.2f}, "
            f"dist={t.depth_units:.2f}, "
            f"area={area}px²)"
        )

    def generate_gripperstring(self) -> str:
        if self._gripped:
            return "GRIPPER: PLACING BERRY"

        if not _HAS_GRIPPER:
            return "GRIPPER: SIMULATED"

        state = _gripper.get_state()

        if self.current_target is None:
            return f"GRIPPER: {state} (no target)"

        required = config.GRIPPER_CONTAINMENT_FRAMES
        return (
            f"GRIPPER: {state} "
            f"({self._gripper_containment_frames}/{required} "
            f"fill={self.object_fill_ratio():.2f})"
        )

    def generate_armstring(self, contained: bool) -> str:
        if not _HAS_ARM:
            return "ARM: SIMULATED"
        if self._gripped:
            return "ARM: STOPPED (placing)"
        return f"ARM: {'EXTENDING' if contained else 'STOPPED'}"


    def _drive_arm(self, gripper_x: int, gripper_y: int) -> Tuple[bool, str]:
        if self.current_target is None:
            if _HAS_ARM:
                _arm.stop()
            self._arm_extending = False
            return False, "ARM: no target"

        if not config.AUTO_MODE_ALLOW_MOVE:
            if _HAS_ARM:
                _arm.stop()
            if self._arm_extending:
                print("[ARM] Auto-move disabled — arm stopped")
                self._arm_extending = False
            return False, "ARM: DISABLED (auto-move off)"

        bbox      = self.get_gripper_bbox(gripper_x, gripper_y)
        contained = self.is_detection_fully_contained(
            self.current_target.detection, bbox
        )

        if contained:
            if _HAS_ARM:
                _arm.move_forward()
            if not self._arm_extending:
                print("[ARM] Target contained — extending arm forward")
                self._arm_extending = True
            return True, "ARM: EXTENDING (target contained)"
        else:
            if _HAS_ARM:
                _arm.stop()
            if self._arm_extending:
                print("[ARM] Target left gripper area — arm stopped")
                self._arm_extending = False
            return False, "ARM: STOPPED"

    def drive_hardware(self, gripper_x: int, gripper_y: int) -> None:
        self._hw_log_counter += 1
        do_log = self._hw_log_counter % HW_LOG_EVERY == 0

        # while bin.place_berry() is running on its worker thread, hold all
        # servos stopped. EMA decays toward zero so no lurch when tracking resumes.
        if self._gripped:
            self._stop_all_servos()
            self._update_ema(0, 0)
            return

        if self.current_target is None:
            if _HAS_TURNTABLE:
                _turntable.stop()
            if _HAS_LIFT:
                _lift.stop()
            if _HAS_ARM:
                _arm.stop()
            self._arm_extending = False
            self._update_ema(0, 0)
            return

        # hold position (let EMA decay) until the target has been stable long
        # enough to compensate for camera feed latency
        if not self._debounce_ready():
            self._update_ema(0, 0)
            if _HAS_TURNTABLE:
                _turntable.stop()
            if _HAS_LIFT:
                _lift.stop()
            if _HAS_ARM:
                _arm.stop()
            return

        raw_dx = self.generate_dx(gripper_x)
        raw_dy = self.generate_dy(gripper_y)
        dx, dy = self._update_ema(raw_dx, raw_dy)

        if _HAS_TURNTABLE:
            tt_msg = _turntable.update(-dx)
        else:
            tt_msg = f"TURNTABLE dx={-dx}"

        if _HAS_LIFT:
            lift_msg = _lift.update(dy)
        else:
            lift_msg = f"LIFT dy={dy}"

        contained, arm_msg = self._drive_arm(gripper_x, gripper_y)

        gripper_msg = "GRIPPER: no hardware"
        if _HAS_GRIPPER:
            self.update_gripper_containment(gripper_x, gripper_y)
            gripper_msg = self.generate_gripperstring()
            if config.GRIPPER_AUTO_GRIP_ENABLED and config.AUTO_MODE_ALLOW_MOVE:
                if self._gripper_containment_frames >= config.GRIPPER_CONTAINMENT_FRAMES:
                    fill = self.object_fill_ratio()
                    print(
                        f"[GRAB CHECK] "
                        f"fill={fill:.2f} (need {config.MIN_GRAB_AREA_RATIO:.2f}) "
                        f"frames={self._gripper_containment_frames}"
                    )
                    if fill >= config.MIN_GRAB_AREA_RATIO:
                        result = _gripper.grip()
                        if result["status"] == "ok":
                            self._gripper_containment_frames = 0
                            gripper_msg = "GRIPPER: GRIPPED"

                            self._stop_all_servos()
                            self._gripped = True

                            threading.Thread(
                                target=self._run_place_berry,
                                daemon=True,
                                name="bin-placer",
                            ).start()
            else:
                gripper_msg = "GRIPPER SIMULATED"

        if do_log:
            print(
                f"[HW] "
                f"{tt_msg} | "
                f"{lift_msg} | "
                f"{arm_msg} | "
                f"{gripper_msg}"
            )