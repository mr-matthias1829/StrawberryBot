"""
robot_controller.py
===================
Target selection and XY→servo movement for the strawberry harvester arm.

OVERVIEW
--------
RobotController selects which detected strawberry to pursue and translates
the pixel-space error (distance between gripper and target) into servo commands.

Two axes, two servos:
  - Horizontal (X) error → pan servo  (JointServo or WheelServo depending on rig)
  - Vertical   (Y) error → tilt servo (JointServo)

The controller works in image-space pixels. The frame centre is treated as the
gripper's current aim point. Error is the vector from that point to the target
berry's bounding-box centre.

DEAD ZONES
----------
X_THRESHOLD and Y_THRESHOLD are pixel dead zones. If the error is smaller than
the threshold the axis is considered "locked" and its servo is not moved.
This prevents constant micro-corrections when the arm is already on-target.

PRIORITISATION
--------------
PRIORITIZE_Y = True means vertical alignment is corrected first. This is useful
when the picking mechanism needs to be at the right height before advancing
horizontally. Set to False to prioritise horizontal movement.

SPEED SCALING
-------------
All servo commands are gated through ServoController.speed_scale (0.0–1.0).
As long as that is 0.0 (the default) nothing physical moves, regardless of
what the target logic produces. Raise it from the web dashboard slider.
"""

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from detection import Detection
from dynamixel import JointServo, WheelServo, _BaseServo

# ─────────────────────────────────────────────────────────────────────────────
# TUNING CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

X_THRESHOLD = 25    # Pixels. Horizontal error below this → don't move pan servo.
                    # Increase if the arm vibrates/hunts around the target.

Y_THRESHOLD = 25    # Pixels. Vertical error below this → don't move tilt servo.

PRIORITIZE_Y = True # If True, fix vertical alignment before horizontal.
                    # Set False to prioritise left/right movement instead.

# How much of the full servo range one pixel of error maps to.
# Smaller = slower/more precise movements. Larger = faster but more overshoot.
# These are starting values — tune for your physical arm geometry.
PAN_PIXELS_PER_STEP  = 8    # Every N pixels of X error → move pan by 1 position unit
TILT_PIXELS_PER_STEP = 8    # Every N pixels of Y error → move tilt by 1 position unit

# Maximum position step per update cycle (caps how fast the arm can sweep).
# Prevents a large initial error from flinging the arm to its limit in one go.
MAX_PAN_STEP  = 40   # position units per update
MAX_TILT_STEP = 40   # position units per update


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RobotTarget:
    """
    A strawberry that has been selected as the current pursuit target.

    Attributes
    ----------
    detection : Detection   The underlying detection object (bounding box + confidence).
    center_x  : int         Pixel X of the berry's bounding-box centre.
    center_y  : int         Pixel Y of the berry's bounding-box centre.
    distance  : float       Euclidean pixel distance from gripper at time of selection.
                            Updated to 0.0 when an existing target is re-confirmed.
    """
    detection: Detection
    center_x:  int
    center_y:  int
    distance:  float


# ─────────────────────────────────────────────────────────────────────────────
# CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class RobotController:
    """
    Selects target strawberries and drives servos to aim the arm at them.

    Parameters
    ----------
    pan_servo  : JointServo or WheelServo or None
        The servo that moves the arm left/right (horizontal / X axis).
        Pass None if you have no pan servo yet — horizontal commands are silently skipped.
    tilt_servo : JointServo or None
        The servo that moves the arm up/down (vertical / Y axis).
        Pass None to skip vertical commands.

    Attributes
    ----------
    current_target : RobotTarget or None
        The berry currently being tracked. None when no detections are available
        or between frames. Automatically updated by update().
    pan_servo  : servo instance or None
    tilt_servo : servo instance or None

    Usage
    -----
    # In your main loop, after fusion.process_frame():
    robot.update(confirmed_hits, frame_width, frame_height)
    """

    def __init__(
        self,
        pan_servo:  Optional[_BaseServo] = None,
        tilt_servo: Optional[JointServo] = None,
    ) -> None:
        self.pan_servo:     Optional[_BaseServo] = pan_servo
        self.tilt_servo:    Optional[JointServo] = tilt_servo
        self.current_target: Optional[RobotTarget] = None




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
                if dy > 0:
                    return "ARM GO DOWN"
                return "ARM GO UP"
        if abs(dx) > X_THRESHOLD:
            if dx > 0:
                return "MOVE RIGHT"
            return "MOVE LEFT"


        return "TARGET LOCKED"



    # ── target selection ──────────────────────────────────────────────────────

    @staticmethod
    def get_box_center(det: Detection) -> Tuple[int, int]:
        """Return the pixel centre (cx, cy) of a detection's bounding box."""
        return (det.x1 + det.x2) // 2, (det.y1 + det.y2) // 2

    @staticmethod
    def _distance(x1: int, y1: int, x2: int, y2: int) -> float:
        return math.hypot(x2 - x1, y2 - y1)

    @staticmethod
    def _iou(a: Detection, b: Detection) -> float:
        """IoU between two Detection bounding boxes (0.0–1.0)."""
        ix1 = max(a.x1, b.x1)
        iy1 = max(a.y1, b.y1)
        ix2 = min(a.x2, b.x2)
        iy2 = min(a.y2, b.y2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
        area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
        union  = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _target_still_exists(self, detections: List[Detection]) -> bool:
        """
        Check if the current target is still visible in the new detection list.

        Uses IoU > 0.3 as a "same berry" heuristic. If found, updates the
        target's bounding box and centre in-place.

        Returns True if the target was found and updated, False if it is gone.
        """
        if self.current_target is None:
            return False
        for det in detections:
            if self._iou(self.current_target.detection, det) > 0.3:
                cx, cy = self.get_box_center(det)
                self.current_target.detection = det
                self.current_target.center_x  = cx
                self.current_target.center_y  = cy
                self.current_target.distance  = 0.0
                return True
        return False

    def choose_target(
        self,
        detections: List[Detection],
        gripper_x:  int,
        gripper_y:  int,
    ) -> Optional[RobotTarget]:
        """
        Select or maintain the pursuit target.

        Strategy: keep the existing target if it is still visible (stability).
        If the target is gone, pick the closest berry to the gripper (greedy).

        Parameters
        ----------
        detections : list of Detection   All confirmed detections this frame.
        gripper_x  : int                 Pixel X of the gripper's current aim point.
        gripper_y  : int                 Pixel Y of the gripper's current aim point.

        Returns
        -------
        RobotTarget or None
        """
        if self._target_still_exists(detections):
            return self.current_target

        self.current_target = None
        if not detections:
            return None

        closest: Optional[RobotTarget] = None
        for det in detections:
            cx, cy = self.get_box_center(det)
            dist   = self._distance(gripper_x, gripper_y, cx, cy)
            cand   = RobotTarget(det, cx, cy, dist)
            if closest is None or dist < closest.distance:
                closest = cand

        self.current_target = closest
        return closest

    # ── error helpers (still useful for external callers / logging) ───────────

    def dx(self, gripper_x: int) -> int:
        """Signed pixel error on X axis. Positive = target is to the right."""
        return 0 if self.current_target is None else self.current_target.center_x - gripper_x

    def dy(self, gripper_y: int) -> int:
        """Signed pixel error on Y axis. Positive = target is below."""
        return 0 if self.current_target is None else self.current_target.center_y - gripper_y

    # ── servo movement ────────────────────────────────────────────────────────

    def _move_pan(self, error_x: int) -> None:
        """
        Translate a horizontal pixel error into a pan servo command.

        For JointServo: nudges the current position by a scaled step.
        For WheelServo: spins briefly in the error direction (not yet implemented
                        — wheel pan is tricky without position feedback; placeholder).

        Parameters
        ----------
        error_x : int   Signed pixel distance. Positive = need to move right.
        """
        if self.pan_servo is None or abs(error_x) <= X_THRESHOLD:
            return

        if isinstance(self.pan_servo, JointServo):
            # Calculate a proportional step, capped at MAX_PAN_STEP
            step = int(abs(error_x) / PAN_PIXELS_PER_STEP)
            step = min(step, MAX_PAN_STEP)
            current = self.pan_servo.get_position()
            if current is None:
                return
            target_pos = current + step if error_x > 0 else current - step
            self.pan_servo.move_to(target_pos)

        elif isinstance(self.pan_servo, WheelServo):
            # Wheel pan: spin proportionally. No position feedback — caller
            # must call stop() when aligned (see update() below).
            speed = min(
                int(abs(error_x) / PAN_PIXELS_PER_STEP) * 10,
                self.pan_servo.max_speed,
            )
            self.pan_servo.spin(max(1, speed), clockwise=(error_x > 0))

    def _move_tilt(self, error_y: int) -> None:
        """
        Translate a vertical pixel error into a tilt servo command.

        Only JointServo is supported for tilt — a free-spinning wheel on the
        vertical axis would be dangerous if position feedback is lost.

        Parameters
        ----------
        error_y : int   Signed pixel distance. Positive = target is below.
        """
        if self.tilt_servo is None or abs(error_y) <= Y_THRESHOLD:
            return

        step = int(abs(error_y) / TILT_PIXELS_PER_STEP)
        step = min(step, MAX_TILT_STEP)
        current = self.tilt_servo.get_position()
        if current is None:
            return
        target_pos = current + step if error_y > 0 else current - step
        self.tilt_servo.move_to(target_pos)

    # ── main update (call once per frame) ─────────────────────────────────────

    def update(
        self,
        detections:    List[Detection],
        frame_width:   int,
        frame_height:  int,
    ) -> str:
        """
        Run one control cycle: select target, compute error, move servos.

        Call this once per processed frame, after fusion.process_frame().

        Parameters
        ----------
        detections   : list of Detection   Confirmed hits from this frame.
        frame_width  : int                 Width of the camera frame in pixels.
        frame_height : int                 Height of the camera frame in pixels.

        Returns
        -------
        str   A human-readable status string for logging / dashboard display.
              One of: "NO TARGET", "TARGET LOCKED", "MOVE RIGHT", "MOVE LEFT",
              "ARM GO DOWN", "ARM GO UP", or a combination description.

        Behaviour when speed_scale == 0.0
        ----------------------------------
        The target is still selected and errors are computed (so the dashboard
        can show what the arm *would* do), but no servo commands are sent.
        The status string will include "(PAUSED)" in this case.
        """
        # Treat frame centre as the gripper's aim point
        gripper_x = frame_width  // 2
        gripper_y = frame_height // 2

        target = self.choose_target(detections, gripper_x, gripper_y)
        if target is None:
            return "NO TARGET"

        error_x = self.dx(gripper_x)
        error_y = self.dy(gripper_y)

        locked_x = abs(error_x) <= X_THRESHOLD
        locked_y = abs(error_y) <= Y_THRESHOLD

        if locked_x and locked_y:
            # Stop wheel pan if it was running
            if isinstance(self.pan_servo, WheelServo):
                self.pan_servo.stop()
            return "TARGET LOCKED"

        # Check if movement is actually allowed
        scale_zero = False
        if self.pan_servo is not None:
            scale_zero = self.pan_servo.ctrl.speed_scale == 0.0
        elif self.tilt_servo is not None:
            scale_zero = self.tilt_servo.ctrl.speed_scale == 0.0

        # Build status string describing what the arm wants to do
        parts = []
        if PRIORITIZE_Y:
            if not locked_y:
                parts.append("ARM GO DOWN" if error_y > 0 else "ARM GO UP")
                self._move_tilt(error_y)
            elif not locked_x:
                parts.append("MOVE RIGHT" if error_x > 0 else "MOVE LEFT")
                self._move_pan(error_x)
        else:
            if not locked_x:
                parts.append("MOVE RIGHT" if error_x > 0 else "MOVE LEFT")
                self._move_pan(error_x)
            elif not locked_y:
                parts.append("ARM GO DOWN" if error_y > 0 else "ARM GO UP")
                self._move_tilt(error_y)

        status = " + ".join(parts) if parts else "ALIGNING"
        if scale_zero:
            status += " (PAUSED)"

        return status