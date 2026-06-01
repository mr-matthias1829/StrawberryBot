"""
turntable.py
============
Controls the turntable servo (AX-12A, ID 13) in wheel / continuous-rotation
mode so the robot can pan left or right to track a strawberry on the X-axis.

Architecture
------------
- Does NOT open serial/GPIO — motor.py owns those resources.
- Call motor.init() then turntable.init() once at startup.
- update(dx) is the only call needed per frame from RobotController.
- All serial writes run on a tiny background thread so they never block
  the vision/inference pipeline.

AX-12A wheel-mode register layout (MOVING_SPEED reg 32)
--------------------------------------------------------
    Bits 0-9  →  speed magnitude  (0 = stop)
    Bit  10   →  0 = CCW (LEFT) / 1 = CW (RIGHT)

Wheel mode is activated by setting both angle-limit registers to 0.

Coordinate convention (matches robot_controller.py)
----------------------------------------------------
    dx > 0  →  target is RIGHT of gripper  →  spin RIGHT (CW)
    dx < 0  →  target is LEFT  of gripper  →  spin LEFT  (CCW)
    |dx| <= DEAD_ZONE  →  aligned, stop
"""

import threading
import time

import motor

# =============================================================================
# TUNING
# =============================================================================

SERVO_ID  = 13
DEAD_ZONE = 25          # px — mirrors X_THRESHOLD in robot_controller.py

# Speed tiers (0–1023)
SPEED_SLOW   = 150      # just outside dead zone
SPEED_MEDIUM = 300      # normal tracking
SPEED_FAST   = 500      # large error

THRESHOLD_SLOW   = 50   # |dx| ≤ this → SLOW
THRESHOLD_MEDIUM = 150  # |dx| ≤ this → MEDIUM, above → FAST

# AX-12A registers
_REG_CW_LIMIT   = 6
_REG_CCW_LIMIT  = 8
_REG_TORQUE_EN  = 24
_REG_SPEED      = 32

# Direction bits
_DIR_CCW = 0            # counter-clockwise → LEFT
_DIR_CW  = 1 << 10      # clockwise         → RIGHT

# =============================================================================
# STATE
# =============================================================================

_initialized:  bool = False
_pending_word: int  = -1        # -1 = nothing pending
_last_word:    int  = -1        # last word actually sent
_lock = threading.Lock()
_event = threading.Event()
_stop_flag = False
_thread: threading.Thread = None  # type: ignore[assignment]


# =============================================================================
# BACKGROUND WRITER THREAD
# =============================================================================
# Serial writes to the AX-12A take ~2 ms each.  Running them on the main
# thread costs real frame time.  This thread wakes up whenever update() posts
# a new speed word, sends it, then goes back to sleep.  If multiple frames
# arrive while one write is in progress only the latest word is sent — we
# never queue stale commands.

def _writer() -> None:
    global _last_word
    while not _stop_flag:
        fired = _event.wait(timeout=0.1)
        if not fired:
            continue
        _event.clear()
        with _lock:
            word = _pending_word
        if word < 0:
            continue
        if word == _last_word:
            continue           # nothing changed — skip the write
        try:
            motor._write_word(SERVO_ID, _REG_SPEED, word)
            _last_word = word
        except Exception as e:
            print(f"[turntable] serial error: {e}")


# =============================================================================
# LIFECYCLE
# =============================================================================

def init() -> None:
    """Switch servo to wheel mode and start background writer. Idempotent."""
    global _initialized, _stop_flag, _thread

    if _initialized:
        return

    _stop_flag = False

    # Disable torque before changing mode registers
    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 0)
    time.sleep(0.05)

    # Wheel mode: both angle limits = 0
    motor._write_word(SERVO_ID, _REG_CW_LIMIT,  0)
    time.sleep(0.02)
    motor._write_word(SERVO_ID, _REG_CCW_LIMIT, 0)
    time.sleep(0.02)

    # Re-enable torque
    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 1)
    time.sleep(0.05)

    # Explicit stop
    motor._write_word(SERVO_ID, _REG_SPEED, 0)

    # Start background writer
    _thread = threading.Thread(target=_writer, daemon=True, name="turntable-writer")
    _thread.start()

    _initialized = True
    print(f"✅ Turntable initialised (ID {SERVO_ID}, wheel mode).")


def shutdown() -> None:
    """Stop motor and disable torque. Called automatically via motor.shutdown()."""
    global _initialized, _stop_flag

    if not _initialized:
        return

    _stop_flag = True
    _event.set()
    if _thread is not None:
        _thread.join(timeout=1.0)

    try:
        motor._write_word(SERVO_ID, _REG_SPEED,    0)
        motor._write_word(SERVO_ID, _REG_TORQUE_EN, 0)
    except Exception:
        pass

    _initialized = False
    print("🛑 Turntable shut down.")


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _post_word(word: int) -> None:
    """Post a speed word to the background writer (non-blocking)."""
    with _lock:
        global _pending_word
        _pending_word = word
    _event.set()


def _speed_from_dx(dx_abs: int) -> int:
    if dx_abs <= DEAD_ZONE:
        return 0
    if dx_abs <= THRESHOLD_SLOW:
        return SPEED_SLOW
    if dx_abs <= THRESHOLD_MEDIUM:
        return SPEED_MEDIUM
    return SPEED_FAST


# =============================================================================
# PUBLIC API
# =============================================================================

def stop() -> None:
    """Post a stop command (non-blocking)."""
    _post_word(0)


def spin_left(speed: int = SPEED_MEDIUM) -> dict:
    """Spin CCW (gripper moves LEFT). Non-blocking."""
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CCW | speed)
    return {"direction": "left", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def spin_right(speed: int = SPEED_MEDIUM) -> dict:
    """Spin CW (gripper moves RIGHT). Non-blocking."""
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CW | speed)
    return {"direction": "right", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def update(dx: int) -> str:
    """
    Main per-frame entry point. Posts the appropriate speed word and returns
    a log string. Never blocks — serial write is handled in background thread.

    Args:
        dx: target_x - gripper_x  (from RobotController.generate_dx)
    """
    if not _initialized:
        return f"TURNTABLE SIMULATED (dx={dx:+d})"

    speed = _speed_from_dx(abs(dx))

    if dx > DEAD_ZONE:
        spin_right(speed)
        return f"TURNTABLE RIGHT (dx={dx:+d}, speed={speed})"

    if dx < -DEAD_ZONE:
        spin_left(speed)
        return f"TURNTABLE LEFT  (dx={dx:+d}, speed={speed})"

    stop()
    return f"TURNTABLE STOP  (dx={dx:+d}, aligned)"