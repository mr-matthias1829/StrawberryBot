"""
turntable.py
============
Controls the turntable servo (AX-12A, ID 13) in wheel /
continuous-rotation mode so the robot can pan left or right to track
a strawberry on the X-axis.

Architecture
------------
- Does NOT open serial/GPIO — motor.py owns those resources.
- Call motor.init() then turntable.init() once at startup.
- update(dx) is the only call needed per frame from RobotController.
- All serial writes run on a background thread so they never block the
  vision/inference pipeline.

Corner sensor (AS5600 via TCA9548A, TCA channel: see SENSOR_CHANNEL)
---------------------------------------------------------------------
Travel limits are enforced in update() via MIN_DEG / MAX_DEG.
If the sensor is unavailable the motor runs open-loop (no limit enforcement).

Coordinate convention (matches robot_controller.py)
----------------------------------------------------
    dx > 0  → target is RIGHT of gripper → spin RIGHT (CW)
    dx < 0  → target is LEFT  of gripper → spin LEFT  (CCW)
    |dx| ≤ DEAD_ZONE  → aligned, stop
"""

import threading
import time

import motor
import servo_status
from corner_sensors import CornerSensorManager

# =============================================================================
# TUNING
# =============================================================================

SERVO_ID  = 13
DEAD_ZONE = 25

SPEED_SLOW   = 150
SPEED_MEDIUM = 300
SPEED_FAST   = 500

THRESHOLD_SLOW   = 50
THRESHOLD_MEDIUM = 150

SENSOR_CHANNEL = 0   # TCA9548A channel wired to this encoder

# Soft travel limits in degrees (0–360).  The turntable can rotate freely,
# so a lap-aware limit makes more sense long-term, but degree-within-one-turn
# is sufficient until calibration is done.
# Set MIN_DEG = MAX_DEG = None to disable limit enforcement entirely.
MIN_DEG: float | None = 10.0    # TODO: calibrate after hardware bring-up
MAX_DEG: float | None = 350.0   # TODO: calibrate after hardware bring-up

# AX-12A registers
_REG_CW_LIMIT  = 6
_REG_CCW_LIMIT = 8
_REG_TORQUE_EN = 24
_REG_SPEED     = 32

_DIR_CCW = 0
_DIR_CW  = 1 << 10

# =============================================================================
# CORNER SENSOR
# =============================================================================

_sensor_mgr = None


def _init_sensor() -> None:
    global _sensor_mgr
    try:
        from corner_sensors import CornerSensorManager
        mgr = CornerSensorManager(bus_num=1)
        if mgr.channel_has_sensor(SENSOR_CHANNEL):
            _sensor_mgr = mgr
            print(f"[turntable] Corner sensor ready on TCA ch {SENSOR_CHANNEL}.")
        else:
            print(f"[turntable] ⚠️  No AS5600 found on TCA ch {SENSOR_CHANNEL} — running open-loop.")
    except Exception as e:
        print(f"[turntable] Corner sensor unavailable ({e}) — running open-loop.")


def _read_sensor() -> dict | None:
    if _sensor_mgr is None:
        return None
    try:
        return _sensor_mgr.read_sensor(SENSOR_CHANNEL)
    except Exception as e:
        print(f"[turntable] Sensor read error: {e}")
        return None


def get_sensor_reading() -> dict | None:
    return _read_sensor()


def _at_limit(direction: str) -> bool:
    """
    Return True if moving in *direction* ('left'/'right') would violate the
    soft travel limits.  Returns False when the sensor is unavailable or
    limits are disabled.
    """
    if MIN_DEG is None or MAX_DEG is None:
        return False
    reading = _read_sensor()
    if reading is None:
        return False
    deg = CornerSensorManager.total_position(reading)
    if direction == "right" and deg >= MAX_DEG:
        return True
    if direction == "left" and deg <= MIN_DEG:
        return True
    return False

# =============================================================================
# STATE
# =============================================================================

_initialized:  bool = False
_pending_word: int  = -1
_last_word:    int  = -1
_lock      = threading.Lock()
_event     = threading.Event()
_stop_flag = False
_thread: threading.Thread = None  # type: ignore[assignment]

# =============================================================================
# BACKGROUND WRITER THREAD
# =============================================================================

def _writer() -> None:
    global _last_word
    while not _stop_flag:
        if not _event.wait(timeout=0.1):
            continue
        _event.clear()
        with _lock:
            word = _pending_word
        if word < 0 or word == _last_word:
            continue
        try:
            motor._write_word(SERVO_ID, _REG_SPEED, word)
            _last_word = word
        except Exception as e:
            print(f"[turntable] Serial error: {e}")

# =============================================================================
# LIFECYCLE
# =============================================================================

def init() -> None:
    global _initialized, _stop_flag, _thread

    if _initialized:
        return

    _stop_flag = False

    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 0); time.sleep(0.05)
    motor._write_word(SERVO_ID, _REG_CW_LIMIT,  0); time.sleep(0.02)
    motor._write_word(SERVO_ID, _REG_CCW_LIMIT, 0); time.sleep(0.02)
    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 1); time.sleep(0.05)
    motor._write_word(SERVO_ID, _REG_SPEED,     0)

    _thread = threading.Thread(target=_writer, daemon=True, name="turntable-writer")
    _thread.start()

    _init_sensor()

    _initialized = True
    print(f"✅ Turntable initialised (ID {SERVO_ID}, wheel mode).")


def shutdown() -> None:
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
    global _pending_word
    with _lock:
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
    _post_word(0)
    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)


def spin_left(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CCW | speed)
    servo_status.update(SERVO_ID, "LEFT", speed, real=_initialized)
    return {"direction": "left", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def spin_right(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CW | speed)
    servo_status.update(SERVO_ID, "RIGHT", speed, real=_initialized)
    return {"direction": "right", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def update(dx: int) -> str:
    """
    Main per-frame entry point.  Enforces soft travel limits when a corner
    sensor is available.  Never blocks.

    Args:
        dx: target_x - gripper_x  (from RobotController.generate_dx)
    """
    if not _initialized:
        speed = _speed_from_dx(abs(dx))
        if dx > DEAD_ZONE:
            servo_status.update(SERVO_ID, "RIGHT", speed, real=False)
            return f"TURNTABLE SIMULATED (dx={dx:+d})"
        if dx < -DEAD_ZONE:
            servo_status.update(SERVO_ID, "LEFT", speed, real=False)
            return f"TURNTABLE SIMULATED (dx={dx:+d})"
        servo_status.update(SERVO_ID, "STOP", 0, real=False)
        return f"TURNTABLE SIMULATED (dx={dx:+d})"

    speed = _speed_from_dx(abs(dx))

    if dx > DEAD_ZONE:
        if _at_limit("right"):
            stop()
            return f"TURNTABLE LIMIT RIGHT (dx={dx:+d}, pos≥{MAX_DEG})"
        spin_right(speed)
        return f"TURNTABLE RIGHT (dx={dx:+d}, speed={speed})"

    if dx < -DEAD_ZONE:
        if _at_limit("left"):
            stop()
            return f"TURNTABLE LIMIT LEFT  (dx={dx:+d}, pos≤{MIN_DEG})"
        spin_left(speed)
        return f"TURNTABLE LEFT  (dx={dx:+d}, speed={speed})"

    stop()
    return f"TURNTABLE STOP  (dx={dx:+d}, aligned)"