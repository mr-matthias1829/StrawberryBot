"""
pivot.py
========
Controls the gripper-pivot servo (AX-12A, ID 2) in wheel /
continuous-rotation mode.

Travel limits
-------------
MIN_DEG / MAX_DEG are enforced in update() when a corner sensor is available.
Rotating down increases the encoder reading; rotating up decreases it.

Coordinate convention
---------------------
    dp > 0  → pivot DOWN  (gripper nose tilts down)
    dp < 0  → pivot UP    (gripper nose tilts up)
    |dp| ≤ DEAD_ZONE  → aligned, stop
"""

import threading
import time

import motor
import servo_status
from corner_sensors import CornerSensorManager

# =============================================================================
# TUNING
# =============================================================================

SERVO_ID  = 2
DEAD_ZONE = 25

SPEED_SLOW   = 150
SPEED_MEDIUM = 300
SPEED_FAST   = 500

THRESHOLD_SLOW   = 50
THRESHOLD_MEDIUM = 150

SENSOR_CHANNEL = 5   # TODO: set to correct TCA channel

# Soft travel limits in degrees.
# Set both to None to disable limit enforcement.
MIN_DEG: float | None = 10.0   # fully UP   — TODO: calibrate
MAX_DEG: float | None = 350.0  # fully DOWN — TODO: calibrate

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
            print(f"[pivot] Corner sensor ready on TCA ch {SENSOR_CHANNEL}.")
        else:
            print(f"[pivot] ⚠️  No AS5600 found on TCA ch {SENSOR_CHANNEL} — running open-loop.")
    except Exception as e:
        print(f"[pivot] Corner sensor unavailable ({e}) — running open-loop.")


def _read_sensor() -> dict | None:
    if _sensor_mgr is None:
        return None
    try:
        return _sensor_mgr.read_sensor(SENSOR_CHANNEL)
    except Exception as e:
        print(f"[pivot] Sensor read error: {e}")
        return None


def get_sensor_reading() -> dict | None:
    return _read_sensor()


def _at_limit(direction: str) -> bool:
    """Return True if moving in *direction* ('up'/'down') would hit a limit."""
    if MIN_DEG is None or MAX_DEG is None:
        return False
    reading = _read_sensor()
    if reading is None:
        return False
    deg = CornerSensorManager.total_position(reading)
    if direction == "down" and deg >= MAX_DEG:
        return True
    if direction == "up" and deg <= MIN_DEG:
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
            print(f"[pivot] Serial error: {e}")

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

    _thread = threading.Thread(target=_writer, daemon=True, name="pivot-writer")
    _thread.start()

    _init_sensor()

    _initialized = True
    print(f"✅ Pivot initialised (ID {SERVO_ID}, wheel mode).")


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
    print("🛑 Pivot shut down.")

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _post_word(word: int) -> None:
    global _pending_word
    with _lock:
        _pending_word = word
    _event.set()


def _speed_from_dp(dp_abs: int) -> int:
    if dp_abs <= DEAD_ZONE:
        return 0
    if dp_abs <= THRESHOLD_SLOW:
        return SPEED_SLOW
    if dp_abs <= THRESHOLD_MEDIUM:
        return SPEED_MEDIUM
    return SPEED_FAST

# =============================================================================
# PUBLIC API
# =============================================================================

def stop() -> None:
    _post_word(0)
    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)


def rotate_up(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CCW | speed)
    servo_status.update(SERVO_ID, "UP", speed, real=_initialized)
    return {"direction": "up", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def rotate_down(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CW | speed)
    servo_status.update(SERVO_ID, "DOWN", speed, real=_initialized)
    return {"direction": "down", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def update(dp: int) -> str:
    """
    Per-frame entry point.  Enforces soft travel limits when a corner
    sensor is available.  Never blocks.

    Args:
        dp: pivot error — positive = tilt down, negative = tilt up.
    """
    if not _initialized:
        speed = _speed_from_dp(abs(dp))
        if dp > DEAD_ZONE:
            servo_status.update(SERVO_ID, "DOWN", speed, real=False)
            return f"PIVOT SIMULATED (dp={dp:+d})"
        if dp < -DEAD_ZONE:
            servo_status.update(SERVO_ID, "UP", speed, real=False)
            return f"PIVOT SIMULATED (dp={dp:+d})"
        servo_status.update(SERVO_ID, "STOP", 0, real=False)
        return f"PIVOT SIMULATED (dp={dp:+d})"

    speed = _speed_from_dp(abs(dp))

    if dp > DEAD_ZONE:
        if _at_limit("down"):
            stop()
            return f"PIVOT LIMIT DOWN (dp={dp:+d}, pos≥{MAX_DEG})"
        rotate_down(speed)
        return f"PIVOT DOWN (dp={dp:+d}, speed={speed})"

    if dp < -DEAD_ZONE:
        if _at_limit("up"):
            stop()
            return f"PIVOT LIMIT UP   (dp={dp:+d}, pos≤{MIN_DEG})"
        rotate_up(speed)
        return f"PIVOT UP   (dp={dp:+d}, speed={speed})"

    stop()
    return f"PIVOT STOP (dp={dp:+d}, aligned)"