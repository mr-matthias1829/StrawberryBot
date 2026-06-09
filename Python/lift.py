"""
lift.py
=======
Controls the lift mechanism using two AX-12A servos (IDs 3 and 4) in
wheel / continuous-rotation mode so the arm can move up or down.

Travel limits
-------------
MIN_DEG / MAX_DEG apply to SENSOR_CHANNEL_A (the primary encoder).
If that sensor is unavailable the lift runs open-loop.

Coordinate convention
---------------------
    dy > 0  → target is BELOW  gripper → move DOWN
    dy < 0  → target is ABOVE  gripper → move UP
    |dy| ≤ DEAD_ZONE  → aligned, stop
"""

import threading
import time

import motor
import servo_status
from corner_sensors import CornerSensorManager

# =============================================================================
# TUNING
# =============================================================================

SERVO_ID_A = 3
SERVO_ID_B = 4

DEAD_ZONE = 25

SPEED_SLOW   = 150
SPEED_MEDIUM = 300
SPEED_FAST   = 500

THRESHOLD_SLOW   = 50
THRESHOLD_MEDIUM = 150

SENSOR_CHANNEL_A = 1   # TODO: set to correct TCA channel
SENSOR_CHANNEL_B = 2   # TODO: set to correct TCA channel

# Soft travel limits in degrees.  The UP direction decreases the encoder
# reading on the physical rig; adjust the convention if it's the other way.
# Set both to None to disable limit enforcement.
MIN_DEG: float | None = 5.0     # fully UP limit   — TODO: calibrate
MAX_DEG: float | None = 355.0   # fully DOWN limit  — TODO: calibrate

# AX-12A registers
_REG_CW_LIMIT  = 6
_REG_CCW_LIMIT = 8
_REG_TORQUE_EN = 24
_REG_SPEED     = 32

_DIR_CCW = 0
_DIR_CW  = 1 << 10

# =============================================================================
# CORNER SENSORS
# =============================================================================

_sensor_mgr = None


def _init_sensors() -> None:
    global _sensor_mgr
    try:
        from corner_sensors import CornerSensorManager
        mgr = CornerSensorManager(bus_num=1)
        found = [ch for ch in (SENSOR_CHANNEL_A, SENSOR_CHANNEL_B)
                 if mgr.channel_has_sensor(ch)]
        if found:
            _sensor_mgr = mgr
            print(f"[lift] Corner sensor(s) ready on TCA ch {found}.")
        else:
            print(f"[lift] ⚠️  No AS5600 found on TCA ch "
                  f"{SENSOR_CHANNEL_A}/{SENSOR_CHANNEL_B} — running open-loop.")
    except Exception as e:
        print(f"[lift] Corner sensor unavailable ({e}) — running open-loop.")


def _read_sensor(channel: int) -> dict | None:
    if _sensor_mgr is None or not _sensor_mgr.channel_has_sensor(channel):
        return None
    try:
        return _sensor_mgr.read_sensor(channel)
    except Exception as e:
        print(f"[lift] Sensor ch{channel} read error: {e}")
        return None


def get_sensor_readings() -> dict[str, dict | None]:
    return {
        "A": _read_sensor(SENSOR_CHANNEL_A),
        "B": _read_sensor(SENSOR_CHANNEL_B),
    }


def _at_limit(direction: str) -> bool:
    """
    Return True if moving in *direction* ('up'/'down') would violate the
    soft travel limits.  Uses channel A as the position reference.
    """
    if MIN_DEG is None or MAX_DEG is None:
        return False
    reading = _read_sensor(SENSOR_CHANNEL_A)
    if reading is None:
        return False
    deg = CornerSensorManager.total_position(reading)
    if direction == "up" and deg <= MIN_DEG:
        return True
    if direction == "down" and deg >= MAX_DEG:
        return True
    return False

# =============================================================================
# STATE
# =============================================================================

_initialized:    bool = False
_pending_word_a: int  = -1
_pending_word_b: int  = -1
_last_word_a:    int  = -1
_last_word_b:    int  = -1
_lock      = threading.Lock()
_event     = threading.Event()
_stop_flag = False
_thread: threading.Thread = None  # type: ignore[assignment]

# =============================================================================
# BACKGROUND WRITER THREAD
# =============================================================================

def _writer() -> None:
    global _last_word_a, _last_word_b
    while not _stop_flag:
        if not _event.wait(timeout=0.1):
            continue
        _event.clear()

        with _lock:
            word_a = _pending_word_a
            word_b = _pending_word_b

        if word_a >= 0 and word_a != _last_word_a:
            try:
                motor._write_word(SERVO_ID_A, _REG_SPEED, word_a)
                _last_word_a = word_a
            except Exception as e:
                print(f"[lift] Serial error (servo {SERVO_ID_A}): {e}")

        if word_b >= 0 and word_b != _last_word_b:
            try:
                motor._write_word(SERVO_ID_B, _REG_SPEED, word_b)
                _last_word_b = word_b
            except Exception as e:
                print(f"[lift] Serial error (servo {SERVO_ID_B}): {e}")

# =============================================================================
# LIFECYCLE
# =============================================================================

def init() -> None:
    global _initialized, _stop_flag, _thread

    if _initialized:
        return

    _stop_flag = False

    for sid in (SERVO_ID_A, SERVO_ID_B):
        motor._write_word(sid, _REG_TORQUE_EN, 0); time.sleep(0.05)
        motor._write_word(sid, _REG_CW_LIMIT,  0); time.sleep(0.02)
        motor._write_word(sid, _REG_CCW_LIMIT, 0); time.sleep(0.02)
        motor._write_word(sid, _REG_TORQUE_EN, 1); time.sleep(0.05)
        motor._write_word(sid, _REG_SPEED,     0)

    _thread = threading.Thread(target=_writer, daemon=True, name="lift-writer")
    _thread.start()

    _init_sensors()

    _initialized = True
    print(f"✅ Lift initialised (IDs {SERVO_ID_A} & {SERVO_ID_B}, wheel mode).")


def shutdown() -> None:
    global _initialized, _stop_flag

    if not _initialized:
        return

    _stop_flag = True
    _event.set()
    if _thread is not None:
        _thread.join(timeout=1.0)

    for sid in (SERVO_ID_A, SERVO_ID_B):
        try:
            motor._write_word(sid, _REG_SPEED,    0)
            motor._write_word(sid, _REG_TORQUE_EN, 0)
        except Exception:
            pass

    _initialized = False
    print("🛑 Lift shut down.")

# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _post_words(word_a: int, word_b: int) -> None:
    global _pending_word_a, _pending_word_b
    with _lock:
        _pending_word_a = word_a
        _pending_word_b = word_b
    _event.set()


def _speed_from_dy(dy_abs: int) -> int:
    if dy_abs <= DEAD_ZONE:
        return 0
    if dy_abs <= THRESHOLD_SLOW:
        return SPEED_SLOW
    if dy_abs <= THRESHOLD_MEDIUM:
        return SPEED_MEDIUM
    return SPEED_FAST

# =============================================================================
# PUBLIC API
# =============================================================================

def stop() -> None:
    _post_words(0, 0)
    servo_status.update(SERVO_ID_A, "STOP", 0, real=_initialized)
    servo_status.update(SERVO_ID_B, "STOP", 0, real=_initialized)


def move_up(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    _post_words(_DIR_CCW | speed, _DIR_CCW | speed)
    servo_status.update(SERVO_ID_A, "UP", speed, real=_initialized)
    servo_status.update(SERVO_ID_B, "UP", speed, real=_initialized)
    return {"direction": "up", "servo_ids": [SERVO_ID_A, SERVO_ID_B],
            "speed": speed, "status": "ok"}


def move_down(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    _post_words(_DIR_CW | speed, _DIR_CW | speed)
    servo_status.update(SERVO_ID_A, "DOWN", speed, real=_initialized)
    servo_status.update(SERVO_ID_B, "DOWN", speed, real=_initialized)
    return {"direction": "down", "servo_ids": [SERVO_ID_A, SERVO_ID_B],
            "speed": speed, "status": "ok"}


def update(dy: int) -> str:
    """
    Main per-frame entry point.  Enforces soft travel limits when a corner
    sensor is available.  Never blocks.

    Args:
        dy: target_y - gripper_y  (from RobotController.generate_dy)
            dy > 0 → target below → move DOWN
            dy < 0 → target above → move UP
    """
    if not _initialized:
        speed = _speed_from_dy(abs(dy))
        if dy > DEAD_ZONE:
            servo_status.update(SERVO_ID_A, "DOWN", speed, real=False)
            servo_status.update(SERVO_ID_B, "DOWN", speed, real=False)
            return f"LIFT SIMULATED (dy={dy:+d})"
        if dy < -DEAD_ZONE:
            servo_status.update(SERVO_ID_A, "UP", speed, real=False)
            servo_status.update(SERVO_ID_B, "UP", speed, real=False)
            return f"LIFT SIMULATED (dy={dy:+d})"
        servo_status.update(SERVO_ID_A, "STOP", 0, real=False)
        servo_status.update(SERVO_ID_B, "STOP", 0, real=False)
        return f"LIFT SIMULATED (dy={dy:+d})"

    speed = _speed_from_dy(abs(dy))

    if dy > DEAD_ZONE:
        if _at_limit("down"):
            stop()
            return f"LIFT LIMIT DOWN (dy={dy:+d}, pos≥{MAX_DEG})"
        move_down(speed)
        return f"LIFT DOWN (dy={dy:+d}, speed={speed})"

    if dy < -DEAD_ZONE:
        if _at_limit("up"):
            stop()
            return f"LIFT LIMIT UP   (dy={dy:+d}, pos≤{MIN_DEG})"
        move_up(speed)
        return f"LIFT UP   (dy={dy:+d}, speed={speed})"

    stop()
    return f"LIFT STOP (dy={dy:+d}, aligned)"