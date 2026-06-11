"""
pivot.py
========
Controls the gripper-pivot servo (AX-12A, ID 2) in wheel /
continuous-rotation mode.

Zero-point tracking & _home() — see _homing_utils.py for full explanation.
"""

import threading
import time

import motor
import servo_status
from _homing_utils import (
    home_with_sensor, home_dead_reckoning, accumulate, DR_HOME_SPEED
)

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

SENSOR_CHANNEL = -1

MIN_DEG: float | None = -45.0
MAX_DEG: float | None = 45.0

# Dead-reckoning limit conversion: degrees per (speed-unit × second).
# Tune by running at a known speed for a known time and measuring degrees moved.
# Start conservative (small value = limits trigger sooner).
SPEED_TO_DEG = 0.3

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
    if MIN_DEG is None or MAX_DEG is None:
        return False

    # --- Sensor path ---
    if _sensor_mgr is not None:
        reading = _read_sensor()
        if reading is not None:
            deg = _sensor_mgr.total_position(reading)
            if direction == "down" and deg >= MAX_DEG:
                return True
            if direction == "up"   and deg <= MIN_DEG:
                return True
            return False
        # sensor present but read failed — fall through to DR

    # --- Dead-reckoning path ---
    estimated_deg = _dead_pos[0] * SPEED_TO_DEG
    if direction == "down" and estimated_deg >= MAX_DEG:
        return True
    if direction == "up"   and estimated_deg <= MIN_DEG:
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

# Zero-point tracking
_zero_deg:        float | None = None
_dead_pos:        list          = [0.0]
_last_write_time: float         = 0.0

# =============================================================================
# BACKGROUND WRITER THREAD
# =============================================================================

def _writer() -> None:
    global _last_word, _last_write_time
    while not _stop_flag:
        if not _event.wait(timeout=0.1):
            continue
        _event.clear()
        with _lock:
            word = _pending_word
        if word < 0 or word == _last_word:
            continue
        if motor.is_locked():
            continue
        try:
            now = time.monotonic()
            dt  = now - _last_write_time if _last_write_time else 0.0
            if _last_word >= 0 and dt > 0:
                accumulate(_dead_pos, _last_word, dt)
            motor._write_word(SERVO_ID, _REG_SPEED, word)
            _last_word       = word
            _last_write_time = now
        except Exception as e:
            print(f"[pivot] Serial error: {e}")

# =============================================================================
# LIFECYCLE
# =============================================================================

def init() -> None:
    global _initialized, _stop_flag, _thread, _zero_deg, _last_write_time

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

    reading = _read_sensor()
    if reading is not None:
        _zero_deg =  _sensor_mgr.total_position(reading)
        print(f"[pivot] Zero point: {_zero_deg:.1f}° (sensor)")
    else:
        _zero_deg = None
        print("[pivot] Zero point: dead-reckoning only")

    _dead_pos[0]     = 0.0
    _last_write_time = time.monotonic()
    _initialized     = True
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
    if motor.is_locked():
        return
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

# =============================================================================
# HOMING  (called by motor.home_all())
# =============================================================================

def _home() -> None:
    """Drive pivot to its zero (neutral/startup) position.  Blocks until complete."""
    if not _initialized:
        print("[pivot] _home(): not initialised — skipping")
        return

    motor._write_word(SERVO_ID, _REG_SPEED, 0)

    if _zero_deg is not None:
        print(f"[pivot] Homing with sensor → target {_zero_deg:.1f}°")
        ok = home_with_sensor(
            read_sensor_fn    = _read_sensor,
            total_position_fn =  _sensor_mgr.total_position,
            zero_deg          = _zero_deg,
            drive_positive_fn = lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CCW | s),
            drive_negative_fn = lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CW  | s),
            stop_fn           = lambda:   motor._write_word(SERVO_ID, _REG_SPEED, 0),
        )
        print(f"[pivot] Homing {'complete' if ok else 'incomplete — sensor timeout'}.")
    else:
        print("[pivot] Homing with dead-reckoning…")
        home_dead_reckoning(
            dead_pos_ref      = _dead_pos,
            drive_positive_fn = lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CCW | s),
            drive_negative_fn = lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CW  | s),
            stop_fn           = lambda:   motor._write_word(SERVO_ID, _REG_SPEED, 0),
        )
        print("[pivot] Homing complete (dead-reckoning).")

    _dead_pos[0] = 0.0
    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)