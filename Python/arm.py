"""
arm.py
======
Controls the arm servo (AX-12A, ID 5) in wheel / continuous-rotation mode.

Zero-point tracking  (see _homing_utils.py for full explanation)
-------------------
At init() the current sensor reading is stored as _zero_deg.  A
dead-reckoning accumulator (_dead_pos) tracks movement when no sensor
is present.  _home() drives back to zero using whichever is available.

Coordinate convention
---------------------
    dz > 0  → target is FAR    → extend  (forward)
    dz < 0  → target is CLOSE  → retract (backward)
    |dz| ≤ DEAD_ZONE  → aligned, stop
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

SERVO_ID  = 5
DEAD_ZONE = 25

SPEED_SLOW   = 150
SPEED_MEDIUM = 300
SPEED_FAST   = 500

THRESHOLD_SLOW   = 50
THRESHOLD_MEDIUM = 150

SENSOR_CHANNEL = 0

MIN_DEG: float | None = -45.0
MAX_DEG: float | None = 45.0

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
            print(f"[arm] Corner sensor ready on TCA ch {SENSOR_CHANNEL}.")
        else:
            print(f"[arm] ⚠️  No AS5600 found on TCA ch {SENSOR_CHANNEL} — running open-loop.")
    except Exception as e:
        print(f"[arm] Corner sensor unavailable ({e}) — running open-loop.")


def _read_sensor() -> dict | None:
    if _sensor_mgr is None:
        return None
    try:
        return _sensor_mgr.read_sensor(SENSOR_CHANNEL)
    except Exception as e:
        print(f"[arm] Sensor read error: {e}")
        return None


def get_sensor_reading() -> dict | None:
    return _read_sensor()


def _at_limit(direction: str) -> bool:
    if MIN_DEG is None or MAX_DEG is None:
        return False
    reading = _read_sensor()
    if reading is None:
        return False
    deg =  _sensor_mgr.total_position(reading)
    if direction == "forward" and deg >= MAX_DEG:
        return True
    if direction == "backward" and deg <= MIN_DEG:
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
            print(f"[arm] Serial error: {e}")

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

    _thread = threading.Thread(target=_writer, daemon=True, name="arm-writer")
    _thread.start()

    _init_sensor()

    reading = _read_sensor()
    if reading is not None:
        _zero_deg =  _sensor_mgr.total_position(reading)
        print(f"[arm] Zero point: {_zero_deg:.1f}° (sensor)")
    else:
        _zero_deg = None
        print("[arm] Zero point: dead-reckoning only")

    _dead_pos[0]     = 0.0
    _last_write_time = time.monotonic()
    _initialized     = True
    print(f"✅ Arm initialised (ID {SERVO_ID}, wheel mode).")


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
    print("🛑 Arm shut down.")

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


def _speed_from_dz(dz_abs: int) -> int:
    if dz_abs <= DEAD_ZONE:
        return 0
    if dz_abs <= THRESHOLD_SLOW:
        return SPEED_SLOW
    if dz_abs <= THRESHOLD_MEDIUM:
        return SPEED_MEDIUM
    return SPEED_FAST

# =============================================================================
# PUBLIC API
# =============================================================================

def stop() -> None:
    _post_word(0)
    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)


def move_forward(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CCW | speed)
    servo_status.update(SERVO_ID, "FORWARD", speed, real=_initialized)
    return {"direction": "forward", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def move_backward(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CW | speed)
    servo_status.update(SERVO_ID, "BACKWARD", speed, real=_initialized)
    return {"direction": "backward", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def update(dz: int) -> str:
    if not _initialized:
        speed = _speed_from_dz(abs(dz))
        if dz > DEAD_ZONE:
            servo_status.update(SERVO_ID, "FORWARD", speed, real=False)
            return f"ARM SIMULATED (dz={dz:+d})"
        if dz < -DEAD_ZONE:
            servo_status.update(SERVO_ID, "BACKWARD", speed, real=False)
            return f"ARM SIMULATED (dz={dz:+d})"
        servo_status.update(SERVO_ID, "STOP", 0, real=False)
        return f"ARM SIMULATED (dz={dz:+d})"

    speed = _speed_from_dz(abs(dz))

    if dz > DEAD_ZONE:
        if _at_limit("forward"):
            stop()
            return f"ARM LIMIT FORWARD  (dz={dz:+d}, pos≥{MAX_DEG})"
        move_forward(speed)
        return f"ARM FORWARD  (dz={dz:+d}, speed={speed})"

    if dz < -DEAD_ZONE:
        if _at_limit("backward"):
            stop()
            return f"ARM LIMIT BACKWARD (dz={dz:+d}, pos≤{MIN_DEG})"
        move_backward(speed)
        return f"ARM BACKWARD (dz={dz:+d}, speed={speed})"

    stop()
    return f"ARM STOP     (dz={dz:+d}, aligned)"

# =============================================================================
# HOMING  (called by motor.home_all())
# =============================================================================

def _home() -> None:
    """Retract arm to its zero (startup) position.  Blocks until complete."""
    if not _initialized:
        print("[arm] _home(): not initialised — skipping")
        return

    motor._write_word(SERVO_ID, _REG_SPEED, 0)

    if _zero_deg is not None:
        print(f"[arm] Homing with sensor → target {_zero_deg:.1f}°")
        ok = home_with_sensor(
            read_sensor_fn    = _read_sensor,
            total_position_fn =  _sensor_mgr.total_position,
            zero_deg          = _zero_deg,
            drive_positive_fn = lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CCW | s),
            drive_negative_fn = lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CW  | s),
            stop_fn           = lambda:   motor._write_word(SERVO_ID, _REG_SPEED, 0),
        )
        print(f"[arm] Homing {'complete' if ok else 'incomplete — sensor timeout'}.")
    else:
        print("[arm] Homing with dead-reckoning…")
        home_dead_reckoning(
            dead_pos_ref      = _dead_pos,
            drive_positive_fn = lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CCW | s),
            drive_negative_fn = lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CW  | s),
            stop_fn           = lambda:   motor._write_word(SERVO_ID, _REG_SPEED, 0),
        )
        print("[arm] Homing complete (dead-reckoning).")

    _dead_pos[0] = 0.0
    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)