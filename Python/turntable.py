"""
turntable.py - FIXED VERSION
============
Controls the turntable servo (AX-12A, ID 13) in wheel mode.
"""

import threading
import time

import motor
import servo_status
from _homing_utils import (
    home_with_sensor, home_dead_reckoning, accumulate, DR_HOME_SPEED
)

# tuning
SERVO_ID = 13
DEAD_ZONE = 25
SPEED_SLOW = 40
SPEED_MEDIUM = 80
SPEED_FAST = 160
THRESHOLD_SLOW = 50
THRESHOLD_MEDIUM = 150
SENSOR_CHANNEL = 1
MIN_DEG: float | None = -999999999
MAX_DEG: float | None = 999999999
SPEED_TO_DEG = 0.3

# AX-12A registers
_REG_CW_LIMIT = 6
_REG_CCW_LIMIT = 8
_REG_TORQUE_EN = 24
_REG_SPEED = 32
_DIR_CCW = 0
_DIR_CW = 1 << 10

_sensor_mgr = None

# --- state ---
_initialized: bool = False
_pending_word: int = -1
_last_word: int = -1
_lock = threading.Lock()
_event = threading.Event()
_stop_flag = False
_thread: threading.Thread = None
_zero_deg: float | None = None
_dead_pos: list = [0.0]
_last_write_time: float = 0.0
_force_write: bool = False  # 🔥 NEW: force write even if same word


def _init_sensor() -> None:
    global _sensor_mgr
    try:
        from corner_sensors import CornerSensorManager
        mgr = CornerSensorManager(bus_num=1)
        if mgr.channel_has_sensor(SENSOR_CHANNEL):
            _sensor_mgr = mgr
            print(f"[turntable] Corner sensor ready on TCA ch {SENSOR_CHANNEL}.")
        else:
            print(f"[turntable] !!! No AS5600 found on TCA ch {SENSOR_CHANNEL} — running open-loop.")
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


def get_dead_reckoning() -> dict:
    return {
        "accumulator": round(_dead_pos[0], 3),
        "estimated_deg": round(_dead_pos[0] * SPEED_TO_DEG, 2),
        "zero_deg": round(_zero_deg, 2) if _zero_deg is not None else None,
        "sensor_active": _sensor_mgr is not None,
        "min_deg": MIN_DEG,
        "max_deg": MAX_DEG,
    }


def _at_limit(direction: str) -> bool:
    if MIN_DEG is None or MAX_DEG is None:
        return False
    if _sensor_mgr is not None:
        reading = _read_sensor()
        if reading is not None:
            abs_deg = _sensor_mgr.total_position(reading)
            deg = abs_deg - (_zero_deg or 0)
            if direction == "right" and deg >= MAX_DEG:
                return True
            if direction == "left" and deg <= MIN_DEG:
                return True
            return False
    estimated_deg = _dead_pos[0] * SPEED_TO_DEG
    if direction == "right" and estimated_deg >= MAX_DEG:
        return True
    if direction == "left" and estimated_deg <= MIN_DEG:
        return True
    return False


# 🔥 FIXED WRITER THREAD
def _writer() -> None:
    global _last_word, _last_write_time, _force_write

    while not _stop_flag:
        time.sleep(0.05)

        now = time.monotonic()
        dt = now - _last_write_time if _last_write_time else 0.0
        _last_write_time = now

        # Update dead-reckoning with current speed
        if _last_word >= 0 and dt > 0:
            accumulate(_dead_pos, _last_word, dt)

        _event.clear()

        with _lock:
            word = _pending_word
            force = _force_write
            _force_write = False  # Reset after reading

        # 🔥 CRITICAL FIX: Skip only if word is invalid
        if word < 0:
            continue

        # 🔥 FIX: Always write if forced, or if word differs from last
        if force or word != _last_word:
            try:
                motor._write_word(SERVO_ID, _REG_SPEED, word)
                _last_word = word
                if force:
                    print(f"[turntable] FORCED WRITE: {word}")
            except Exception as e:
                print(f"[turntable] serial error: {e}")
        # else: word is same as last, skip to avoid spam


def _post_word(word: int, force: bool = False) -> None:
    """Post a speed command to the writer thread.

    Args:
        word: Speed command to send
        force: If True, force write even if same as last word
    """
    global _pending_word, _force_write

    # Wait for lock to clear (but don't drop!)
    timeout = 5.0
    start = time.monotonic()
    while motor.is_locked() and (time.monotonic() - start) < timeout:
        time.sleep(0.01)

    if motor.is_locked():
        print(f"[turntable] WARNING: lock still active after {timeout}s - forcing write anyway")

    with _lock:
        _pending_word = word
        if force:
            _force_write = True
    _event.set()


def init() -> None:
    global _initialized, _stop_flag, _thread, _zero_deg, _last_write_time, _last_word

    if _initialized:
        return

    _stop_flag = False

    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 0)
    time.sleep(0.05)
    motor._write_word(SERVO_ID, _REG_CW_LIMIT, 0)
    time.sleep(0.02)
    motor._write_word(SERVO_ID, _REG_CCW_LIMIT, 0)
    time.sleep(0.02)
    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 1)
    time.sleep(0.05)
    motor._write_word(SERVO_ID, _REG_SPEED, 0)
    _last_word = -1

    _thread = threading.Thread(target=_writer, daemon=True, name="turntable-writer")
    _thread.start()

    _init_sensor()

    reading = _read_sensor()
    if reading is not None:
        _zero_deg = _sensor_mgr.total_position(reading)
        print(f"[turntable] zero point: {_zero_deg:.1f}° (sensor)")
    else:
        _zero_deg = None
        print("[turntable] zero point: dead-reckoning only")

    _dead_pos[0] = 0.0
    _last_write_time = time.monotonic()
    _initialized = True
    print(f"✅ turntable initialised (ID {SERVO_ID}, wheel mode).")


def shutdown() -> None:
    global _initialized, _stop_flag, _last_word

    if not _initialized:
        return

    _stop_flag = True
    _event.set()
    if _thread is not None:
        _thread.join(timeout=1.0)

    try:
        motor._write_word(SERVO_ID, _REG_SPEED, 0)
        motor._write_word(SERVO_ID, _REG_TORQUE_EN, 0)
    except Exception:
        pass

    _last_word = -1
    _initialized = False
    print("🛑 turntable shut down.")


def _speed_from_dx(dx_abs: int) -> int:
    if dx_abs <= DEAD_ZONE:
        return 0
    if dx_abs <= THRESHOLD_SLOW:
        return SPEED_SLOW
    if dx_abs <= THRESHOLD_MEDIUM:
        return SPEED_MEDIUM
    return SPEED_FAST


def stop() -> None:
    _post_word(0, force=True)  # 🔥 Force stop to override any stale command
    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)


def spin_left(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    word = _DIR_CCW | speed
    _post_word(word, force=True)  # 🔥 Force write to override stale homing command
    servo_status.update(SERVO_ID, "LEFT", speed, real=_initialized)
    print(f"[turntable] spin_left: speed={speed}, word={word}")  # 🔥 DEBUG
    return {"direction": "left", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def spin_right(speed: int = SPEED_MEDIUM) -> dict:
    speed = max(0, min(1023, speed))
    word = _DIR_CW | speed
    _post_word(word, force=True)  # 🔥 Force write to override stale homing command
    servo_status.update(SERVO_ID, "RIGHT", speed, real=_initialized)
    print(f"[turntable] spin_right: speed={speed}, word={word}")  # 🔥 DEBUG
    return {"direction": "right", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def update(dx: int) -> str:
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


# homing (called by motor.home_all())
def _home() -> None:
    """Drive turntable back to its zero point. Blocks until complete."""
    global _last_word, _force_write

    if not _initialized:
        print("[turntable] _home(): not initialised — skipping")
        return

    # Stop current movement
    motor._write_word(SERVO_ID, _REG_SPEED, 0)
    _last_word = -1  # Reset bookkeeping
    _force_write = True  # Next write will be forced

    if _zero_deg is not None:
        print(f"[turntable] homing with sensor → target {_zero_deg:.1f}°")
        ok = home_with_sensor(
            read_sensor_fn=_read_sensor,
            total_position_fn=_sensor_mgr.total_position,
            zero_deg=_zero_deg,
            drive_positive_fn=lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CCW | s),
            drive_negative_fn=lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CW | s),
            stop_fn=lambda: motor._write_word(SERVO_ID, _REG_SPEED, 0),
        )
        if ok:
            print("[turntable] homing complete (sensor).")
        else:
            print("[turntable] homing incomplete — sensor timeout.")
    else:
        print("[turntable] homing with dead-reckoning…")
        home_dead_reckoning(
            dead_pos_ref=_dead_pos,
            drive_positive_fn=lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CCW | s),
            drive_negative_fn=lambda s: motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CW | s),
            stop_fn=lambda: motor._write_word(SERVO_ID, _REG_SPEED, 0),
        )
        print("[turntable] homing complete (dead-reckoning).")

    _dead_pos[0] = 0.0
    # Reset bookkeeping
    _last_word = -1
    _force_write = True  # Next write will be forced

    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)


def home_physical() -> None:
    """Drive turntable to its physical zero position by hitting the end stop."""
    global _last_word, _zero_deg, _force_write

    if not _initialized:
        print("[turntable] home_physical(): not initialised — skipping")
        return

    print("[turntable] physical homing - moving to end stop...")

    motor._write_word(SERVO_ID, _REG_SPEED, 0)
    _last_word = -1
    _force_write = True
    time.sleep(0.1)

    motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CW | SPEED_MEDIUM)
    time.sleep(3.0)
    motor._write_word(SERVO_ID, _REG_SPEED, 0)
    time.sleep(0.3)

    motor._write_word(SERVO_ID, _REG_SPEED, _DIR_CCW | SPEED_SLOW)
    time.sleep(0.3)
    motor._write_word(SERVO_ID, _REG_SPEED, 0)

    _dead_pos[0] = 0.0

    if _sensor_mgr is not None:
        reading = _read_sensor()
        if reading is not None:
            _zero_deg = _sensor_mgr.total_position(reading)
            print(f"[turntable] physical home set to {_zero_deg:.1f}°")

    _last_word = -1
    _force_write = True
    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)
    print("[turntable] physical homing complete")


#  DEBUG: check if turntable is responding
def test_move(duration: float = 1.0) -> None:
    """Test if turntable moves - call this from console."""
    print(f"[turntable] TEST: spinning left for {duration}s")
    spin_left(SPEED_MEDIUM)
    time.sleep(duration)
    stop()
    print("[turntable] TEST: complete")