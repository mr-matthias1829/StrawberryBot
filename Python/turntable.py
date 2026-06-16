"""
turntable.py
============
Controls the turntable servo (AX-12A, ID 13) in wheel /
continuous-rotation mode so the robot can pan left or right to track
a strawberry on the X-axis.

Zero-point tracking
-------------------
At init() the current sensor reading (if available) is stored as
_zero_deg.  A dead-reckoning accumulator (_dead_pos) tracks movement
even when no sensor is present.

_home()
-------
Called by motor.home_all().  Drives back to the zero position using the
sensor if available, otherwise reverses the dead-reckoning accumulator.
Blocks until complete.

_last_word bookkeeping
-----------------------
The background writer thread (_writer) only sends a new _REG_SPEED word
to the servo when it differs from _last_word, to avoid spamming the
serial bus with redundant writes. Several functions (init, shutdown,
_home, home_physical) write _REG_SPEED directly, bypassing the writer
thread entirely. Whenever that happens, _last_word must be reset to -1
afterward — otherwise the writer thread's bookkeeping goes stale and it
can silently skip the very next legitimate move command if it happens to
match the stale value.

FIX (2024) — "silent drop" bug
-------------------------------
_post_word() used to return immediately when motor.is_locked(), causing
bin.py move commands to be silently discarded.  The writer thread also
skipped writes when locked.  This meant spin_left() / spin_right() calls
during a lockout period were lost forever.

The fix:
  1. _post_word() now BLOCKS (with a short sleep loop) until the lock
     clears instead of dropping the command.
  2. The writer thread no longer checks is_locked() — the lock is already
     handled upstream in _post_word(), so the thread just writes whatever
     word arrives.
  3. The _last_word de-duplicate check no longer blocks resending the same
     speed after a homing/reset cycle; only negative (invalid) words are
     skipped.
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

SENSOR_CHANNEL = 1  # TCA9548A channel wired to this encoder

MIN_DEG: float | None = -999999999
MAX_DEG: float | None = 999999999

# dead-reckoning limit conversion: degrees per (speed-unit * second).
SPEED_TO_DEG = 0.3  # untested, as it always has a sensor

# AX-12A registers
_REG_CW_LIMIT = 6
_REG_CCW_LIMIT = 8
_REG_TORQUE_EN = 24
_REG_SPEED = 32

_DIR_CCW = 0
_DIR_CW = 1 << 10

# How long _post_word() will wait for a lock to clear before giving up (s).
# Set to 0 to wait forever.
POST_WORD_LOCK_TIMEOUT_S: float = 15.0

# corner sensor

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
            print(f"[turntable] !!!  No AS5600 found on TCA ch {SENSOR_CHANNEL} — running open-loop.")
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
    """return dead-reckoning state for dashboard display."""
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

    # sensor path
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
        # sensor present but read failed — fall through to DR

    # dead-reckoning path
    estimated_deg = _dead_pos[0] * SPEED_TO_DEG
    if direction == "right" and estimated_deg >= MAX_DEG:
        return True
    if direction == "left" and estimated_deg <= MIN_DEG:
        return True
    return False


_initialized: bool = False
_pending_word: int = -1
_last_word: int = -1
_lock = threading.Lock()
_event = threading.Event()
_stop_flag = False
_thread: threading.Thread = None  # type: ignore[assignment]

# Zero-point tracking
_zero_deg: float | None = None
_dead_pos: list = [0.0]
_last_write_time: float = 0.0


# background writer thread

def _writer() -> None:
    global _last_word, _last_write_time
    while not _stop_flag:
        time.sleep(0.05)

        now = time.monotonic()
        dt = now - _last_write_time if _last_write_time else 0.0
        _last_write_time = now

        if _last_word >= 0 and dt > 0:
            accumulate(_dead_pos, _last_word, dt)

        _event.clear()
        with _lock:
            word = _pending_word

        # Skip invalid words.
        if word < 0:
            continue

        # Dedup: only write if the word changed.
        # _last_word is reset to -1 after every direct write (homing etc.),
        # so the first command after homing always goes through even if it
        # happens to match the previous speed.
        if word == _last_word:
            continue

        # Do NOT check motor.is_locked() here — already handled in _post_word().

        try:
            motor._write_word(SERVO_ID, _REG_SPEED, word)
            _last_word = word
        except Exception as e:
            print(f"[turntable] serial error: {e}")


# lifecycle

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
    # direct write above bypasses the writer thread's bookkeeping —
    # reset so the next _post_word() is never mistaken for a duplicate.
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

    # direct write above bypasses the writer thread's bookkeeping —
    # reset so a future re-init / writer restart doesn't inherit a stale value.
    _last_word = -1

    _initialized = False
    print("🛑 turntable shut down.")


def _post_word(word: int) -> None:
    """
    Queue a speed word for the writer thread.

    Blocks until the motor lock clears (instead of silently dropping the
    command), then queues the word.  POST_WORD_LOCK_TIMEOUT_S caps the wait.

    Bails out immediately (with a clear log line) when turntable.init() has
    not been called — without init() the writer thread is not running and
    nothing would ever read _pending_word.
    """
    global _pending_word

    if not _initialized:
        print(f"[turntable] _post_word({word}): turntable not initialised — call turntable.init() first!")
        return

    if motor.is_locked():
        print(f"[turntable] _post_word({word}): motor locked — waiting for lock to clear…")
        deadline = (
            time.monotonic() + POST_WORD_LOCK_TIMEOUT_S
            if POST_WORD_LOCK_TIMEOUT_S > 0
            else float("inf")
        )
        while motor.is_locked():
            if time.monotonic() >= deadline:
                print("[turntable] _post_word(): lock wait timed out — dropping command")
                return
            time.sleep(0.01)
        print("[turntable] _post_word(): lock cleared — queuing command")

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
    """drive turntable back to its zero point.  Blocks until complete."""
    global _last_word

    if not _initialized:
        print("[turntable] _home(): not initialised — skipping")
        return

    motor._write_word(SERVO_ID, _REG_SPEED, 0)

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
    # all the writes above (including inside home_with_sensor /
    # home_dead_reckoning) go straight to the servo, bypassing the writer
    # thread's bookkeeping — reset so the next _post_word() always lands.
    _last_word = -1
    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)


def home_physical() -> None:
    """
    drive turntable to its physical zero position by hitting the end stop.
    this is for the UI home button, NOT for bin placement.
    """
    global _last_word, _zero_deg

    if not _initialized:
        print("[turntable] home_physical(): not initialised — skipping")
        return

    print("[turntable] physical homing - moving to end stop...")

    motor._write_word(SERVO_ID, _REG_SPEED, 0)
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

    # every write above went straight to the servo, bypassing the writer
    # thread's bookkeeping — reset so the next _post_word() always lands.
    _last_word = -1

    servo_status.update(SERVO_ID, "STOP", 0, real=_initialized)
    print("[turntable] physical homing complete")