"""
pivot.py
========
Controls the gripper-pivot servo (AX-12A, ID 2) in wheel /
continuous-rotation mode so the gripper mount can rotate up or down.

Architecture
------------
- Does NOT open serial/GPIO — motor.py owns those resources.
- Call motor.init() then pivot.init() once at startup.
- update(dp) is the per-frame entry point for autonomous use.
- rotate_up() / rotate_down() / stop() are used by manual_controller.
- All serial writes run on a background thread so they never block
  the vision/inference pipeline.

Coordinate convention (for autonomous use)
-------------------------------------------
    dp > 0  → pivot DOWN  (gripper nose tilts down)
    dp < 0  → pivot UP    (gripper nose tilts up)
    |dp| ≤ DEAD_ZONE  → aligned, stop

AX-12A wheel-mode  (MOVING_SPEED reg 32)
-----------------------------------------
    Bits 0-9  → speed magnitude  (0 = stop)
    Bit  10   → 0 = CCW (up)  /  1 = CW (down)

TODO (hardware bring-up):
  Verify which spin direction corresponds to up/down on the physical pivot
  and swap _DIR_CCW / _DIR_CW if needed.

Corner sensor (AS5600 via TCA9548A)
------------------------------------
SENSOR_CHANNEL is reserved but reads are currently STUBBED.

TODO:
  1. Set SENSOR_CHANNEL to the correct TCA channel.
  2. Define soft travel limits (MIN_DEG, MAX_DEG) to avoid over-rotation.
  3. Use _read_sensor() in update() to enforce those limits.
"""

import threading
import time

import motor

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

# AX-12A registers
_REG_CW_LIMIT  = 6
_REG_CCW_LIMIT = 8
_REG_TORQUE_EN = 24
_REG_SPEED     = 32

_DIR_CCW = 0          # up
_DIR_CW  = 1 << 10    # down

# =============================================================================
# CORNER SENSOR  (optional — gracefully absent on non-Pi or pre-wiring)
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
    """
    Return the latest pivot encoder reading, or None if unavailable.
    Dict keys: channel, raw, deg, laps
    TODO: use in update() to enforce soft travel limits.
    """
    if _sensor_mgr is None:
        return None
    try:
        return _sensor_mgr.read_sensor(SENSOR_CHANNEL)
    except Exception as e:
        print(f"[pivot] Sensor read error: {e}")
        return None


def get_sensor_reading() -> dict | None:
    """Public wrapper — returns latest pivot encoder reading or None."""
    return _read_sensor()

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
    """Switch servo to wheel mode and start background writer.  Idempotent."""
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
    """Stop motor and disable torque."""
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
    """Post a stop command (non-blocking)."""
    _post_word(0)


def rotate_up(speed: int = SPEED_MEDIUM) -> dict:
    """Rotate pivot up — CCW.  Non-blocking."""
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CCW | speed)
    return {"direction": "up", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def rotate_down(speed: int = SPEED_MEDIUM) -> dict:
    """Rotate pivot down — CW.  Non-blocking."""
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CW | speed)
    return {"direction": "down", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def update(dp: int) -> str:
    """
    Per-frame entry point for autonomous control.  Posts the appropriate
    speed word and returns a log string.  Never blocks.

    Args:
        dp: pivot error — positive = tilt down, negative = tilt up.

    TODO: call _read_sensor() here to enforce soft travel limits.
    """
    if not _initialized:
        return f"PIVOT SIMULATED (dp={dp:+d})"

    speed = _speed_from_dp(abs(dp))

    if dp > DEAD_ZONE:
        rotate_down(speed)
        return f"PIVOT DOWN (dp={dp:+d}, speed={speed})"

    if dp < -DEAD_ZONE:
        rotate_up(speed)
        return f"PIVOT UP   (dp={dp:+d}, speed={speed})"

    stop()
    return f"PIVOT STOP (dp={dp:+d}, aligned)"