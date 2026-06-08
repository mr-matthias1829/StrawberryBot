"""
lift.py
=======
Controls the lift mechanism using two AX-12A servos (IDs 3 and 4) in
wheel / continuous-rotation mode so the arm can move up or down to track
a strawberry on the Y-axis.

Architecture
------------
- Does NOT open serial/GPIO — motor.py owns those resources.
- Call motor.init() then lift.init() once at startup.
- update(dy) is the only call needed per frame from RobotController.
- All serial writes run on a background thread so they never block
  the vision/inference pipeline.

Mechanical note
---------------
Both servos must spin in the SAME direction to move the lift:

    Lift UP   → servo 3 CCW  + servo 4 CCW
    Lift DOWN → servo 3 CW   + servo 4 CW

Corner sensor (AS5600 via TCA9548A)
------------------------------------
The CornerSensorManager is imported and instantiated at init() time.
Two sensor channels are reserved — one per servo side if available.
Reads are currently STUBBED.

TODO (hardware bring-up):
  1. Set SENSOR_CHANNEL_A / _B to the correct TCA channels.
  2. Define travel limits in degrees (MIN_DEG, MAX_DEG).
  3. Call _read_sensor() inside update() or a dedicated safety thread
     to stop the lift before it hits a hard end-stop.

AX-12A wheel-mode  (MOVING_SPEED reg 32)
-----------------------------------------
    Bits 0-9  → speed magnitude  (0 = stop)
    Bit  10   → 0 = CCW  /  1 = CW

Coordinate convention (matches robot_controller.py)
----------------------------------------------------
    dy > 0  → target is BELOW  gripper → move DOWN
    dy < 0  → target is ABOVE  gripper → move UP
    |dy| ≤ DEAD_ZONE  → aligned, stop
"""

import threading
import time

try:
    from . import motor
except ImportError:
    import motor

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

# TCA9548A channels wired to the lift AS5600 encoders.
# TODO: set to the correct channels once hardware is confirmed.
SENSOR_CHANNEL_A = 1   # servo 3 side
SENSOR_CHANNEL_B = 2   # servo 4 side

# AX-12A registers
_REG_CW_LIMIT  = 6
_REG_CCW_LIMIT = 8
_REG_TORQUE_EN = 24
_REG_SPEED     = 32

_DIR_CCW = 0
_DIR_CW  = 1 << 10

# =============================================================================
# CORNER SENSORS  (optional — gracefully absent on non-Pi or pre-wiring)
# =============================================================================

_sensor_mgr = None

def _init_sensors() -> None:
    global _sensor_mgr
    try:
        from ..corner_sensors import CornerSensorManager
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
    """
    Return the latest reading for one lift encoder, or None if unavailable.

    Dict keys: channel, raw, deg, laps
    TODO: use in update() to enforce soft travel limits (MIN_DEG / MAX_DEG).
    """
    if _sensor_mgr is None or not _sensor_mgr.channel_has_sensor(channel):
        return None
    try:
        return _sensor_mgr.read_sensor(channel)
    except Exception as e:
        print(f"[lift] Sensor ch{channel} read error: {e}")
        return None

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
    """Switch both servos to wheel mode and start background writer.  Idempotent."""
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
    """Stop motors and disable torque."""
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
    """Post a stop command to both servos (non-blocking)."""
    _post_words(0, 0)


def move_up(speed: int = SPEED_MEDIUM) -> dict:
    """Move lift UP — both servos CCW.  Non-blocking."""
    speed = max(0, min(1023, speed))
    _post_words(_DIR_CCW | speed, _DIR_CCW | speed)
    return {"direction": "up", "servo_ids": [SERVO_ID_A, SERVO_ID_B],
            "speed": speed, "status": "ok"}


def move_down(speed: int = SPEED_MEDIUM) -> dict:
    """Move lift DOWN — both servos CW.  Non-blocking."""
    speed = max(0, min(1023, speed))
    _post_words(_DIR_CW | speed, _DIR_CW | speed)
    return {"direction": "down", "servo_ids": [SERVO_ID_A, SERVO_ID_B],
            "speed": speed, "status": "ok"}


def update(dy: int) -> str:
    """
    Main per-frame entry point.  Posts the appropriate speed words and returns
    a log string.  Never blocks.

    Args:
        dy: target_y - gripper_y  (from RobotController.generate_dy)
            dy > 0 → target below  → move DOWN
            dy < 0 → target above  → move UP

    TODO: call _read_sensor(SENSOR_CHANNEL_A) here to check position against
    soft travel limits before commanding movement.
    """
    if not _initialized:
        return f"LIFT SIMULATED (dy={dy:+d})"

    speed = _speed_from_dy(abs(dy))

    if dy > DEAD_ZONE:
        move_down(speed)
        return f"LIFT DOWN (dy={dy:+d}, speed={speed})"

    if dy < -DEAD_ZONE:
        move_up(speed)
        return f"LIFT UP   (dy={dy:+d}, speed={speed})"

    stop()
    return f"LIFT STOP (dy={dy:+d}, aligned)"