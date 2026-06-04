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
- All serial writes run on a background thread so they never block the
  vision/inference pipeline.

Corner sensor (AS5600 via TCA9548A, TCA channel: see SENSOR_CHANNEL)
---------------------------------------------------------------------
The CornerSensorManager is imported and instantiated at init() time.
Reads are currently STUBBED — the sensor is present in the object but its
output is not yet wired into any control logic.

TODO (hardware bring-up):
  1. Set SENSOR_CHANNEL to the correct TCA multiplexer channel.
  2. Decide what "zero" means for the turntable (call sensor.reset_laps()).
  3. Use sensor.read_sensor(SENSOR_CHANNEL) in update() or a separate
     feedback loop to implement position limits / closed-loop control.

AX-12A wheel-mode register layout  (MOVING_SPEED reg 32)
---------------------------------------------------------
    Bits 0-9  → speed magnitude  (0 = stop)
    Bit  10   → 0 = CCW (LEFT)  /  1 = CW (RIGHT)

Wheel mode is activated by setting both angle-limit registers to 0.

Coordinate convention (matches robot_controller.py)
----------------------------------------------------
    dx > 0  → target is RIGHT of gripper → spin RIGHT (CW)
    dx < 0  → target is LEFT  of gripper → spin LEFT  (CCW)
    |dx| ≤ DEAD_ZONE  → aligned, stop
"""

import threading
import time

import motor

# =============================================================================
# TUNING
# =============================================================================

SERVO_ID  = 13
DEAD_ZONE = 25          # px — mirrors X_THRESHOLD in robot_controller.py

SPEED_SLOW   = 150
SPEED_MEDIUM = 300
SPEED_FAST   = 500

THRESHOLD_SLOW   = 50
THRESHOLD_MEDIUM = 150

# TCA9548A channel wired to the turntable AS5600 encoder.
# TODO: set to the correct channel once hardware is confirmed.
SENSOR_CHANNEL = 0

# AX-12A registers
_REG_CW_LIMIT  = 6
_REG_CCW_LIMIT = 8
_REG_TORQUE_EN = 24
_REG_SPEED     = 32

_DIR_CCW = 0
_DIR_CW  = 1 << 10

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
            print(f"[turntable] Corner sensor ready on TCA ch {SENSOR_CHANNEL}.")
        else:
            print(f"[turntable] ⚠️  No AS5600 found on TCA ch {SENSOR_CHANNEL} — running open-loop.")
    except Exception as e:
        print(f"[turntable] Corner sensor unavailable ({e}) — running open-loop.")


def _read_sensor() -> dict | None:
    """
    Return the latest sensor reading dict, or None if unavailable.

    Dict keys: channel, raw, deg, laps
    TODO: use this in update() for closed-loop / limit enforcement.
    """
    if _sensor_mgr is None:
        return None
    try:
        return _sensor_mgr.read_sensor(SENSOR_CHANNEL)
    except Exception as e:
        print(f"[turntable] Sensor read error: {e}")
        return None

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

    _thread = threading.Thread(target=_writer, daemon=True, name="turntable-writer")
    _thread.start()

    _init_sensor()

    _initialized = True
    print(f"✅ Turntable initialised (ID {SERVO_ID}, wheel mode).")


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
    """Post a stop command (non-blocking)."""
    _post_word(0)


def spin_left(speed: int = SPEED_MEDIUM) -> dict:
    """Spin CCW — gripper moves LEFT.  Non-blocking."""
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CCW | speed)
    return {"direction": "left", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def spin_right(speed: int = SPEED_MEDIUM) -> dict:
    """Spin CW — gripper moves RIGHT.  Non-blocking."""
    speed = max(0, min(1023, speed))
    _post_word(_DIR_CW | speed)
    return {"direction": "right", "servo_id": SERVO_ID, "speed": speed, "status": "ok"}


def update(dx: int) -> str:
    """
    Main per-frame entry point.  Posts the appropriate speed word and returns
    a log string.  Never blocks.

    Args:
        dx: target_x - gripper_x  (from RobotController.generate_dx)

    TODO: call _read_sensor() here to enforce soft travel limits once the
    encoder zero-point and degree budget have been defined.
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