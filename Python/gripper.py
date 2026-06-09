"""
gripper.py
==========
Controls the gripper servo (AX-12A, ID 8) in wheel / continuous-rotation mode.

Architecture
------------
- Does NOT open serial/GPIO — motor.py owns those resources.
- Call motor.init() then gripper.init() once at startup.
- grip() / open_gripper() fire async commands via a background thread.
- All serial writes run on that thread so they never block the pipeline.

State tracking & safety
------------------------
- Prevents overlapping commands (busy guard).
- Prevents consecutive identical commands (no "open → open").
- Current state: "OPEN", "GRIPPED", or "BUSY".

Corner sensor (AS5600 via TCA9548A)
------------------------------------
The CornerSensorManager is imported and instantiated at init() time.
The sensor is present in the object but its output is currently STUBBED.

TODO (hardware bring-up):
  1. Set SENSOR_CHANNEL to the correct TCA channel.
  2. Use _read_sensor() inside _execute_grip() / _execute_open() to detect
     stall (encoder stopped moving) and stop early instead of relying on
     fixed GRIP_TIME / OPEN_TIME constants.

Calibration
-----------
Put gripper in open position, then:
    python gripper.py grip   → tune GRIP_TIME
    python gripper.py open   → tune OPEN_TIME
"""

import threading
import time

import motor
import servo_status

# =============================================================================
# TUNING
# =============================================================================

SERVO_ID  = 8

GRIP_TIME = 2.5   # seconds to spin closed — tune after testing
OPEN_TIME = 2.5   # seconds to spin open   — tune after testing
SPEED     = 1023  # 0–1023

# TCA9548A channel wired to the gripper AS5600 encoder.
# TODO: set to the correct channel once hardware is confirmed.
SENSOR_CHANNEL = 3

# AX-12A registers
_REG_CW_LIMIT  = 6
_REG_CCW_LIMIT = 8
_REG_TORQUE_EN = 24
_REG_SPEED     = 32

_SPEED_GRIP = SPEED           # CCW — spin to close
_SPEED_OPEN = 1024 + SPEED    # CW  — spin to open
_SPEED_STOP = 0

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
            print(f"[gripper] Corner sensor ready on TCA ch {SENSOR_CHANNEL}.")
        else:
            print(f"[gripper] ⚠️  No AS5600 found on TCA ch {SENSOR_CHANNEL} — running timed mode.")
    except Exception as e:
        print(f"[gripper] Corner sensor unavailable ({e}) — running timed mode.")


def _read_sensor() -> dict | None:
    """
    Return the latest gripper encoder reading, or None if unavailable.

    Dict keys: channel, raw, deg, laps
    TODO: poll this inside _execute_grip() / _execute_open() in a loop and
    break early when the encoder stops moving (stall detection), then stop
    the motor.  This removes the dependency on fixed GRIP_TIME / OPEN_TIME.
    """
    if _sensor_mgr is None:
        return None
    try:
        return _sensor_mgr.read_sensor(SENSOR_CHANNEL)
    except Exception as e:
        print(f"[gripper] Sensor read error: {e}")
        return None

def get_sensor_reading() -> dict | None:
    """Public wrapper — returns latest arm encoder reading or None."""
    return _read_sensor()

# =============================================================================
# STATE
# =============================================================================

_initialized:  bool = False
_state:        str  = "OPEN"
_last_action:  str  = ""
_pending_action: str = ""
_lock  = threading.Lock()
_event = threading.Event()
_stop_flag = False
_thread: threading.Thread = None  # type: ignore[assignment]

# =============================================================================
# BACKGROUND WORKER THREAD
# =============================================================================

def _worker() -> None:
    global _state, _last_action

    while not _stop_flag:
        if not _event.wait(timeout=0.5):
            continue
        _event.clear()

        with _lock:
            action = _pending_action
            if not action:
                continue

        with _lock:
            _state = "BUSY"
        servo_status.update(SERVO_ID, "BUSY", SPEED, real=_initialized)

        try:
            if action == "grip":
                _execute_grip()
                with _lock:
                    _state = "GRIPPED"
                    _last_action = "grip"
                servo_status.update(SERVO_ID, "GRIPPED", 0, real=_initialized)
            elif action == "open":
                _execute_open()
                with _lock:
                    _state = "OPEN"
                    _last_action = "open"
                servo_status.update(SERVO_ID, "OPEN", 0, real=_initialized)
        except Exception as e:
            print(f"[gripper] Error during {action}: {e}")
            with _lock:
                _state = "OPEN"   # failsafe
            servo_status.update(SERVO_ID, "OPEN", 0, real=_initialized)


def _execute_grip() -> None:
    """Spin closed for GRIP_TIME seconds (or until stall — see TODO above)."""
    print(f"🤏 Gripper gripping ({GRIP_TIME}s)…")
    try:
        motor._write_word(SERVO_ID, _REG_SPEED, _SPEED_GRIP)
        time.sleep(GRIP_TIME)
        motor._write_word(SERVO_ID, _REG_SPEED, _SPEED_STOP)
        print("✅ Gripper gripped.")
    except Exception as e:
        print(f"❌ Grip failed: {e}")
        motor._write_word(SERVO_ID, _REG_SPEED, _SPEED_STOP)


def _execute_open() -> None:
    """Spin open for OPEN_TIME seconds (or until stall — see TODO above)."""
    print(f"✋ Gripper opening ({OPEN_TIME}s)…")
    try:
        motor._write_word(SERVO_ID, _REG_SPEED, _SPEED_OPEN)
        time.sleep(OPEN_TIME)
        motor._write_word(SERVO_ID, _REG_SPEED, _SPEED_STOP)
        print("✅ Gripper opened.")
    except Exception as e:
        print(f"❌ Open failed: {e}")
        motor._write_word(SERVO_ID, _REG_SPEED, _SPEED_STOP)

# =============================================================================
# LIFECYCLE
# =============================================================================

def init() -> None:
    """Setup servo in wheel mode and start background worker.  Idempotent."""
    global _initialized, _stop_flag, _thread, _state

    if _initialized:
        return

    _stop_flag = False
    _state     = "OPEN"

    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 0); time.sleep(0.05)
    motor._write_word(SERVO_ID, _REG_CW_LIMIT,  0); time.sleep(0.02)
    motor._write_word(SERVO_ID, _REG_CCW_LIMIT, 0); time.sleep(0.02)
    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 1); time.sleep(0.05)
    motor._write_word(SERVO_ID, _REG_SPEED,     0)

    _thread = threading.Thread(target=_worker, daemon=True, name="gripper-worker")
    _thread.start()

    _init_sensor()

    _initialized = True
    servo_status.update(SERVO_ID, "OPEN", 0, real=True)
    print(f"✅ Gripper initialised (ID {SERVO_ID}, wheel mode).")


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
    print("🛑 Gripper shut down.")

# =============================================================================
# PUBLIC API
# =============================================================================

def get_state() -> str:
    """Return current state: 'OPEN', 'GRIPPED', or 'BUSY'."""
    with _lock:
        return _state


def command(action: str) -> dict:
    """
    Command the gripper.  Non-blocking.

    Args:
        action: "grip" or "open"

    Returns:
        {"action": action, "status": "ok" | "ignored" | "busy" | "invalid"}
    """
    global _pending_action

    if not _initialized:
        servo_status.update(SERVO_ID, action.upper(), 0, real=False)
        return {"action": action, "status": "simulated"}

    action = action.lower()
    if action not in ("grip", "open"):
        return {"action": action, "status": "invalid"}

    with _lock:
        current_state = _state
        last          = _last_action

    if last == action:
        return {"action": action, "status": "ignored", "reason": "same_as_last"}

    if current_state == "BUSY":
        return {"action": action, "status": "busy", "reason": "command_in_progress"}

    with _lock:
        _pending_action = action
    _event.set()

    return {"action": action, "status": "ok"}


def grip() -> dict:
    """Request the gripper to close.  Non-blocking."""
    return command("grip")


def open_gripper() -> dict:
    """Request the gripper to open.  Non-blocking."""
    return command("open")


# =============================================================================
# STANDALONE / TEST ENTRYPOINT
# =============================================================================

if __name__ == "__main__":
    import sys

    action = sys.argv[1].lower() if len(sys.argv) > 1 else "grip"

    motor.init()
    try:
        init()
        result = grip() if action == "grip" else open_gripper()
        print(f"Command result: {result}")
        for _ in range(30):
            if get_state() != "BUSY":
                break
            time.sleep(0.1)
        print(f"Final state: {get_state()}")
    except KeyboardInterrupt:
        print("\n⏹ Interrupted.")
    finally:
        shutdown()
        motor.shutdown()