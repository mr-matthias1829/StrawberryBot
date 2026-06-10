"""
gripper.py
==========
Controls the gripper servo (AX-12A, ID 8) in wheel / continuous-rotation mode.

State gate
----------
The gripper tracks whether it is currently OPEN or CLOSED (GRIPPED).
Sending the same command twice in a row is silently ignored.

_home()
-------
Called by motor.home_all().  Ensures the gripper is OPEN (the safe
resting state).  If it is already open, this is a no-op.
Blocks until the open command completes.
"""

import threading
import time

import motor
import servo_status

# =============================================================================
# TUNING
# =============================================================================

SERVO_ID  = 8

GRIP_TIME = 3.1
OPEN_TIME = 3.1
SPEED     = 1023

SENSOR_CHANNEL = 3

_REG_CW_LIMIT  = 6
_REG_CCW_LIMIT = 8
_REG_TORQUE_EN = 24
_REG_SPEED     = 32

_SPEED_GRIP = SPEED
_SPEED_OPEN = 1024 + SPEED
_SPEED_STOP = 0

# =============================================================================
# CORNER SENSOR  (optional)
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
    if _sensor_mgr is None:
        return None
    try:
        return _sensor_mgr.read_sensor(SENSOR_CHANNEL)
    except Exception as e:
        print(f"[gripper] Sensor read error: {e}")
        return None


def get_sensor_reading() -> dict | None:
    return _read_sensor()

# =============================================================================
# STATE
# =============================================================================

_initialized:    bool = False
_state:          str  = "OPEN"
_gripper_open:   bool = True
_pending_action: str  = ""
_lock  = threading.Lock()
_event = threading.Event()
_stop_flag = False
_thread: threading.Thread = None  # type: ignore[assignment]

# =============================================================================
# BACKGROUND WORKER THREAD
# =============================================================================

def _worker() -> None:
    global _state, _gripper_open, _pending_action

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
                    _state       = "GRIPPED"
                    _gripper_open = False
                servo_status.update(SERVO_ID, "GRIPPED", 0, real=_initialized)
                print("[gripper] State → GRIPPED")
            elif action == "open":
                _execute_open()
                with _lock:
                    _state       = "OPEN"
                    _gripper_open = True
                servo_status.update(SERVO_ID, "OPEN", 0, real=_initialized)
                print("[gripper] State → OPEN")
        except Exception as e:
            print(f"[gripper] Error during {action}: {e}")
            with _lock:
                _state       = "OPEN"
                _gripper_open = True
            servo_status.update(SERVO_ID, "OPEN", 0, real=_initialized)
        finally:
            with _lock:
                _pending_action = ""


def _execute_grip() -> None:
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
    global _initialized, _stop_flag, _thread, _state, _gripper_open

    if _initialized:
        return

    _stop_flag    = False
    _state        = "OPEN"
    _gripper_open = True

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
    print(f"✅ Gripper initialised (ID {SERVO_ID}, wheel mode). State: OPEN")


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
    print("🛑 Gripper shut down.")

# =============================================================================
# PUBLIC API
# =============================================================================

def get_state() -> str:
    with _lock:
        return _state


def is_open() -> bool:
    with _lock:
        return _gripper_open


def is_gripped() -> bool:
    with _lock:
        return not _gripper_open and _state == "GRIPPED"


def command(action: str) -> dict:
    """
    Command the gripper.  Non-blocking.
    Ignored while the post-home lockout (motor.is_locked()) is active.
    """
    global _pending_action

    action = action.lower()
    if action not in ("grip", "open"):
        return {"action": action, "status": "invalid"}

    if motor.is_locked():
        return {"action": action, "status": "locked", "reason": "post_home_lockout"}

    if not _initialized:
        sim_status = "GRIPPED" if action == "grip" else "OPEN"
        servo_status.update(SERVO_ID, sim_status, 0, real=False)
        return {"action": action, "status": "simulated"}

    with _lock:
        current    = _state
        is_open_now = _gripper_open

    if current == "BUSY":
        return {"action": action, "status": "busy", "reason": "command_in_progress"}

    if action == "grip" and not is_open_now:
        return {"action": action, "status": "ignored", "reason": "already_gripped"}
    if action == "open" and is_open_now:
        return {"action": action, "status": "ignored", "reason": "already_open"}

    with _lock:
        _pending_action = action
    _event.set()

    return {"action": action, "status": "ok"}


def grip() -> dict:
    return command("grip")


def open_gripper() -> dict:
    return command("open")

# =============================================================================
# HOMING  (called by motor.home_all())
# =============================================================================

def _home() -> None:
    """
    Ensure the gripper is OPEN.  Blocks until the open command completes
    (or is already confirmed open).
    """
    global _state, _gripper_open

    if not _initialized:
        print("[gripper] _home(): not initialised — skipping")
        return

    with _lock:
        already_open = _gripper_open
        busy         = (_state == "BUSY")

    if busy:
        # Wait for the current action to finish before we proceed
        print("[gripper] _home(): waiting for current action to complete…")
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            with _lock:
                if _state != "BUSY":
                    break
            time.sleep(0.1)

    with _lock:
        already_open = _gripper_open

    if already_open:
        print("[gripper] _home(): already OPEN — nothing to do.")
        return

    print("[gripper] _home(): opening gripper…")
    _execute_open()
    with _lock:
        _state       = "OPEN"
        _gripper_open = True
    servo_status.update(SERVO_ID, "OPEN", 0, real=_initialized)
    print("[gripper] _home(): OPEN.")


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