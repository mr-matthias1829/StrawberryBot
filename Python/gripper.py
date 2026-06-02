"""
gripper.py
==========
Controls the gripper servo (AX-12A, ID 8) in wheel / continuous-rotation mode.
The gripper can be commanded to open or grip, with state tracking and safety guards.

Architecture
------------
- Does NOT open serial/GPIO — motor.py owns those resources.
- Call motor.init() then gripper.init() once at startup.
- command(action) fires async commands via background thread.
- All serial writes run on a tiny background thread so they never block.

State tracking & safety
-----------------------
- Prevents overlapping commands (can't start a new one until previous finishes)
- Prevents consecutive identical commands (no "open → open" or "grip → grip")
- Current state is "OPEN", "GRIPPED", or "BUSY"

Kalibratie
----------
Zet gripper handmatig in open stand, dan:
  python gripper.py grip   → pas GRIP_TIME aan
  python gripper.py open   → pas OPEN_TIME aan

Standalone use:
  python gripper.py grip
  python gripper.py open
"""

import threading
import time

import motor

# =============================================================================
# TUNING
# =============================================================================

SERVO_ID = 8

GRIP_TIME = 2.5   # seconds to spin closed — tune after testing
OPEN_TIME = 2.5   # seconds to spin open   — tune after testing
SPEED     = 1023  # 0–1023

# AX-12A registers
_REG_CW_LIMIT   = 6
_REG_CCW_LIMIT  = 8
_REG_TORQUE_EN  = 24
_REG_SPEED      = 32

# Wheel mode speed values
_SPEED_GRIP = SPEED           # CCW (spin to grip/close)
_SPEED_OPEN = 1024 + SPEED    # CW  (spin to open)
_SPEED_STOP = 0

# =============================================================================
# STATE
# =============================================================================

_initialized:  bool = False
_state:        str  = "OPEN"          # "OPEN", "GRIPPED", or "BUSY"
_last_action:  str  = ""              # to prevent repetition
_lock = threading.Lock()
_event = threading.Event()
_pending_action: str = ""             # "" = nothing, "grip" or "open"
_stop_flag = False
_thread: threading.Thread = None  # type: ignore[assignment]


# =============================================================================
# BACKGROUND WORKER THREAD
# =============================================================================

def _worker() -> None:
    """Execute gripper commands sequentially."""
    global _state, _last_action

    while not _stop_flag:
        fired = _event.wait(timeout=0.5)
        if not fired:
            continue
        _event.clear()

        with _lock:
            action = _pending_action
            if not action:
                continue

        # Mark as busy while executing
        with _lock:
            _state = "BUSY"

        try:
            if action == "grip":
                _execute_grip()
                with _lock:
                    _state = "GRIPPED"
                    _last_action = "grip"
            elif action == "open":
                _execute_open()
                with _lock:
                    _state = "OPEN"
                    _last_action = "open"
        except Exception as e:
            print(f"[gripper] error during {action}: {e}")
            with _lock:
                _state = "OPEN"  # failsafe to open


def _execute_grip() -> None:
    """Spin closed for GRIP_TIME seconds."""
    print(f"🤏 Gripper gripping ({GRIP_TIME}s)...")
    try:
        motor._write_word(SERVO_ID, _REG_SPEED, _SPEED_GRIP)
        time.sleep(GRIP_TIME)
        motor._write_word(SERVO_ID, _REG_SPEED, _SPEED_STOP)
        print("✅ Gripper gripped.")
    except Exception as e:
        print(f"❌ Grip failed: {e}")
        motor._write_word(SERVO_ID, _REG_SPEED, _SPEED_STOP)


def _execute_open() -> None:
    """Spin open for OPEN_TIME seconds."""
    print(f"✋ Gripper opening ({OPEN_TIME}s)...")
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
    """Setup servo in wheel mode and start background worker. Idempotent."""
    global _initialized, _stop_flag, _thread, _state

    if _initialized:
        return

    _stop_flag = False
    _state = "OPEN"

    # Disable torque before changing mode registers
    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 0)
    time.sleep(0.05)

    # Wheel mode: both angle limits = 0
    motor._write_word(SERVO_ID, _REG_CW_LIMIT,  0)
    time.sleep(0.02)
    motor._write_word(SERVO_ID, _REG_CCW_LIMIT, 0)
    time.sleep(0.02)

    # Re-enable torque
    motor._write_word(SERVO_ID, _REG_TORQUE_EN, 1)
    time.sleep(0.05)

    # Explicit stop
    motor._write_word(SERVO_ID, _REG_SPEED, 0)

    # Start background worker
    _thread = threading.Thread(target=_worker, daemon=True, name="gripper-worker")
    _thread.start()

    _initialized = True
    print(f"✅ Gripper initialised (ID {SERVO_ID}, wheel mode).")


def shutdown() -> None:
    """Stop motor and disable torque. Called automatically via motor.shutdown()."""
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
    Command the gripper. Non-blocking.

    Args:
        action: "grip" or "open"

    Returns:
        {"action": action, "status": "ok" | "ignored" | "busy"}
    """
    if not _initialized:
        return {"action": action, "status": "simulated"}

    action = action.lower()
    if action not in ("grip", "open"):
        return {"action": action, "status": "invalid"}

    with _lock:
        current_state = _state
        last = _last_action

    # Prevent repetition
    if last == action:
        return {"action": action, "status": "ignored", "reason": "same_as_last"}

    # Prevent overlapping commands
    if current_state == "BUSY":
        return {"action": action, "status": "busy", "reason": "command_in_progress"}

    # Accept the command
    with _lock:
        global _pending_action
        _pending_action = action
    _event.set()

    return {"action": action, "status": "ok"}


def grip() -> dict:
    """Request the gripper to close. Non-blocking."""
    return command("grip")


def open_gripper() -> dict:
    """Request the gripper to open. Non-blocking."""
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

        if action == "grip":
            result = grip()
            print(f"Command result: {result}")
            # Wait for it to complete
            for _ in range(30):
                if get_state() != "BUSY":
                    break
                time.sleep(0.1)
            print(f"Final state: {get_state()}")
        elif action == "open":
            result = open_gripper()
            print(f"Command result: {result}")
            # Wait for it to complete
            for _ in range(30):
                if get_state() != "BUSY":
                    break
                time.sleep(0.1)
            print(f"Final state: {get_state()}")
        else:
            print("Usage: python gripper.py grip   or   open")
    except KeyboardInterrupt:
        print("\n⏹ Interrupted.")
    finally:
        shutdown()
        motor.shutdown()