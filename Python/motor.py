"""
motor.py
========
Hardware owner for all Dynamixel AX-12A servos.

Owns the single serial connection and GPIO handle shared by every motor
sub-module (turntable, lift, gripper).  Nothing else opens serial/GPIO.

Lifecycle
---------
    import motor
    motor.init()       # once at startup — idempotent
    # ... run ...
    motor.shutdown()   # on exit (also registered via atexit automatically)

Low-level helpers (_write_word, _write_byte, _send_packet) are intentionally
private.  Sub-modules import and call them directly; nothing outside this
package should need them.

Homing
------
    motor.home_all()   # homes all motors sequentially, then locks out
                       # all motor commands for 5 seconds.

    motor.is_locked()  # True while the post-home lockout is active.
                       # Sub-modules call this in _post_word / _send_packet
                       # to silently drop commands during the lockout window.
"""

import atexit
import platform
import time
import threading

ON_PI = platform.system() == "Linux"

if ON_PI:
    import lgpio

import serial

# =============================================================================
# CONFIG
# =============================================================================

DIRECTION_PIN  = 17
PORT           = "/dev/ttyAMA0"
BAUDRATE       = 1_000_000

# All known servo IDs — torque is disabled on all of them during shutdown.
ALL_SERVO_IDS = [2, 3, 4, 5, 8, 13]

# AX-12A register addresses
TORQUE_ENABLE = 24
GOAL_POSITION = 30
MOVING_SPEED  = 32

# =============================================================================
# HARDWARE STATE  (module-level singletons)
# =============================================================================

_h:            object = None   # lgpio chip handle  (Pi only)
_ser:          object = None   # serial.Serial instance
_initialized:  bool   = False

# =============================================================================
# HOMING LOCKOUT
# =============================================================================

_locked:        bool            = False   # True during post-home lockout
_locked_until:  float           = 0.0     # monotonic timestamp when lock expires
_locked_set_at: float           = 0.0     # monotonic timestamp when lock was set
_lock_mutex:    threading.Lock  = threading.Lock()

POST_HOME_LOCKOUT_S = 0.3   # seconds to refuse commands after home_all()


def is_locked() -> bool:
    """Return True if the post-home lockout is currently active."""
    global _locked
    with _lock_mutex:
        if _locked and time.monotonic() >= _locked_until:
            _locked = False
        return _locked


def lock_was_set_recently(threshold: float = 0.5) -> bool:
    """
    Return True if the lockout was set within the last 'threshold' seconds.
    This helps detect lockouts that start after we think we've waited for them.
    """
    with _lock_mutex:
        if _locked_set_at == 0:
            return False
        return time.monotonic() - _locked_set_at < threshold


def _set_lock(duration: float) -> None:
    global _locked, _locked_until, _locked_set_at
    with _lock_mutex:
        _locked = True
        _locked_until = time.monotonic() + duration
        _locked_set_at = time.monotonic()


def _clear_lock() -> None:
    global _locked, _locked_set_at
    with _lock_mutex:
        _locked = False
        _locked_set_at = 0.0

# =============================================================================
# LIFECYCLE
# =============================================================================

def init() -> None:
    """Open GPIO and serial port.  Call once at startup before any motor use."""
    global _h, _ser, _initialized

    if _initialized:
        print("⚠️  motor.init() called more than once — skipping.")
        return

    if ON_PI:
        _h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(_h, DIRECTION_PIN)

    _ser = serial.Serial(port=PORT, baudrate=BAUDRATE, timeout=0.1)

    _initialized = True
    print(" Motor hardware initialised.")

    atexit.register(shutdown)


def shutdown() -> None:
    """Torque-off all servos, close serial and GPIO.  Safe to call multiple times."""
    global _ser, _h, _initialized

    if not _initialized:
        return

    # Guard against double-call from atexit + manual shutdown
    _initialized = False

    print(" Motor shutdown: disabling torque on all servos…")
    for sid in ALL_SERVO_IDS:
        try:
            _write_word(sid, TORQUE_ENABLE, 0)
        except Exception as e:
            print(f"  ⚠️  Could not disable torque on ID {sid}: {e}")

    time.sleep(0.1)

    if _ser is not None:
        try:
            _ser.close()
        except Exception:
            pass
        _ser = None

    if ON_PI and _h is not None:
        try:
            lgpio.gpiochip_close(_h)
        except Exception:
            pass
        _h = None

    print(" Motor shutdown complete.")

# =============================================================================
# LOW-LEVEL COMMS  (private — used by sub-modules only)
# =============================================================================

def _send_packet(packet: bytes) -> None:
    if not _initialized:
        raise RuntimeError("motor.init() has not been called.")
    if ON_PI:
        lgpio.gpio_write(_h, DIRECTION_PIN, 1)
    _ser.write(packet)
    _ser.flush()
    time.sleep(0.001)
    if ON_PI:
        lgpio.gpio_write(_h, DIRECTION_PIN, 0)


def _checksum(data: list) -> int:
    return (~sum(data)) & 0xFF


def _write_word(servo_id: int, address: int, value: int) -> None:
    low  = value & 0xFF
    high = (value >> 8) & 0xFF
    data = [servo_id, 5, 0x03, address, low, high]
    _send_packet(bytes([0xFF, 0xFF] + data + [_checksum(data)]))


def _write_byte(servo_id: int, address: int, value: int) -> None:
    data = [servo_id, 4, 0x03, address, value]
    _send_packet(bytes([0xFF, 0xFF] + data + [_checksum(data)]))

# =============================================================================
# PUBLIC TORQUE HELPERS
# =============================================================================

def enable_torque(servo_id: int) -> None:
    print(f"  Torque ON  (ID {servo_id})")
    _write_word(servo_id, TORQUE_ENABLE, 1)


def disable_torque(servo_id: int) -> None:
    print(f" Torque OFF (ID {servo_id})")
    _write_word(servo_id, TORQUE_ENABLE, 0)

# =============================================================================
# HOMING  (sequential, blocks until all motors are zeroed)
# =============================================================================

def home_all() -> None:
    """
    Home every motor sequentially then lock out all motor commands for
    POST_HOME_LOCKOUT_S seconds.

    Order (safe for the physical layout):
      1. Gripper  — open (simplest: just use existing state machine)
      2. Pivot    — tilt to neutral (NOW NO LONGER THE CASE)
      3. Arm      — retract fully
      4. Lift     — move to bottom
      5. Turntable — rotate to centre

    This function blocks the calling thread until homing is complete, then
    starts the lockout timer before returning.  During the lockout window
    is_locked() returns True and sub-modules silently ignore commands.
    """
    import gripper as _gripper
    import arm     as _arm
    import lift    as _lift
    import turntable as _turntable

    print(" home_all(): starting sequential homing…")

    # 1 ── Gripper ────────────────────────────────────────────────────────────
    print("  [1/4] Gripper → OPEN")
    _gripper._home()

    # 3 ── Arm ────────────────────────────────────────────────────────────────
    print("  [2/4] Arm → zero (retract)")
    _arm._home()

    # 4 ── Lift ───────────────────────────────────────────────────────────────
    print("  [3/4] Lift → zero (bottom)")
    _lift._home()

    # 5 ── Turntable ──────────────────────────────────────────────────────────
    print("  [4/4] Turntable → zero (centre)")
    _turntable._home()

    print(f" home_all(): complete — locking commands for {POST_HOME_LOCKOUT_S:.0f}s")
    _set_lock(POST_HOME_LOCKOUT_S)
    # Spin in a background thread so we don't block the caller forever, but
    # print a clear message when the lock expires.
    def _log_unlock():
        time.sleep(POST_HOME_LOCKOUT_S + 0.05)
        print(" Motor lockout expired — commands accepted again.")
    threading.Thread(target=_log_unlock, daemon=True, name="home-lock-expire").start()