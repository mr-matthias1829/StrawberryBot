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
"""

import atexit
import platform
import time

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
# LIFECYCLE
# =============================================================================

def init() -> None: # TODO: brothar we need to make it reset when it shuts down, the zero point my guy, dewit.
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
    print("✅ Motor hardware initialised.")

    atexit.register(shutdown)


def shutdown() -> None:
    """Torque-off all servos, close serial and GPIO.  Safe to call multiple times."""
    global _ser, _h, _initialized

    if not _initialized:
        return

    # Guard against double-call from atexit + manual shutdown
    _initialized = False

    print("🛑 Motor shutdown: disabling torque on all servos…")
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

    print("✅ Motor shutdown complete.")

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
    print(f"⚙️  Torque ON  (ID {servo_id})")
    _write_word(servo_id, TORQUE_ENABLE, 1)


def disable_torque(servo_id: int) -> None:
    print(f"🛑 Torque OFF (ID {servo_id})")
    _write_word(servo_id, TORQUE_ENABLE, 0)