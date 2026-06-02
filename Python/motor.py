"""
motor.py
========
Hardware owner for all Dynamixel AX-12A servos.
Owns the single serial connection and GPIO handle shared across all motor modules.

All other motor files import from here — they do NOT open their own serial/GPIO.

Lifecycle:
    import motor
    motor.init()          # call once at startup
    # ... do stuff ...
    motor.shutdown()      # call on exit (also registered via atexit automatically)
"""

import platform
import atexit
import time

ON_PI = platform.system() == "Linux"

if ON_PI:
    import lgpio
import serial

# =========================
# CONFIG
# =========================
DIRECTION_PIN = 17
PORT          = "/dev/ttyAMA0"
BAUDRATE      = 1000000

# All known servo IDs — all get torque-off on shutdown
ALL_SERVO_IDS = [2, 3, 4, 5, 8, 13]

# Registers (AX-12A)
TORQUE_ENABLE = 24
GOAL_POSITION = 30
MOVING_SPEED  = 32

# Positional servo defaults (ID 13)
POS_LEFT  = 300
POS_RIGHT = 700
SPEED     = 300

# =========================
# HARDWARE STATE
# =========================
_h   = None   # lgpio chip handle
_ser = None   # serial port
_initialized = False

# =========================
# LIFECYCLE
# =========================

def init():
    """Open GPIO and serial. Call once at startup before using any motor."""
    global _h, _ser, _initialized

    if _initialized:
        print("⚠️  motor.init() called more than once — skipping.")
        return

    if ON_PI:
        _h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(_h, DIRECTION_PIN)

    _ser = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        timeout=0.1
    )

    _initialized = True
    print("✅ Motor hardware initialized.")

    # Register shutdown so it fires automatically on any exit
    atexit.register(shutdown)


def shutdown():
    """Torque-off all known servos, close serial + GPIO. Safe to call multiple times."""
    global _ser, _h, _initialized

    if not _initialized:
        return

    _initialized = False  # prevent double-shutdown from atexit + manual call

    print("🛑 Motor shutdown: disabling torque on all servos...")
    for servo_id in ALL_SERVO_IDS:
        try:
            _write_word(servo_id, TORQUE_ENABLE, 0)
        except Exception as e:
            print(f"  ⚠️  Could not disable torque on ID {servo_id}: {e}")

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


# =========================
# LOW-LEVEL COMMS
# (private — use write_word / write_byte from here or submodules)
# =========================

def _send_packet(packet: bytes):
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


def _write_word(servo_id: int, address: int, value: int):
    low  = value & 0xFF
    high = (value >> 8) & 0xFF
    data = [servo_id, 5, 0x03, address, low, high]
    chk  = _checksum(data)
    _send_packet(bytes([0xFF, 0xFF] + data + [chk]))


def _write_byte(servo_id: int, address: int, value: int):
    data = [servo_id, 4, 0x03, address, value]
    chk  = _checksum(data)
    _send_packet(bytes([0xFF, 0xFF] + data + [chk]))


# =========================
# PUBLIC HELPERS
# =========================

def enable_torque(servo_id: int):
    print(f"⚙️  Torque ON  (ID {servo_id})")
    _write_word(servo_id, TORQUE_ENABLE, 1)


def disable_torque(servo_id: int):
    print(f"🛑 Torque OFF (ID {servo_id})")
    _write_word(servo_id, TORQUE_ENABLE, 0)


# =========================
# POSITIONAL SERVO MOVES
# (default ID 13, but accepts any)
# =========================

def turn_left(servo_id: int = 13, position: int = POS_LEFT, speed: int = SPEED) -> dict:
    """
    Move a positional servo left.

    Args:
        servo_id: Target servo ID (default 13)
        position: Goal position    (default POS_LEFT = 300)
        speed:    Movement speed   (default 300)

    Returns:
        dict with direction, servo_id, position, speed, status
    """
    _write_word(servo_id, MOVING_SPEED, speed)
    _write_word(servo_id, GOAL_POSITION, position)
    time.sleep(1.5)
    return {"direction": "left", "servo_id": servo_id, "position": position, "speed": speed, "status": "ok"}


def turn_right(servo_id: int = 13, position: int = POS_RIGHT, speed: int = SPEED) -> dict:
    """
    Move a positional servo right.

    Args:
        servo_id: Target servo ID (default 13)
        position: Goal position    (default POS_RIGHT = 700)
        speed:    Movement speed   (default 300)

    Returns:
        dict with direction, servo_id, position, speed, status
    """
    _write_word(servo_id, MOVING_SPEED, speed)
    _write_word(servo_id, GOAL_POSITION, position)
    time.sleep(1.5)
    return {"direction": "right", "servo_id": servo_id, "position": position, "speed": speed, "status": "ok"}