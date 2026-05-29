"""
gripper_grip.py
===============
Wheel-mode gripper — ID 8, spins for X seconds then stops.

Kalibreren:
  1. Zet gripper handmatig in open stand
  2. python gripper_grip.py grip  → pas GRIP_TIME aan
  3. python gripper_grip.py open  → pas OPEN_TIME aan

Gebruik (standalone):
  python gripper_grip.py grip
  python gripper_grip.py open

Gebruik (als module):
  import motor; motor.init()
  from gripper_grip import Gripper
  g = Gripper()
  g.open()
  g.grip()
"""

import time
import motor  # hardware owner — serial + GPIO live here

# =========================
# CONFIG
# =========================
GRIPPER_ID = 8

GRIP_TIME = 2.5   # seconds to spin closed — tune after test
OPEN_TIME = 2.5   # seconds to spin open   — tune after test
SPEED     = 1023  # 0–1023

# AX-12A registers
TORQUE_ENABLE    = 0x18
CW_ANGLE_LIMIT   = 0x06
CCW_ANGLE_LIMIT  = 0x08
MOVING_SPEED_REG = 0x20

# Wheel mode speed values
SPEED_LEFT  = SPEED           # CCW
SPEED_RIGHT = 1024 + SPEED    # CW
SPEED_STOP  = 0


# =========================
# GRIPPER CLASS
# =========================

class Gripper:
    """Wheel-mode gripper wrapper. Assumes motor.init() has already been called."""

    def __init__(self, servo_id: int = GRIPPER_ID):
        self.id        = servo_id
        self.grip_time = GRIP_TIME
        self.open_time = OPEN_TIME
        self._setup()

    def _setup(self):
        """Put servo in wheel mode (both angle limits = 0) and enable torque."""
        motor._write_word(self.id, CW_ANGLE_LIMIT,  0)
        motor._write_word(self.id, CCW_ANGLE_LIMIT, 0)
        time.sleep(0.05)
        motor._write_byte(self.id, TORQUE_ENABLE, 1)
        time.sleep(0.05)

    def _stop(self):
        motor._write_word(self.id, MOVING_SPEED_REG, SPEED_STOP)

    def grip(self, speed: int = SPEED) -> dict:
        """Spin closed for grip_time seconds."""
        print(f"🤏 Gripper closing ({self.grip_time}s)...")
        motor._write_word(self.id, MOVING_SPEED_REG, SPEED_LEFT)
        time.sleep(self.grip_time)
        self._stop()
        print("✅ Gripper closed.")
        return {"action": "grip", "servo_id": self.id, "duration": self.grip_time, "status": "ok"}

    def open(self, speed: int = SPEED) -> dict:
        """Spin open for open_time seconds."""
        print(f"✋ Gripper opening ({self.open_time}s)...")
        motor._write_word(self.id, MOVING_SPEED_REG, SPEED_RIGHT)
        time.sleep(self.open_time)
        self._stop()
        print("✅ Gripper open.")
        return {"action": "open", "servo_id": self.id, "duration": self.open_time, "status": "ok"}


# =========================
# STANDALONE ENTRYPOINT
# =========================

if __name__ == "__main__":
    import sys
    actie = sys.argv[1].lower() if len(sys.argv) > 1 else "grip"

    motor.init()

    try:
        g = Gripper()
        if actie == "grip":
            g.grip()
        elif actie == "open":
            g.open()
        else:
            print("Gebruik: python gripper_grip.py grip   of   open")
    except KeyboardInterrupt:
        print("\nAfgebroken.")
    finally:
        motor.shutdown()