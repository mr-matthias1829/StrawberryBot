"""
_homing_utils.py
================
Shared helpers for the per-motor _home() routines.

Each motor module:
  • Records its zero point at init() time:
      _zero_deg  (float | None)  — sensor reading at startup, if available
      _dead_pos  (float)         — dead-reckoning accumulator (speed-units × s)

  • Calls accumulate(speed_word, dt) after every successful _write_word so
    the dead-reckoning counter stays current.

  • Calls _home() which uses sensor when available, otherwise reverses the
    dead-reckoning accumulator.

Dead-reckoning unit:
    The accumulator stores  Σ (signed_speed × dt_seconds).
    "signed speed" is:  +speed for CCW/forward, -speed for CW/backward
    (same convention the motor modules use internally).
    It is NOT in degrees — it is only used to reverse the same amount of
    motion, so absolute calibration is not required.

Sensor homing:
    Drive toward zero_deg until |current - zero_deg| ≤ SENSOR_DEADBAND
    or timeout expires.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

# How close (degrees) we need to be before we consider the motor "at zero"
SENSOR_DEADBAND = 5.0          # degrees

# How long to wait for sensor-guided homing before giving up
SENSOR_HOME_TIMEOUT = 10.0     # seconds

# Speed used when homing without a sensor (slow, safe)
DR_HOME_SPEED = 150            # servo speed units


# ---------------------------------------------------------------------------
# Dead-reckoning helpers
# ---------------------------------------------------------------------------

def signed_speed(speed_word: int) -> float:
    """
    Convert a raw AX-12A speed register word back to a signed speed value.

    AX-12A wheel-mode encoding:
        bit 10 = direction  (0 = CCW/positive, 1 = CW/negative)
        bits 9-0 = magnitude 0-1023
    """
    if speed_word == 0:
        return 0.0
    magnitude = speed_word & 0x3FF
    direction = (speed_word >> 10) & 1
    return float(-magnitude if direction else magnitude)


def accumulate(current: list, speed_word: int, dt: float) -> None:
    """
    Update a mutable dead-reckoning accumulator in-place.

    current must be a single-element list [float] so the caller's
    module-level variable is mutated.  Pass it as [_dead_pos].
    """
    current[0] += signed_speed(speed_word) * dt


# ---------------------------------------------------------------------------
# Sensor-guided homing
# ---------------------------------------------------------------------------

def home_with_sensor(
    read_sensor_fn:     Callable[[], Optional[dict]],
    total_position_fn:  Callable[[dict], float],
    zero_deg:           float,
    drive_positive_fn:  Callable[[int], None],
    drive_negative_fn:  Callable[[int], None],
    stop_fn:            Callable[[], None],
    speed:              int = DR_HOME_SPEED,
    timeout:            float = SENSOR_HOME_TIMEOUT,
    ignore_laps:        bool = False,
) -> bool:
    """
    Drive the motor toward zero_deg using sensor feedback.

    Returns True on success, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reading = read_sensor_fn()
        if reading is None:
            stop_fn()
            print("    [home_with_sensor] sensor lost mid-home — aborting")
            return False

        pos = total_position_fn(reading)

        if ignore_laps:
            error = ((zero_deg - pos + 180.0) % 360.0) - 180.0
        else:
            error = zero_deg - pos

        if abs(error) <= SENSOR_DEADBAND:
            stop_fn()
            return True

        if error > 0:
            drive_positive_fn(speed)
        else:
            drive_negative_fn(speed)

        time.sleep(0.05)

    stop_fn()
    print(f"    [home_with_sensor] timeout after {timeout:.1f}s — stopped where we are")
    return False


# ---------------------------------------------------------------------------
# Dead-reckoning homing (no sensor)
# ---------------------------------------------------------------------------

def home_dead_reckoning(
    dead_pos_ref:       list,          # mutable [float] accumulator
    drive_positive_fn:  Callable[[int], None],
    drive_negative_fn:  Callable[[int], None],
    stop_fn:            Callable[[], None],
    speed:              int = DR_HOME_SPEED,
    timeout:            float = SENSOR_HOME_TIMEOUT,
) -> None:
    """
    Reverse the accumulated dead-reckoning position back to zero.

    Because we only know relative movement, we drive in the opposite
    direction of the accumulated displacement until it reaches zero,
    then stop.  The actual position will be approximate.
    """
    target = dead_pos_ref[0]

    if abs(target) < 1.0:
        stop_fn()
        dead_pos_ref[0] = 0.0
        return

    # We need to subtract |target| worth of motion in the correct direction.
    # We do this by timing: run at DR_HOME_SPEED until elapsed × speed ≥ |target|
    # (same units as the accumulator).
    needed   = abs(target)
    deadline = time.monotonic() + timeout
    t_start  = time.monotonic()

    if target > 0:
        # Accumulated forward — now go backward
        drive_negative_fn(speed)
    else:
        # Accumulated backward — now go forward
        drive_positive_fn(speed)

    while time.monotonic() < deadline:
        elapsed  = time.monotonic() - t_start
        traveled = elapsed * speed
        if traveled >= needed:
            break
        time.sleep(0.02)

    stop_fn()
    dead_pos_ref[0] = 0.0