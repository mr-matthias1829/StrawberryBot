"""
_homing_utils.py

shared helpers for the per-motor _home() routines.

each motor module:
  1. records its zero point at init() time:
      _zero_deg  (float | None)  — sensor reading at startup, if available
      _dead_pos  (float)         — dead-reckoning accumulator (speed-units * s)

  2. calls accumulate(speed_word, dt) after every successful _write_word so
    the dead-reckoning counter stays current.

  3. calls _home() which uses sensor when available, otherwise reverses the
    dead-reckoning accumulator.

dead-reckoning:
    the accumulator stores funny formula stuff aka time
    (same convention the motor modules use internally).
    it is NOT in degrees, it is only used to reverse the same amount of
    motion, so absolute calibration is not required
    ... just see this as a fallback

sensor homing:
    drive toward zero_deg until (current - zero_deg) ≤ SENSOR_DEADBAND
    or timeout expires.
    also means that if servo rotates too slowly, it will fail to home.
"""

from __future__ import annotations

import time
from typing import Callable, Optional

# how close we need to be before we consider the motor "at zero"
# effectively "wiggle" room
SENSOR_DEADBAND = 5.0          # degrees

# how long to wait for sensor-guided homing before giving up
SENSOR_HOME_TIMEOUT = 10.0     # seconds

# speed used when homing without a sensor
# highly recommend not putting this too high
DR_HOME_SPEED = 400           # servo speed units


# dead-reckoning stuff

def signed_speed(speed_word: int) -> float:
    """
    convert a raw AX-12A speed register word back to a signed speed value.

    AX-12A wheel-mode encoding:
        bit 10 = direction  (0 = CCW/positive, 1 = CW/negative)
        bits 9-0 = magnitude 0-1023

    TL:DR, convert binary value to something we can actually use
    """
    if speed_word == 0:
        return 0.0
    magnitude = speed_word & 0x3FF
    direction = (speed_word >> 10) & 1
    return float(-magnitude if direction else magnitude)


def accumulate(current: list, speed_word: int, dt: float) -> None:
    """
    update a mutable dead-reckoning accumulator in-place.

    current must be a single-element list [float] so the caller's
    module-level variable is mutated.  Pass it as [_dead_pos].
    """
    current[0] += signed_speed(speed_word) * dt


# Sensor-guided homing

# buncha parameters we kinda need and want to properly home the servo
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
    drive the motor toward zero_deg using sensor feedback.

    returns True on success, False on timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        reading = read_sensor_fn()
        if reading is None:
            stop_fn()
            print("    [home_with_sensor] sensor lost mid-home — aborting")
            return False

        pos = total_position_fn(reading)

        if ignore_laps: # servo doesn't need to spin all laps back, ignore laps
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


# dead-reckoning homing (no sensor)
def home_dead_reckoning(
    dead_pos_ref:       list,          # mutable [float] accumulator
    drive_positive_fn:  Callable[[int], None],
    drive_negative_fn:  Callable[[int], None],
    stop_fn:            Callable[[], None],
    speed:              int = DR_HOME_SPEED,
    timeout:            float = SENSOR_HOME_TIMEOUT,
) -> None:
    """
    reverse the accumulated dead-reckoning position back to zero.

    because we only know relative movement, we drive in the opposite
    direction of the accumulated displacement until it reaches zero,
    then stop.  the actual position will be approximate.

    this again, acts as a fallback if no sensors are used. it's better than nothing.
    as a result, this method stays mostly unused as most have a sensor.
    """
    target = dead_pos_ref[0]

    if abs(target) < 1.0:
        stop_fn()
        dead_pos_ref[0] = 0.0
        return

    # we need to subtract target worth of motion in the correct direction.
    # we do this by timing: run at DR_HOME_SPEED until elapsed * speed >= target
    needed   = abs(target)
    deadline = time.monotonic() + timeout
    t_start  = time.monotonic()

    if target > 0:
        # accumulated forward — now go backward
        drive_negative_fn(speed)
    else:
        # accumulated backward — now go forward
        drive_positive_fn(speed)

    while time.monotonic() < deadline:
        elapsed  = time.monotonic() - t_start
        traveled = elapsed * speed
        if traveled >= needed:
            break
        time.sleep(0.02)

    stop_fn()
    dead_pos_ref[0] = 0.0