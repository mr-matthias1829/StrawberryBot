"""
bin.py
======

post-grip bin-placement sequencer for StrawberryBot.

SIMPLE FLOW:
  1. gripper is already closed (called after successful grip)
  2. HOME ALL AXES except gripper (turntable, lift, arm, pivot)
  3. wait for motor lockout to expire
  4. drive to bin with slot offset (turntable + arm based on grid position)
  5. lower pivot, open gripper, raise pivot
  6. hOME ALL AXES again (including gripper this time)
  7. done - robot resumes searching for next berry

GRID LAYOUT (3x3, left-to-right, top-to-bottom):

    row 0 (achter):  [0]  [1]  [2]
    row 1 (midden):  [3]  [4]  [5]
    row 2 (voor):    [6]  [7]  [8]

    turntable → kolommen (links = minder tijd, rechts = meer tijd)
    arm → rijen (achter = minder tijd, voor = meer tijd)
"""

import time
import threading

# hardware imports

try:
    import turntable as _turntable
    _HAS_TURNTABLE = True
except ImportError:
    _HAS_TURNTABLE = False
    print("[bin] turntable not available — simulating")

try:
    import lift as _lift
    _HAS_LIFT = True
except ImportError:
    _HAS_LIFT = False
    print("[bin] lift not available — simulating")

try:
    import arm as _arm
    _HAS_ARM = True
except ImportError:
    _HAS_ARM = False
    print("[bin] arm not available — simulating")

try:
    import pivot as _pivot
    _HAS_PIVOT = True
except ImportError:
    _HAS_PIVOT = False
    print("[bin] pivot not available — simulating")

try:
    import gripper as _gripper
    _HAS_GRIPPER = True
except ImportError:
    _HAS_GRIPPER = False
    print("[bin] gripper not available — simulating")

try:
    import motor as _motor
    _HAS_MOTOR = True
except ImportError:
    _HAS_MOTOR = False
    print("[bin] motor not available — simulating")

# config

# direction to rotate toward bin ("left" or "right")
BIN_SIDE: str = "left"

# turntable: base time for center column (col 1)
TURNTABLE_BASE_S: float = 8.0
# turntable: extra time per column step (col 0 = -step, col 2 = +step)
TURNTABLE_STEP_S: float = 0.6

# arm: base time for center row (row 1)
ARM_BASE_S: float = 3.5
# arm: extra time per row step (row 0 = -step, row 2 = +step)
ARM_STEP_S: float = 0.5

# pivot: seconds to lower berry into slot
PIVOT_LOWER_S: float = 0.5

# pivot: seconds to raise back up
PIVOT_RAISE_S: float = 0.5

# wait after gripper opens before moving away
GRIPPER_DROP_WAIT_S: float = 0.5

# settle pause after each move
SETTLE_S: float = 0.2

# how long to wait for motor lockout to expire after homing (s)
LOCKOUT_WAIT_TIMEOUT_S: float = 15.0

# grid dimensions (3x3)
GRID_COLS: int = 3
GRID_ROWS: int = 3


_lock = threading.Lock()
_slot_index: int = 0
_busy: bool = False


def _slot_to_row_col(slot: int):
    """convert slot number (0-8) to (row, col)."""
    row = slot // GRID_COLS  # 0, 1, 2
    col = slot % GRID_COLS   # 0, 1, 2
    return row, col


def grid_full() -> bool:
    with _lock:
        return _slot_index >= GRID_ROWS * GRID_COLS


def collected_count() -> int:
    with _lock:
        return _slot_index


def reset() -> None:
    global _slot_index
    with _lock:
        _slot_index = 0
    print("[bin] reset — starting fresh bin.")


# lockout
def _wait_for_lockout() -> None:
    """block until the motor post-home lockout has expired."""
    if not _HAS_MOTOR:
        return
    deadline = time.monotonic() + LOCKOUT_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if not _motor.is_locked():
            return
        time.sleep(0.05)
    print("[bin] WARNING: lockout wait timed out — proceeding anyway")


# =============================================================================
# HOME HELPERS
# =============================================================================

def _home_axes_keep_gripper_closed() -> None:
    """
    home arm, lift, turntable and pivot — but NOT the gripper.
    the gripper stays closed because we are still holding the berry.
    also waits for the motor lockout to expire before returning,
    so subsequent timed moves are not silently dropped.
    """
    print("[bin] homing axes (gripper stays closed)...")

    if _HAS_ARM and hasattr(_arm, '_home'):
        print("[bin]  → arm")
        _arm._home()
        time.sleep(SETTLE_S)

    if _HAS_LIFT and hasattr(_lift, '_home'):
        print("[bin]  → lift")
        _lift._home()
        time.sleep(SETTLE_S)

    if _HAS_TURNTABLE and hasattr(_turntable, '_home'):
        print("[bin]  → turntable")
        _turntable._home()
        time.sleep(SETTLE_S)

    if _HAS_PIVOT and hasattr(_pivot, '_home'):
        print("[bin]  → pivot")
        _pivot._home()
        time.sleep(SETTLE_S)

    # Individual _home() calls don't set a lockout, but wait just in case
    # motor.home_all() was triggered elsewhere and is still ticking.
    _wait_for_lockout()

    print("[bin] homing complete (gripper kept closed)")


def _home_all_including_gripper() -> None:
    """
    home everything including the gripper (berry has been released).
    uses motor.home_all() if available, otherwise homes individually.
    waits for the lockout to expire before returning.
    """
    print("[bin] homing ALL axes (including gripper)...")

    if _HAS_MOTOR and hasattr(_motor, 'home_all'):
        _motor.home_all()
        # home_all() sets the lockout — wait for it to expire
        _wait_for_lockout()
    else:
        print("[bin] no motor.home_all() — homing individually...")

        if _HAS_GRIPPER and hasattr(_gripper, '_home'):
            print("[bin]  → gripper")
            _gripper._home()
            time.sleep(SETTLE_S)

        if _HAS_ARM and hasattr(_arm, '_home'):
            print("[bin]  → arm")
            _arm._home()
            time.sleep(SETTLE_S)

        if _HAS_LIFT and hasattr(_lift, '_home'):
            print("[bin]  → lift")
            _lift._home()
            time.sleep(SETTLE_S)

        if _HAS_TURNTABLE and hasattr(_turntable, '_home'):
            print("[bin]  → turntable")
            _turntable._home()
            time.sleep(SETTLE_S)

        if _HAS_PIVOT and hasattr(_pivot, '_home'):
            print("[bin]  → pivot")
            _pivot._home()
            time.sleep(SETTLE_S)

        _wait_for_lockout()

    print("[bin] Full homing complete")


# movement helpers

def _turntable_move(duration: float, direction: str = None) -> None:
    """rotate turntable for specific duration."""
    if direction is None:
        direction = BIN_SIDE

    if not _HAS_TURNTABLE:
        print(f"[bin][SIM] turntable {direction} {duration:.2f}s")
        time.sleep(duration)
        return

    print(f"[bin] turntable {direction} {duration:.2f}s")
    if direction == "left":
        _turntable.spin_left()
    else:
        _turntable.spin_right()

    time.sleep(duration)
    _turntable.stop()
    time.sleep(SETTLE_S)


def _arm_move(duration: float, forward: bool = True) -> None:
    """Move arm forward or backward."""
    if not _HAS_ARM:
        print(f"[bin][SIM] arm {'forward' if forward else 'backward'} {duration:.2f}s")
        time.sleep(duration)
        return

    print(f"[bin] arm {'forward' if forward else 'backward'} {duration:.2f}s")
    if forward and hasattr(_arm, 'move_forward'):
        _arm.move_forward()
    elif not forward and hasattr(_arm, 'move_backward'):
        _arm.move_backward()
    else:
        _arm.stop()
        time.sleep(duration)
        return

    time.sleep(duration)
    _arm.stop()
    time.sleep(SETTLE_S)


def _pivot_lower_drop() -> None:
    """lower pivot to drop berry into bin slot."""
    if not _HAS_PIVOT:
        print(f"[bin][SIM] pivot lower {PIVOT_LOWER_S}s")
        time.sleep(PIVOT_LOWER_S)
        return

    print(f"[bin] pivot lower {PIVOT_LOWER_S}s")
    _pivot.rotate_down()          # pivot.py uses rotate_down(), not move_down()
    time.sleep(PIVOT_LOWER_S)
    _pivot.stop()
    time.sleep(SETTLE_S)


def _pivot_raise_home() -> None:
    """raise pivot back to home position."""
    if not _HAS_PIVOT:
        print(f"[bin][SIM] pivot raise {PIVOT_RAISE_S}s")
        time.sleep(PIVOT_RAISE_S)
        return

    print(f"[bin] pivot raise {PIVOT_RAISE_S}s")
    _pivot.rotate_up()            # pivot.py uses rotate_up(), not move_up()
    time.sleep(PIVOT_RAISE_S)
    _pivot.stop()
    time.sleep(SETTLE_S)


def _open_gripper() -> None:
    """open gripper to release berry."""
    if not _HAS_GRIPPER:
        print("[bin][SIM] gripper open")
        return

    print("[bin] opening gripper...")
    result = _gripper.open_gripper()
    print(f"[bin] gripper open result: {result}")

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if _gripper.get_state() != "BUSY":
            break
        time.sleep(0.1)

    time.sleep(GRIPPER_DROP_WAIT_S)


# slot position calcuations

def _calculate_slot_times(row: int, col: int) -> tuple[float, float]:
    """
    Calculate turntable and arm times for a specific slot.

    Row 0 (achter): arm minder tijd
    Row 1 (midden): arm base tijd
    Row 2 (voor):   arm meer tijd

    Col 0 (links):  turntable minder tijd
    Col 1 (midden): turntable base tijd
    Col 2 (rechts): turntable meer tijd
    """
    col_offset = col - 1  # -1, 0, or 1
    turntable_time = TURNTABLE_BASE_S + (col_offset * TURNTABLE_STEP_S)

    row_offset = row - 1  # -1, 0, or 1
    arm_time = ARM_BASE_S + (row_offset * ARM_STEP_S)

    return turntable_time, arm_time


def _drive_to_slot(row: int, col: int) -> None:
    """drive turntable and arm to the specific slot position."""
    turntable_time, arm_time = _calculate_slot_times(row, col)

    print(f"[bin] driving to slot (row={row}, col={col})")
    print(f"[bin]   turntable: {turntable_time:.2f}s")
    print(f"[bin]   arm: {arm_time:.2f}s")

    _turntable_move(turntable_time)
    _arm_move(arm_time, forward=True)



def place_berry() -> bool:
    """
    Full post-grip deposit sequence. Blocking.

    Returns True on success, False if already busy.
    The gripper is guaranteed to be opened even if an error occurs mid-sequence.
    """
    global _busy, _slot_index

    with _lock:
        if _busy:
            print("[bin] place_berry() called while already busy — ignored.")
            return False
        current_slot = _slot_index
        _busy = True

    success = False
    try:
        total = GRID_ROWS * GRID_COLS
        row, col = _slot_to_row_col(current_slot)

        print(f"\n[bin] === PLACING BERRY #{current_slot + 1}/{total} ===")
        print(f"[bin] Slot {current_slot} → Row {row}, Col {col}")

        # STEP 1: Home all axes EXCEPT the gripper (berry is still held)
        print("[bin] STEP 1: Homing axes (gripper stays closed)...")
        _home_axes_keep_gripper_closed()

        # STEP 2: Drive to specific slot in bin
        print("[bin] STEP 2: Driving to slot...")
        _drive_to_slot(row, col)

        # STEP 3: Drop the berry
        print("[bin] STEP 3: Dropping berry...")
        _pivot_lower_drop()
        _open_gripper()
        _pivot_raise_home()

        # STEP 4: Home all axes (gripper can now open freely)
        print("[bin] STEP 4: Homing all axes...")
        _home_all_including_gripper()

        # STEP 5: Update counter
        with _lock:
            _slot_index += 1
            new_count = _slot_index

        if new_count >= total:
            print(f"[bin] bin full ({total}/{total}) — auto-resetting")
            reset()
        else:
            print(f"[bin] berry placed. {new_count}/{total} slots filled")

        print("[bin] === DONE ===")
        success = True
        return True

    except Exception as e:
        print(f"[bin] ERROR during placement: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # safety net: if anything went wrong before the gripper was opened,
        # force it open now so the robot doesn't stay stuck in GRIPPED state.
        if not success and _HAS_GRIPPER:
            try:
                state = _gripper.get_state()
                if state != "OPEN":
                    print("[bin] Safety: forcing gripper open after error...")
                    _gripper.open_gripper()
                    deadline = time.monotonic() + 10.0
                    while time.monotonic() < deadline:
                        if _gripper.get_state() != "BUSY":
                            break
                        time.sleep(0.1)
                    print(f"[bin] Safety: gripper state is now {_gripper.get_state()}")
            except Exception as e2:
                print(f"[bin] Safety gripper open failed: {e2}")

        with _lock:
            _busy = False


def status() -> dict:
    with _lock:
        idx = _slot_index
        total = GRID_ROWS * GRID_COLS
        busy = _busy

    slots = []
    for i in range(total):
        r, c = _slot_to_row_col(i)
        slots.append({
            "index": i,
            "row": r,
            "col": c,
            "filled": i < idx,
            "turntable_time": round(TURNTABLE_BASE_S + ((c - 1) * TURNTABLE_STEP_S), 2),
            "arm_time": round(ARM_BASE_S + ((r - 1) * ARM_STEP_S), 2)
        })

    return {
        "collected": idx,
        "capacity": total,
        "full": idx >= total,
        "busy": busy,
        "bin_side": BIN_SIDE,
        "grid": f"{GRID_ROWS}x{GRID_COLS}",
        "turntable_base": TURNTABLE_BASE_S,
        "turntable_step": TURNTABLE_STEP_S,
        "arm_base": ARM_BASE_S,
        "arm_step": ARM_STEP_S,
        "slots": slots,
    }


# standalone usage
if __name__ == "__main__":
    print("=== bin.py test (3x3 grid) ===")
    print(f"Grid: {GRID_ROWS}x{GRID_COLS}")
    print(f"Turntable: base={TURNTABLE_BASE_S}s, step={TURNTABLE_STEP_S}s")
    print(f"Arm: base={ARM_BASE_S}s, step={ARM_STEP_S}s\n")

    print("Slot layout:")
    for row in range(3):
        line = []
        for col in range(3):
            slot = row * 3 + col
            line.append(f"{slot}")
        print(f"  Row {row}: {' '.join(line)}")
    print()

    for i in range(GRID_ROWS * GRID_COLS + 1):
        print(f"\n--- Test #{i + 1} ---")
        place_berry()
        print(f"Status: {status()}")
        time.sleep(1)