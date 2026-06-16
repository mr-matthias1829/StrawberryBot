"""
bin.py
======

Post-grip bin-placement sequencer for StrawberryBot.

SIMPLE FLOW:
  1. Gripper is already closed (called after successful grip)
  2. HOME ALL AXES except gripper (turntable, lift, arm, pivot)
  3. Wait for motor lockout to expire
  4. Drive to bin with slot offset (turntable + arm based on grid position)
  5. Lower pivot, open gripper, raise pivot
  6. HOME ALL AXES again (including gripper this time)
  7. Done - robot resumes searching for next berry

GRID LAYOUT (3x3, left-to-right, top-to-bottom):

    Row 0 (achter):  [0]  [1]  [2]
    Row 1 (midden):  [3]  [4]  [5]
    Row 2 (voor):    [6]  [7]  [8]

    Turntable → kolommen (links = minder tijd, rechts = meer tijd)
    Arm → rijen (achter = minder tijd, voor = meer tijd)

FIX: modules zijn nu geïnitialiseerd via _ensure_init() voordat ze gebruikt
worden. Zonder init() draait de writer thread niet en worden alle commando's
stilletjes genegeerd.
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

# tuning

BIN_SIDE: str = "left"

TURNTABLE_BASE_S: float = 8.0
TURNTABLE_STEP_S: float = 0.6

ARM_BASE_S: float = 3.5
ARM_STEP_S: float = 0.5

PIVOT_LOWER_S: float = 0.5
PIVOT_RAISE_S: float = 0.5

GRIPPER_DROP_WAIT_S: float = 0.5
SETTLE_S: float = 0.2

LOCKOUT_WAIT_TIMEOUT_S: float = 15.0
POST_HOME_SETTLE_S: float = 0.5
LOCKOUT_CLEAR_TIMEOUT_S: float = 2.0

GRID_COLS: int = 3
GRID_ROWS: int = 3

# state

_lock = threading.Lock()
_slot_index: int = 0
_busy: bool = False
_modules_initialized: bool = False


def _ensure_init() -> None:
    """
    Make sure all hardware modules are initialised before first use.

    THE ROOT CAUSE OF "turntable doesn't move":
    Each servo module (turntable, arm, lift, …) starts its background
    writer thread only inside its own init() call.  Without init(), the
    writer thread never starts and _post_word() silently queues commands
    that nobody ever reads.

    bin.py used to rely on the caller having already called init() on every
    module — but that assumption was fragile.  This function makes bin.py
    self-sufficient: it calls init() on every module it needs, and init()
    is idempotent so double-calling is safe.
    """
    global _modules_initialized
    if _modules_initialized:
        return

    print("[bin] _ensure_init(): initialising hardware modules…")

    if _HAS_MOTOR and hasattr(_motor, 'init'):
        _motor.init()

    if _HAS_TURNTABLE and hasattr(_turntable, 'init'):
        _turntable.init()

    if _HAS_LIFT and hasattr(_lift, 'init'):
        _lift.init()

    if _HAS_ARM and hasattr(_arm, 'init'):
        _arm.init()

    if _HAS_PIVOT and hasattr(_pivot, 'init'):
        _pivot.init()

    if _HAS_GRIPPER and hasattr(_gripper, 'init'):
        _gripper.init()

    _modules_initialized = True
    print("[bin] _ensure_init(): all modules ready")


def _slot_to_row_col(slot: int):
    row = slot // GRID_COLS
    col = slot % GRID_COLS
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
    print("[bin] Reset — starting fresh bin.")


# lockout helpers

def _wait_for_lockout() -> None:
    """Block until the motor post-home lockout has expired."""
    if not _HAS_MOTOR:
        return
    deadline = time.monotonic() + LOCKOUT_WAIT_TIMEOUT_S
    while time.monotonic() < deadline:
        if not _motor.is_locked():
            return
        time.sleep(0.05)
    print("[bin] WARNING: lockout wait timed out — proceeding anyway")


def _wait_for_lockout_robust() -> None:
    """
    Robust lockout waiting that handles lockouts that start after we start waiting.
    """
    if not _HAS_MOTOR:
        return

    _wait_for_lockout()

    start_time = time.monotonic()
    settle_deadline = start_time + POST_HOME_SETTLE_S

    print(f"[bin] Lockout watch period: {POST_HOME_SETTLE_S:.1f}s")

    while time.monotonic() < settle_deadline:
        if hasattr(_motor, 'lock_was_set_recently'):
            if _motor.lock_was_set_recently(threshold=0.3):
                print("[bin] New lockout detected during settle period — waiting for it to clear...")
                clear_deadline = time.monotonic() + LOCKOUT_CLEAR_TIMEOUT_S
                while time.monotonic() < clear_deadline:
                    if not _motor.is_locked():
                        print("[bin] Lockout cleared")
                        settle_deadline = time.monotonic() + POST_HOME_SETTLE_S
                        break
                    time.sleep(0.05)
                else:
                    print("[bin] WARNING: lockout clear timeout — proceeding anyway")
                    return
        else:
            if _motor.is_locked():
                print("[bin] Lockout became active during settle — waiting...")
                _wait_for_lockout()
                settle_deadline = time.monotonic() + POST_HOME_SETTLE_S

        time.sleep(0.02)

    print("[bin] Lockout watch period complete — safe to move")


# home helpers

def _home_axes_keep_gripper_closed() -> None:
    print("[bin] Homing axes (gripper stays closed)...")

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

    _wait_for_lockout_robust()

    print("[bin] Homing complete (gripper kept closed)")


def _home_all_including_gripper() -> None:
    print("[bin] Homing ALL axes (including gripper)...")

    if _HAS_MOTOR and hasattr(_motor, 'home_all'):
        _motor.home_all()
        _wait_for_lockout_robust()
    else:
        print("[bin] No motor.home_all() — homing individually...")

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

        _wait_for_lockout_robust()

    print("[bin] Full homing complete")


# movement helpers

def _turntable_move(duration: float, direction: str = None) -> None:
    if direction is None:
        direction = BIN_SIDE

    if not _HAS_TURNTABLE:
        print(f"[bin][SIM] turntable {direction} {duration:.2f}s")
        time.sleep(duration)
        return

    print(f"[bin] Turntable {direction} {duration:.2f}s")
    if direction == "left":
        _turntable.spin_left()
    else:
        _turntable.spin_right()

    time.sleep(duration)
    _turntable.stop()
    time.sleep(SETTLE_S)


def _arm_move(duration: float, forward: bool = True) -> None:
    if not _HAS_ARM:
        print(f"[bin][SIM] arm {'forward' if forward else 'backward'} {duration:.2f}s")
        time.sleep(duration)
        return

    print(f"[bin] Arm {'forward' if forward else 'backward'} {duration:.2f}s")
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
    if not _HAS_PIVOT:
        print(f"[bin][SIM] pivot lower {PIVOT_LOWER_S}s")
        time.sleep(PIVOT_LOWER_S)
        return

    print(f"[bin] Pivot lower {PIVOT_LOWER_S}s")
    _pivot.rotate_down()
    time.sleep(PIVOT_LOWER_S)
    _pivot.stop()
    time.sleep(SETTLE_S)


def _pivot_raise_home() -> None:
    if not _HAS_PIVOT:
        print(f"[bin][SIM] pivot raise {PIVOT_RAISE_S}s")
        time.sleep(PIVOT_RAISE_S)
        return

    _wait_for_lockout_robust()

    print(f"[bin] Pivot raise {PIVOT_RAISE_S}s")
    _pivot.rotate_up()
    time.sleep(PIVOT_RAISE_S)
    _pivot.stop()
    time.sleep(SETTLE_S)


def _open_gripper() -> None:
    if not _HAS_GRIPPER:
        print("[bin][SIM] gripper open")
        return

    print("[bin] Opening gripper...")
    result = _gripper.open_gripper()
    print(f"[bin] Gripper open result: {result}")

    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if _gripper.get_state() != "BUSY":
            break
        time.sleep(0.1)

    time.sleep(GRIPPER_DROP_WAIT_S)


# slot position calculation

def _calculate_slot_times(row: int, col: int) -> tuple[float, float]:
    col_offset = col - 1
    turntable_time = TURNTABLE_BASE_S + (col_offset * TURNTABLE_STEP_S)
    row_offset = row - 1
    arm_time = ARM_BASE_S + (row_offset * ARM_STEP_S)
    return turntable_time, arm_time


def _drive_to_slot(row: int, col: int) -> None:
    turntable_time, arm_time = _calculate_slot_times(row, col)

    print(f"[bin] Driving to slot (row={row}, col={col})")
    print(f"[bin]   Turntable: {turntable_time:.2f}s")
    print(f"[bin]   Arm: {arm_time:.2f}s")

    _turntable_move(turntable_time)
    _arm_move(arm_time, forward=True)


# main sequence

def place_berry() -> bool:
    """
    Full post-grip deposit sequence. Blocking.

    Returns True on success, False if already busy.
    The gripper is guaranteed to be opened even if an error occurs mid-sequence.
    """
    global _busy, _slot_index

    # Ensure all modules are initialised — this is what was missing.
    _ensure_init()

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

        print("[bin] STEP 1: Homing axes (gripper stays closed)...")
        _home_axes_keep_gripper_closed()

        print("[bin] STEP 2: Driving to slot...")
        _drive_to_slot(row, col)

        print("[bin] STEP 3: Dropping berry...")
        _pivot_lower_drop()
        _open_gripper()
        _pivot_raise_home()

        print("[bin] STEP 4: Homing all axes...")
        _home_all_including_gripper()

        with _lock:
            _slot_index += 1
            new_count = _slot_index

        if new_count >= total:
            print(f"[bin] Bin full ({total}/{total}) — auto-resetting")
            reset()
        else:
            print(f"[bin] Berry placed. {new_count}/{total} slots filled")

        print("[bin] === DONE ===")
        success = True
        return True

    except Exception as e:
        print(f"[bin] ERROR during placement: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
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


# status

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


# test

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