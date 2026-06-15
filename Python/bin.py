"""
bin.py
======

Post-grip bin-placement sequencer for StrawberryBot.

Flow (called after gripper.grip() confirms a successful grab):

  1. Return turntable to its zero-point via turntable._home().
     The pick-position origin is implicitly preserved in the turntable's
     own dead-reckoning / sensor state — _home() knows how to get back.
  2. Drive the turntable left or right for SIDE_TRAVEL_S seconds so the
     physical robot ends up rotated 90 degrees above the bin.
  3. Apply small row/col offsets to land above the correct slot.
  4. Lower the pivot slowly so the berry is just above the slot.
  5. Open the gripper — berry drops in.
  6. Raise the pivot back to pick height.
  7. Undo any column nudge, then home the turntable again so it is back
     at zero and ready to resume tracking.
  8. Advance the slot counter (left-to-right, top-to-bottom, 3 × 3 grid).
     If the bin is now full, reset automatically for the next bin.

Grid layout (as seen from above):
    0  1  2
    3  4  5
    6  7  8

  The arm always centres on slot 4 first.
  Row offsets → small extra pivot moves.
  Col offsets → small extra turntable nudges.

Configuration
-------------
All tuning constants live at the top of this file.

Thread safety
-------------
place_berry() is blocking and must be called from a worker thread,
NOT from the main capture loop.

Usage
-----
    import bin as bin_placer

    # After a successful grip:
    bin_placer.place_berry()

    # Query state:
    bin_placer.collected_count()
    bin_placer.status()

    # Manual reset (bin swapped out):
    bin_placer.reset()
"""

import time
import threading
from typing import Optional

# =============================================================================
# HARDWARE IMPORTS  (graceful degradation on non-Pi)
# =============================================================================

try:
    import turntable as _turntable
    _HAS_TURNTABLE = True
except ImportError:
    _HAS_TURNTABLE = False
    print("[bin] turntable not available — simulating")

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

# =============================================================================
# TUNING — change these, leave the logic below untouched
# =============================================================================

# Direction the turntable sweeps to reach the bin (from its zero-point).
# "left"  → CCW  |  "right" → CW
BIN_SIDE: str = "left"

# How long the turntable runs for the 90° sweep toward the bin.
# Tune until the physical robot stands exactly 90° rotated over the bin.
SIDE_TRAVEL_S: float = 1.8          # seconds

# How long the turntable runs for the return sweep (usually identical).
RETURN_TRAVEL_S: float = 1.8        # seconds

# Settle pause after every turntable stop (mechanical damping).
TURNTABLE_SETTLE_S: float = 0.3     # seconds

# How long the pivot descends to bring the berry level with the slot opening.
PIVOT_LOWER_S: float = 0.5          # seconds

# How long the pivot rises to return to pick height.
PIVOT_RAISE_S: float = 0.5          # seconds

# Pause between pivot stop and gripper open (let swing damp out).
PIVOT_SETTLE_S: float = 0.2         # seconds

# Extra wait after gripper opens before the robot moves away.
GRIPPER_DROP_WAIT_S: float = 0.5    # seconds

# Per-row extra pivot travel (fraction of PIVOT_LOWER_S per row step).
ROW_STEP_RATIO: float = 0.25

# Per-column extra turntable nudge (fraction of SIDE_TRAVEL_S per col step).
COL_STEP_RATIO: float = 0.12

# Grid dimensions.
GRID_COLS: int = 3
GRID_ROWS: int = 3

# =============================================================================
# STATE
# =============================================================================

_lock                         = threading.Lock()
_slot_index: int              = 0
_busy: bool                   = False

# =============================================================================
# GRID HELPERS
# =============================================================================

def _slot_to_row_col(slot: int):
    return divmod(slot, GRID_COLS)


def grid_full() -> bool:
    with _lock:
        return _slot_index >= GRID_ROWS * GRID_COLS


def collected_count() -> int:
    """Number of berries placed in the current bin run."""
    with _lock:
        return _slot_index


def reset() -> None:
    """Reset slot counter — call when the bin has been emptied / swapped."""
    global _slot_index
    with _lock:
        _slot_index = 0
    print("[bin] Reset — slot counter cleared, starting fresh bin.")

# =============================================================================
# LOW-LEVEL HARDWARE HELPERS
# =============================================================================

def _home_turntable() -> None:
    """Return turntable to its zero-point using its own _home() logic."""
    if not _HAS_TURNTABLE:
        print("[bin][SIM] turntable _home()")
        return
    print("[bin] Homing turntable…")
    _turntable._home()
    print("[bin] Turntable at zero.")


def _turntable_move(direction: str, duration: float) -> None:
    """Spin turntable in direction for duration seconds, then stop."""
    if not _HAS_TURNTABLE:
        print(f"[bin][SIM] turntable {direction} for {duration:.2f}s")
        time.sleep(duration)
        return

    if direction == "left":
        _turntable.spin_left()
    else:
        _turntable.spin_right()

    time.sleep(duration)
    _turntable.stop()


def _pivot_lower(duration: float) -> None:
    if not _HAS_PIVOT:
        print(f"[bin][SIM] pivot lower for {duration:.2f}s")
        time.sleep(duration)
        return
    _pivot.move_down()
    time.sleep(duration)
    _pivot.stop()


def _pivot_raise(duration: float) -> None:
    if not _HAS_PIVOT:
        print(f"[bin][SIM] pivot raise for {duration:.2f}s")
        time.sleep(duration)
        return
    _pivot.move_up()
    time.sleep(duration)
    _pivot.stop()


def _open_gripper() -> None:
    if not _HAS_GRIPPER:
        print("[bin][SIM] gripper open")
        return
    result = _gripper.open_gripper()
    print(f"[bin] gripper.open → {result}")
    deadline = time.monotonic() + 6.0
    while time.monotonic() < deadline:
        if _gripper.get_state() != "BUSY":
            break
        time.sleep(0.1)

# =============================================================================
# SLOT OFFSET MOVEMENTS
# =============================================================================

def _apply_slot_offsets(row: int, col: int) -> tuple[float, float]:
    """
    Move pivot and turntable to align with the target slot.

    Centre row/col have zero offset; outer slots get small extra moves.
    Returns (row_extra_s, col_extra_s) so the caller can undo them.
    """
    centre_row = (GRID_ROWS - 1) // 2
    centre_col = (GRID_COLS - 1) // 2

    row_offset = row - centre_row   # +1 = one row lower, -1 = one row higher
    col_offset = col - centre_col   # +1 = one col right, -1 = one col left

    print(f"[bin] Slot offsets: row={row_offset:+d}, col={col_offset:+d}")

    row_extra_s = abs(row_offset) * PIVOT_LOWER_S * ROW_STEP_RATIO
    col_extra_s = abs(col_offset) * SIDE_TRAVEL_S * COL_STEP_RATIO

    # Row → extra pivot move
    if row_offset > 0:
        _pivot_lower(row_extra_s)
    elif row_offset < 0:
        _pivot_raise(row_extra_s)

    # Col → extra turntable nudge (robot already at 90°, so left/right are swapped)
    if col_offset > 0:
        nudge_dir = BIN_SIDE
    elif col_offset < 0:
        nudge_dir = "right" if BIN_SIDE == "left" else "left"
    else:
        nudge_dir = None

    if nudge_dir:
        _turntable_move(nudge_dir, col_extra_s)
        time.sleep(TURNTABLE_SETTLE_S)

    return row_extra_s, col_extra_s

# =============================================================================
# MAIN SEQUENCE
# =============================================================================

def place_berry() -> bool:
    """
    Full post-grip deposit sequence.  Blocking.

    Returns True on success, False if already busy.
    Auto-resets the bin counter when the last slot is filled.
    """
    global _busy, _slot_index

    with _lock:
        if _busy:
            print("[bin] place_berry() called while already busy — ignored.")
            return False
        current_slot = _slot_index
        _busy = True

    try:
        row, col = _slot_to_row_col(current_slot)
        total    = GRID_ROWS * GRID_COLS

        print(
            f"[bin] ── Placing berry #{current_slot + 1}/{total} "
            f"(slot {current_slot}: row={row}, col={col}) ──"
        )

        # ── Step 1 — Home turntable (returns to zero / pick-origin) ──────────
        print("[bin] Step 1 — homing turntable to zero")
        _home_turntable()
        time.sleep(TURNTABLE_SETTLE_S)

        # ── Step 2 — Rotate 90° toward bin ───────────────────────────────────
        print(f"[bin] Step 2 — rotating {BIN_SIDE} toward bin ({SIDE_TRAVEL_S:.2f}s)")
        _turntable_move(BIN_SIDE, SIDE_TRAVEL_S)
        time.sleep(TURNTABLE_SETTLE_S)

        # ── Step 3 — Align with correct slot ─────────────────────────────────
        print("[bin] Step 3 — applying slot offsets")
        row_extra_s, col_extra_s = _apply_slot_offsets(row, col)

        # ── Step 4 — Lower pivot above slot ──────────────────────────────────
        print(f"[bin] Step 4 — lowering pivot ({PIVOT_LOWER_S:.2f}s)")
        _pivot_lower(PIVOT_LOWER_S)
        time.sleep(PIVOT_SETTLE_S)

        # ── Step 5 — Release berry ────────────────────────────────────────────
        print("[bin] Step 5 — opening gripper")
        _open_gripper()
        time.sleep(GRIPPER_DROP_WAIT_S)

        # ── Step 6 — Raise pivot back to pick height ──────────────────────────
        print(f"[bin] Step 6 — raising pivot ({PIVOT_RAISE_S:.2f}s)")
        _pivot_raise(PIVOT_RAISE_S)

        # Undo row extra (pivot)
        centre_row = (GRID_ROWS - 1) // 2
        row_offset = row - centre_row
        if row_offset > 0 and row_extra_s > 0:
            _pivot_raise(row_extra_s)
        elif row_offset < 0 and row_extra_s > 0:
            _pivot_lower(row_extra_s)

        # Undo col extra (turntable nudge)
        centre_col = (GRID_COLS - 1) // 2
        col_offset = col - centre_col
        if col_offset != 0 and col_extra_s > 0:
            undo_dir = ("right" if BIN_SIDE == "left" else "left") if col_offset > 0 else BIN_SIDE
            _turntable_move(undo_dir, col_extra_s)
            time.sleep(TURNTABLE_SETTLE_S)

        # ── Step 7 — Return to zero (home), ready for next pick ───────────────
        print(f"[bin] Step 7 — returning turntable to zero")
        _home_turntable()
        time.sleep(TURNTABLE_SETTLE_S)

        # ── Step 8 — Advance counter; auto-reset if bin is now full ───────────
        with _lock:
            _slot_index += 1
            new_count  = _slot_index
            bin_full   = new_count >= total

        if bin_full:
            print(f"[bin] ── Bin full ({total}/{total}) — auto-resetting for next bin ──")
            reset()
        else:
            print(f"[bin] ── Berry placed. {new_count}/{total} slots filled ──")

        return True

    except Exception as e:
        print(f"[bin] ERROR during place_berry(): {e}")
        return False

    finally:
        with _lock:
            _busy = False

# =============================================================================
# STATUS
# =============================================================================

def status() -> dict:
    with _lock:
        idx   = _slot_index
        total = GRID_ROWS * GRID_COLS
        busy  = _busy

    slots = []
    for i in range(total):
        r, c = _slot_to_row_col(i)
        slots.append({"index": i, "row": r, "col": c, "filled": i < idx})

    return {
        "collected": idx,
        "capacity":  total,
        "full":      idx >= total,
        "busy":      busy,
        "bin_side":  BIN_SIDE,
        "grid":      f"{GRID_ROWS}x{GRID_COLS}",
        "slots":     slots,
    }

# =============================================================================
# STANDALONE TEST
# =============================================================================

if __name__ == "__main__":
    print("=== bin.py standalone test ===")
    print(f"Grid: {GRID_ROWS}×{GRID_COLS}, bin side: {BIN_SIDE}, travel: {SIDE_TRAVEL_S}s\n")

    for i in range(GRID_ROWS * GRID_COLS + 1):   # +1 to verify auto-reset
        print(f"\n--- Simulating grab #{i + 1} ---")
        ok = place_berry()
        print(f"    place_berry() → {ok} | status: {status()}")
        time.sleep(0.1)