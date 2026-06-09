"""
servo_status.py
===============
Thread-safe store for the last-known command sent to every servo.

This module is the single source of truth for the UI servo status overlay.
It works without any hardware — every servo module calls update() after
posting a command, and fusion_engine reads get_all() to draw the panel.

Usage
-----
    import servo_status
    servo_status.update(13, "RIGHT", 300)   # called by turntable.py
    rows = servo_status.get_all()           # called by draw_annotations
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, List

# =============================================================================
# SERVO REGISTRY
# =============================================================================

# All known servo IDs with their human-readable name and module owner.
# Order here is the order they appear in the UI panel.
_SERVO_NAMES: Dict[int, str] = {
    2:  "Pivot",
    3:  "Lift A",
    4:  "Lift B",
    5:  "Arm",
    8:  "Gripper",
    13: "Turntable",
}


@dataclass
class ServoState:
    id:        int
    name:      str
    status:    str  = "STOP"    # e.g. "STOP", "FORWARD", "RIGHT", "GRIP", …
    speed:     int  = 0         # 0–1023
    simulated: bool = True      # False once real hardware has been seen


# =============================================================================
# MODULE STATE
# =============================================================================

_lock: threading.Lock = threading.Lock()
_states: Dict[int, ServoState] = {
    sid: ServoState(id=sid, name=name)
    for sid, name in _SERVO_NAMES.items()
}


# =============================================================================
# PUBLIC API
# =============================================================================

def update(servo_id: int, status: str, speed: int = 0, *, real: bool = False) -> None:
    """
    Record the latest command for a servo.

    Args:
        servo_id: AX-12A servo ID.
        status:   Human-readable action string, e.g. "STOP", "LEFT", "GRIP".
        speed:    Speed value 0–1023 (0 when stopped).
        real:     Pass True when called from actual hardware code so the UI
                  can show "REAL" instead of "SIM".
    """
    with _lock:
        if servo_id not in _states:
            # Unknown servo — register it on the fly.
            _states[servo_id] = ServoState(
                id=servo_id,
                name=f"Servo {servo_id}",
            )
        s = _states[servo_id]
        s.status    = status.upper()
        s.speed     = max(0, min(1023, speed))
        s.simulated = not real


def get_all() -> List[ServoState]:
    """Return a snapshot of all servo states, sorted by servo ID."""
    with _lock:
        return [ServoState(
                    id=s.id,
                    name=s.name,
                    status=s.status,
                    speed=s.speed,
                    simulated=s.simulated,
                )
                for s in sorted(_states.values(), key=lambda x: x.id)]