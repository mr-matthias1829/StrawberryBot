"""
control_mode.py
===============
Centrale modus-state voor AUTONOMOUS ↔ MANUAL.

De verbindingsdetectie en watchdog zitten volledig in manual_controller.py.
Deze module beheert alleen de state en de callbacks.

Gebruik
-------
    import control_mode

    control_mode.start()         # zet modus op autonomous, start manual_controller
    control_mode.stop()          # stop alles

    if control_mode.is_manual():
        continue                 # sla FusionEngine-verwerking over

    control_mode.on_change(cb)   # callback bij iedere wissel
"""

import threading
from typing import Callable, List

# =============================================================================
# STATE
# =============================================================================

_mode: str = "autonomous"
_lock = threading.Lock()
_callbacks: List[Callable[[str], None]] = []


# =============================================================================
# PUBLIEKE API
# =============================================================================

def get_mode() -> str:
    with _lock:
        return _mode


def is_manual() -> bool:
    with _lock:
        return _mode == "manual"


def is_autonomous() -> bool:
    with _lock:
        return _mode == "autonomous"


def on_change(callback: Callable[[str], None]) -> None:
    """Registreer een callback die de nieuwe modus-string ontvangt."""
    _callbacks.append(callback)


def start() -> None:
    """Start het systeem: zet modus op autonomous en start manual_controller."""
    _set_mode("autonomous")

    import Manual_controller
    Manual_controller.start()
    print("[control_mode] Gestart. Wacht op controller...")


def stop() -> None:
    """Stop alles en zet modus terug op autonomous."""
    import Manual_controller
    Manual_controller.stop()

    _set_mode("autonomous")
    print("[control_mode] Gestopt.")


# =============================================================================
# INTERNE API (aangeroepen door manual_controller)
# =============================================================================

def _set_mode(new_mode: str) -> None:
    """Schakel naar new_mode en roep callbacks aan. Aangeroepen door manual_controller."""
    global _mode

    with _lock:
        if _mode == new_mode:
            return
        old_mode = _mode
        _mode = new_mode

    print(f"[control_mode] {old_mode} → {new_mode}")

    for cb in list(_callbacks):
        try:
            cb(new_mode)
        except Exception as e:
            print(f"[control_mode] Callback fout: {e}")