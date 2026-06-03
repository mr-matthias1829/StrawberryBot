"""
control_mode.py
===============
Centrale schakelaar tussen AUTONOMOUS en MANUAL besturingsmodus.

Gebruik
-------
    import control_mode

    control_mode.set_mode("manual")     # schakel naar handmatig
    control_mode.set_mode("autonomous") # schakel naar autonoom
    control_mode.toggle()               # wissel tussen de twee

    if control_mode.is_manual():
        ...                             # sla autonome verwerking over

Integratie in run_webcam() (main.py)
-------------------------------------
    Voeg bovenaan de main-loop toe:

        if control_mode.is_manual():
            continue   # sla FusionEngine-verwerking over

    Of gebruik de callback om bij mode-wissel hardware te resetten:

        control_mode.on_change(my_callback)

UDP toggle (optioneel)
-----------------------
    De ESP32 kan ook een mode-wissel sturen via UDP op MODE_UDP_PORT:
        { "mode": "manual" }    of    { "mode": "autonomous" }
    Zet USE_UDP_TOGGLE = True om dit in te schakelen.
"""

import json
import socket
import threading
import time
from typing import Callable, List, Optional

# =============================================================================
# CONFIGURATIE
# =============================================================================

# Startmodus bij opstarten
DEFAULT_MODE = "autonomous"   # "autonomous" | "manual"

# Optionele UDP-toggle (aparte poort van manual_controller)
USE_UDP_TOGGLE  = True
MODE_UDP_PORT   = 5006
SOCKET_TIMEOUT  = 0.5


# =============================================================================
# STATE
# =============================================================================

_mode: str = DEFAULT_MODE
_lock = threading.Lock()
_callbacks: List[Callable[[str], None]] = []

_udp_thread: Optional[threading.Thread] = None
_udp_sock:   Optional[socket.socket]    = None
_udp_running = False


# =============================================================================
# CORE API
# =============================================================================

def get_mode() -> str:
    """Geeft de huidige modus: 'autonomous' of 'manual'."""
    with _lock:
        return _mode


def is_manual() -> bool:
    with _lock:
        return _mode == "manual"


def is_autonomous() -> bool:
    with _lock:
        return _mode == "autonomous"


def set_mode(new_mode: str) -> None:
    """
    Schakel naar de opgegeven modus.

    Args:
        new_mode: "autonomous" of "manual"
    """
    global _mode

    new_mode = new_mode.lower().strip()
    if new_mode not in ("autonomous", "manual"):
        print(f"[control_mode] Onbekende modus: '{new_mode}' — genegeerd.")
        return

    with _lock:
        if _mode == new_mode:
            return
        old_mode = _mode
        _mode = new_mode

    print(f"[control_mode] Modus gewisseld: {old_mode} → {new_mode}")

    # Activeer / deactiveer manual_controller
    try:
        import manual_controller
        if new_mode == "manual":
            manual_controller.start()
        else:
            manual_controller.stop()
    except ImportError:
        pass

    # Roep geregistreerde callbacks aan
    for cb in list(_callbacks):
        try:
            cb(new_mode)
        except Exception as e:
            print(f"[control_mode] Callback fout: {e}")


def toggle() -> str:
    """Wissel tussen de twee modi. Geeft de nieuwe modus terug."""
    with _lock:
        current = _mode

    new_mode = "manual" if current == "autonomous" else "autonomous"
    set_mode(new_mode)
    return new_mode


def on_change(callback: Callable[[str], None]) -> None:
    """
    Registreer een callback die aangeroepen wordt bij elke mode-wissel.

    Args:
        callback: functie die de nieuwe modus-string ontvangt
    """
    _callbacks.append(callback)


# =============================================================================
# UDP TOGGLE LISTENER (optioneel)
# =============================================================================

def _udp_listener() -> None:
    print(f"[control_mode] UDP mode-toggle luistert op poort {MODE_UDP_PORT}")

    while _udp_running:
        try:
            data, addr = _udp_sock.recvfrom(256)
        except socket.timeout:
            continue
        except OSError:
            break

        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if "mode" in payload:
            set_mode(str(payload["mode"]))
        elif payload.get("toggle"):
            toggle()

    print("[control_mode] UDP mode-toggle gestopt.")


def start_udp_toggle() -> None:
    """Start de UDP mode-toggle listener (apart van manual_controller)."""
    global _udp_running, _udp_thread, _udp_sock

    if not USE_UDP_TOGGLE or _udp_running:
        return

    _udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _udp_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _udp_sock.settimeout(SOCKET_TIMEOUT)
    _udp_sock.bind(("0.0.0.0", MODE_UDP_PORT))

    _udp_running = True
    _udp_thread = threading.Thread(
        target=_udp_listener,
        daemon=True,
        name="control-mode-udp",
    )
    _udp_thread.start()


def stop_udp_toggle() -> None:
    """Stop de UDP mode-toggle listener."""
    global _udp_running, _udp_sock

    _udp_running = False

    if _udp_sock is not None:
        try:
            _udp_sock.close()
        except Exception:
            pass
        _udp_sock = None

    if _udp_thread is not None:
        _udp_thread.join(timeout=2.0)