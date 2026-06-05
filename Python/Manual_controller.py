"""
manual_controller.py
====================
Ontvangt UDP-pakketten van de ESP32-controller, stuurt alle servo's aan,
en bewaakt de verbindingsstatus.

De ESP32 stuurt continu pakketten — ook zonder invoer. Zolang pakketten
binnenkomen is de controller verbonden. Zodra er DISCONNECT_TIMEOUT seconden
niets binnenkomt, wordt de verbinding als verloren beschouwd en schakelt
het systeem terug naar autonomous.

Controller-layout
-----------------
    Linker stick  X  →  turntable    links/rechts   servo ID 13
    Linker stick  Y  →  lift         omhoog/omlaag  servo ID 3 + 4
    Rechter stick X  →  pivot        omhoog/omlaag  servo ID 2
    Rechter stick Y  →  arm          voor/achter    servo ID 5
    Gripper-knop     →  toggle open ↔ dicht         servo ID 8

UDP-pakketformaat (JSON)
-------------------------
    {
        "lx":   -100..100,   linker stick X
        "ly":   -100..100,   linker stick Y
        "rx":   -100..100,   rechter stick X
        "ry":   -100..100,   rechter stick Y
        "grip": 0 | 1        gripper-knop
    }

    Alle velden optioneel; ontbrekend = 0.
    "grip" toggle werkt op stijgende flank (één toggle per druk).
"""

import json
import socket
import threading
import time
from typing import Optional

# ---------------------------------------------------------------------------
# Optionele imports — elk subsysteem werkt ook zonder de andere modules.
# ---------------------------------------------------------------------------

try:
    import turntable as _turntable
    _HAS_TURNTABLE = True
except ImportError:
    _HAS_TURNTABLE = False

try:
    import lift as _lift
    _HAS_LIFT = True
except ImportError:
    _HAS_LIFT = False

try:
    import arm as _arm
    _HAS_ARM = True
except ImportError:
    _HAS_ARM = False

try:
    import pivot as _pivot
    _HAS_PIVOT = True
except ImportError:
    _HAS_PIVOT = False

try:
    import gripper as _gripper
    _HAS_GRIPPER = True
except ImportError:
    _HAS_GRIPPER = False


# =============================================================================
# CONFIGURATIE
# =============================================================================

UDP_HOST = "0.0.0.0"
UDP_PORT = 5005

# Hoe lang geen pakket voordat de verbinding als verloren wordt beschouwd.
# Stel in op ~3× de zendinterval van de ESP32.
DISCONNECT_TIMEOUT = 1.0

SOCKET_TIMEOUT = 0.2
DEADZONE       = 15
SPEED_MAX      = 800   # maximale snelheid die naar servo's wordt gestuurd


# =============================================================================
# INTERNE STATE
# =============================================================================

_running = False
_thread: Optional[threading.Thread] = None
_sock:   Optional[socket.socket]    = None

_last_packet_time: float = 0.0   # monotonic, bijgewerkt bij elk pakket

_gripper_open  = True
_grip_btn_prev = 0


# =============================================================================
# HULPFUNCTIES
# =============================================================================

def _joystick_to_speed(value: int) -> int:
    """Zet joystick-waarde (-100..100) om naar servo-snelheid (0..SPEED_MAX)."""
    if abs(value) <= DEADZONE:
        return 0
    ratio = (abs(value) - DEADZONE) / (100 - DEADZONE)
    return int(ratio * SPEED_MAX)


def _apply_input(lx: int, ly: int, rx: int, ry: int, grip: int) -> None:
    """
    Vertaal joystick-waarden naar servo-commando's.

    lx  →  turntable   (links/rechts,   servo 13)
    ly  →  lift        (omhoog/omlaag,  servo 3+4)
    rx  →  pivot       (omhoog/omlaag,  servo 2)
    ry  →  arm         (voor/achter,    servo 5)
    grip → gripper     (toggle,         servo 8)
    """
    global _gripper_open, _grip_btn_prev

    # --- Turntable (linker stick X) ---
    if _HAS_TURNTABLE:
        speed = _joystick_to_speed(lx)
        if speed == 0:
            _turntable.stop()
        elif lx > 0:
            _turntable.spin_right(speed)
        else:
            _turntable.spin_left(speed)

    # --- Lift (linker stick Y) ---
    if _HAS_LIFT:
        speed = _joystick_to_speed(ly)
        if speed == 0:
            _lift.stop()
        elif ly > 0:
            _lift.move_up(speed)
        else:
            _lift.move_down(speed)

    # --- Pivot / draaipunt gripper (rechter stick X) ---
    if _HAS_PIVOT:
        speed = _joystick_to_speed(rx)
        if speed == 0:
            _pivot.stop()
        elif rx > 0:
            _pivot.rotate_down(speed)
        else:
            _pivot.rotate_up(speed)

    # --- Arm voor/achter (rechter stick Y) ---
    if _HAS_ARM:
        speed = _joystick_to_speed(ry)
        if speed == 0:
            _arm.stop()
        elif ry > 0:
            _arm.move_forward(speed)
        else:
            _arm.move_backward(speed)

    # --- Gripper toggle (knop, stijgende flank) ---
    if _HAS_GRIPPER:
        if grip and not _grip_btn_prev:
            _gripper_open = not _gripper_open
            if _gripper_open:
                _gripper.open_gripper()
            else:
                _gripper.grip()

    _grip_btn_prev = grip


def _stop_all() -> None:
    """Stop alle bewegende subsystemen op een veilige manier."""
    if _HAS_TURNTABLE:
        try:
            _turntable.stop()
        except Exception:
            pass
    if _HAS_LIFT:
        try:
            _lift.stop()
        except Exception:
            pass
    if _HAS_ARM:
        try:
            _arm.stop()
        except Exception:
            pass
    if _HAS_PIVOT:
        try:
            _pivot.stop()
        except Exception:
            pass


# =============================================================================
# UDP LISTENER
# =============================================================================

def _listener() -> None:
    global _last_packet_time

    import control_mode

    print(f"[manual] Luistert op {UDP_HOST}:{UDP_PORT} "
          f"(verbreking na {DISCONNECT_TIMEOUT}s)")

    connected = False

    while _running:
        # --- Pakket ontvangen ---
        try:
            data, _ = _sock.recvfrom(512)
        except socket.timeout:
            # Geen pakket in dit venster — check watchdog
            if connected and (time.monotonic() - _last_packet_time) >= DISCONNECT_TIMEOUT:
                print("[manual] Verbinding verbroken — overschakelen naar autonomous.")
                connected = False
                _stop_all()
                control_mode._set_mode("autonomous")
            continue
        except OSError:
            break

        # --- Pakket ontvangen: verbinding levend ---
        _last_packet_time = time.monotonic()

        if not connected:
            print("[manual] Controller verbonden.")
            connected = True
            control_mode._set_mode("manual")

        # --- JSON parseren ---
        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[manual] Ongeldig pakket: {e}")
            continue

        lx   = int(payload.get("lx",   0))
        ly   = int(payload.get("ly",   0))
        rx   = int(payload.get("rx",   0))
        ry   = int(payload.get("ry",   0))
        grip = int(bool(payload.get("grip", 0)))

        try:
            _apply_input(lx, ly, rx, ry, grip)
        except Exception as e:
            print(f"[manual] Fout bij servo-aansturing: {e}")

    _stop_all()
    print("[manual] Listener gestopt.")


# =============================================================================
# PUBLIC API
# =============================================================================

def start() -> None:
    global _running, _thread, _sock, _last_packet_time

    if _running:
        return

    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _sock.settimeout(SOCKET_TIMEOUT)
    _sock.bind((UDP_HOST, UDP_PORT))

    _last_packet_time = 0.0
    _running = True

    _thread = threading.Thread(
        target=_listener,
        daemon=True,
        name="manual-udp-listener",
    )
    _thread.start()


def stop() -> None:
    global _running, _sock

    if not _running:
        return

    _running = False

    if _sock is not None:
        try:
            _sock.close()
        except Exception:
            pass
        _sock = None

    if _thread is not None:
        _thread.join(timeout=2.0)

    _stop_all()
    print("[manual] Gestopt.")


def is_active() -> bool:
    return _running