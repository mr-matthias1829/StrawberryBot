"""
manual_controller.py
====================
Ontvangt UDP-pakketten van de zelfgebouwde ESP32-controller en stuurt
de servo's direct aan op basis van de joystick/button-waarden.

Servo toewijzing
----------------
    ID 13  →  turntable   (links/rechts draaien)
    ID 3+4 →  lift        (arm omhoog/omlaag)
    ID 8   →  gripper     (openen / sluiten)

UDP-pakketformaat (JSON, max ~256 bytes)
-----------------------------------------
    {
        "lx": -100..100,   // linker joystick X  → turntable
        "ly": -100..100,   // linker joystick Y  → lift
        "rb": 0 | 1,       // rechter bumper     → grip
        "lb": 0 | 1        // linker  bumper     → open gripper
    }

    Alle velden zijn optioneel; ontbrekende velden worden als 0 behandeld.

Gebruik
-------
    import manual_controller
    manual_controller.start()          # start UDP-listener thread
    manual_controller.stop()           # stop listener thread

Integratie met mode-wisseling
------------------------------
    Zie control_mode.py voor AUTONOMOUS ↔ MANUAL schakelaar.
"""

import json
import socket
import threading
import time
from typing import Optional

# Probeer hardware-modules te laden; geen import-fout op niet-Pi hardware
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
    import gripper as _gripper
    _HAS_GRIPPER = True
except ImportError:
    _HAS_GRIPPER = False


# =============================================================================
# CONFIGURATIE
# =============================================================================

UDP_HOST = "0.0.0.0"   # luister op alle interfaces
UDP_PORT = 5005         # zelfde poort als op de ESP32 ingesteld

DEADZONE          = 15  # joystick-waarden binnen ±DEADZONE worden genegeerd
SOCKET_TIMEOUT    = 0.5 # seconden — zodat de thread netjes kan stoppen
WATCHDOG_TIMEOUT  = 0.3 # seconden zonder pakket → stuur stop naar alle assen

# Snelheidscurve: joystick 0-100 → servo speed 0-1023
_SPEED_MAX = 800        # maximale servo-snelheid bij volle joystick-uitslag


# =============================================================================
# INTERNE STATE
# =============================================================================

_running      = False
_thread: Optional[threading.Thread] = None
_sock:   Optional[socket.socket]    = None
_lock         = threading.Lock()
_last_packet  = 0.0    # timestamp laatste ontvangen pakket


# =============================================================================
# HULPFUNCTIES
# =============================================================================

def _joystick_to_speed(value: int) -> int:
    """
    Schaalt een joystick-waarde (-100..100) naar een servo-snelheid (0..SPEED_MAX).
    Waarden binnen de DEADZONE geven 0 terug.
    """
    if abs(value) <= DEADZONE:
        return 0
    # Lineaire schaling buiten de dode zone
    ratio = (abs(value) - DEADZONE) / (100 - DEADZONE)
    return int(ratio * _SPEED_MAX)


def _apply_input(lx: int, ly: int, rb: int, lb: int) -> None:
    """
    Vertaalt controller-invoer naar servo-commando's.

    lx  →  turntable (links/rechts)
    ly  →  lift      (omhoog/omlaag)  — negatieve Y = joystick omhoog = arm omhoog
    rb  →  gripper grip
    lb  →  gripper open
    """
    # -- Turntable --
    if _HAS_TURNTABLE:
        speed = _joystick_to_speed(lx)
        if speed == 0:
            _turntable.stop()
        elif lx > 0:
            _turntable.spin_right(speed)
        else:
            _turntable.spin_left(speed)

    # -- Lift --
    if _HAS_LIFT:
        speed = _joystick_to_speed(ly)
        if speed == 0:
            _lift.stop()
        elif ly > 0:
            # joystick naar beneden → arm omhoog (conventie: positieve Y = push down)
            _lift.move_up(speed)
        else:
            # joystick naar boven → arm omlaag
            _lift.move_down(speed)

    # -- Gripper --
    if _HAS_GRIPPER:
        if rb:
            _gripper.grip()
        elif lb:
            _gripper.open_gripper()


def _stop_all() -> None:
    """Stuur stop naar alle assen (watchdog / stop bij mode-wissel)."""
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


# =============================================================================
# UDP LISTENER THREAD
# =============================================================================

def _listener() -> None:
    global _last_packet

    print(f"[manual] UDP-listener gestart op {UDP_HOST}:{UDP_PORT}")

    while _running:
        # Watchdog: als er te lang geen pakket is, stop alle assen
        now = time.monotonic()
        if _last_packet > 0 and (now - _last_packet) > WATCHDOG_TIMEOUT:
            _stop_all()
            _last_packet = 0.0  # reset zodat we niet elke loop stoppen

        try:
            data, _ = _sock.recvfrom(512)
        except socket.timeout:
            continue
        except OSError:
            # Socket gesloten door stop()
            break

        try:
            payload = json.loads(data.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            print(f"[manual] Ongeldig pakket ontvangen: {e}")
            continue

        lx = int(payload.get("lx", 0))
        ly = int(payload.get("ly", 0))
        rb = int(bool(payload.get("rb", 0)))
        lb = int(bool(payload.get("lb", 0)))

        _last_packet = time.monotonic()

        try:
            _apply_input(lx, ly, rb, lb)
        except Exception as e:
            print(f"[manual] Fout bij aansturen servo's: {e}")

    _stop_all()
    print("[manual] UDP-listener gestopt.")


# =============================================================================
# PUBLIEKE API
# =============================================================================

def start() -> None:
    """Start de UDP-listener thread. Idempotent."""
    global _running, _thread, _sock, _last_packet

    if _running:
        return

    _sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _sock.settimeout(SOCKET_TIMEOUT)
    _sock.bind((UDP_HOST, UDP_PORT))

    _last_packet = 0.0
    _running = True

    _thread = threading.Thread(
        target=_listener,
        daemon=True,
        name="manual-udp-listener",
    )
    _thread.start()
    print("[manual] Manual controller geactiveerd.")


def stop() -> None:
    """Stop de UDP-listener thread en stuur stop naar alle servo's."""
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
    print("[manual] Manual controller gedeactiveerd.")


def is_active() -> bool:
    """Geeft True als de manual controller momenteel actief is."""
    return _running