"""
dynamixel.py
============
Reusable Dynamixel servo controller for Raspberry Pi (RS-485, daisy-chained).

HARDWARE ASSUMPTIONS (fixed, never change these):
  - Direction pin : GPIO 17
  - Serial port   : /dev/ttyAMA0
  - Baudrate      : 1,000,000

CHAINING NOTE:
  Dynamixel servos are designed to be daisy-chained on a single RS-485 bus.
  Each servo listens for packets addressed to its own ID and ignores the rest.
  The code does not need to know or care about chain order.

PERSISTENT STATE:
  Every servo's configuration (min/pos limits, last known position, mode, etc.)
  is saved to a JSON file (servo_state.json by default) after every change.
  On the next boot, ServoCon loads that file and restores all settings
  automatically — no need to reconfigure anything.

QUICK USAGE EXAMPLE:
  from dynamixel import ServoController, JointServo, WheelServo, safe_shutdown

  with ServoController() as ctrl:
      arm    = JointServo(ctrl, servo_id=1, min_pos=300, max_pos=700)
      drive  = WheelServo(ctrl, servo_id=2, max_speed=400)

      arm.move_to_percent(0.5)   # center
      drive.spin(200, clockwise=True)
      drive.stop()

  # On restart, limits are automatically loaded from servo_state.json
  with ServoController() as ctrl:
      arm = JointServo(ctrl, servo_id=1)  # min/max restored from disk
"""

import json
import logging
import os
import time
from typing import Optional

import platform

ON_PI = platform.system() == "Linux"

if ON_PI:
    import lgpio


import serial

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# FIXED HARDWARE CONSTANTS  (never change these)
# ─────────────────────────────────────────────────────────────────────────────

DIRECTION_PIN = 17             # GPIO pin that switches RS-485 between TX and RX
SERIAL_PORT   = "/dev/ttyAMA0" # UART port the servo bus is connected to
BAUDRATE      = 1_000_000      # Dynamixel standard baud rate

# ─────────────────────────────────────────────────────────────────────────────
# DYNAMIXEL REGISTER ADDRESSES
# These are memory addresses inside each servo. Writing to them changes
# behaviour; reading from them returns current state.
# ─────────────────────────────────────────────────────────────────────────────

REG_CW_ANGLE_LIMIT   = 6   # EEPROM. Clockwise (lower) joint limit (0–1023).
                            # Setting both this AND CCW to 0 enables wheel mode.
REG_CCW_ANGLE_LIMIT  = 8   # EEPROM. Counter-clockwise (upper) joint limit.
REG_TORQUE_ENABLE    = 24  # RAM. 1 = motor powered, 0 = motor free-wheeling.
REG_GOAL_POSITION    = 30  # RAM. Target position for joint mode (0–1023).
REG_MOVING_SPEED     = 32  # RAM. Speed (joint: 1–1023 RPM-ish;
                            #            wheel: 0–1023 CCW, 1024–2047 CW).
REG_PRESENT_POSITION = 36  # RAM. Read-only. Current shaft position (0–1023).
REG_PRESENT_LOAD     = 40  # RAM. Read-only. Current load on motor (0–2047).
REG_MOVING           = 46  # RAM. Read-only. 1 if servo is still in motion.

# Instruction codes sent in every packet
INST_WRITE = 0x03
INST_READ  = 0x02

# ─────────────────────────────────────────────────────────────────────────────
# POSITION AND SPEED LIMITS  (physical hardware constants for AX/MX series)
# ─────────────────────────────────────────────────────────────────────────────

POSITION_MIN_ABSOLUTE = 0     # Absolute minimum position value (full CCW)
POSITION_MAX_ABSOLUTE = 1023  # Absolute maximum position value (full CW)
SPEED_MIN             = 1     # Speed 0 = uncontrolled max — always use at least 1
SPEED_MAX             = 1023  # Maximum speed register value


class DynamixelError(Exception):
    """Raised when a servo command is invalid or cannot be safely executed."""


# ─────────────────────────────────────────────────────────────────────────────
# PERSISTENT STATE  (saved to disk so config survives restarts)
# ─────────────────────────────────────────────────────────────────────────────

class ServoStateStore:
    """
    Loads and saves servo configuration to a JSON file on disk.

    Each servo is stored as an entry keyed by its ID (as a string).
    The file is written every time any value changes.

    Stored per servo
    ----------------
    id           : int   — Dynamixel servo ID (1–253)
    mode         : str   — "joint" or "wheel"
    min_pos      : int   — Minimum allowed position (joint mode only)
    max_pos      : int   — Maximum allowed position (joint mode only)
    max_speed    : int   — Hard speed cap
    last_pos     : int   — Last commanded/known position (joint mode)
    torque_on    : bool  — Whether torque was on when last saved
    label        : str   — Optional human-readable name (e.g. "left_arm")

    Parameters
    ----------
    path : str
        Path to the JSON state file. Created automatically if it does not exist.
    """

    def __init__(self, path: str = "servo_state.json"):
        self.path  = path
        self._data: dict = {}  # keyed by str(servo_id)
        self._load()

    # ── internal ──────────────────────────────────────────────────────────────

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r") as f:
                    self._data = json.load(f)
                logger.info("Loaded servo state from %s", self.path)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Could not load %s: %s — starting fresh", self.path, e)
                self._data = {}
        else:
            logger.info("No state file found at %s — will create on first save", self.path)

    def _save(self):
        try:
            with open(self.path, "w") as f:
                json.dump(self._data, f, indent=2)
        except OSError as e:
            logger.error("Could not save servo state: %s", e)

    # ── public API ────────────────────────────────────────────────────────────

    def get(self, servo_id: int) -> Optional[dict]:
        """Return stored config dict for a servo, or None if not seen before."""
        return self._data.get(str(servo_id))

    def update(self, servo_id: int, **kwargs):
        """
        Update one or more fields for a servo and immediately save to disk.

        Example
        -------
        store.update(1, last_pos=512, torque_on=False)
        """
        key = str(servo_id)
        if key not in self._data:
            self._data[key] = {"id": servo_id}
        self._data[key].update(kwargs)
        self._save()

    def all_ids(self) -> list[int]:
        """Return the list of all servo IDs currently stored."""
        return [int(k) for k in self._data]


# ─────────────────────────────────────────────────────────────────────────────
# SERVO CONTROLLER  (one per bus, shared by all servos)
# ─────────────────────────────────────────────────────────────────────────────

class ServoController:
    """
    Low-level RS-485 bus controller. One instance shared by all servos.

    Owns the GPIO chip and the serial port. Use as a context manager so
    the port is always closed cleanly, even if the code crashes:

        with ServoController() as ctrl:
            arm = JointServo(ctrl, 1, ...)

    Parameters
    ----------
    state_file : str
        Path to the persistent state JSON file. Passed through to ServoStateStore.
        Default: "servo_state.json" in the current working directory.

    Attributes
    ----------
    store : ServoStateStore
        The persistent state store shared by all servos attached to this controller.
    """

    def __init__(self, state_file: str = "servo_state.json"):
        self.store = ServoStateStore(state_file)
        self._h:   Optional[int]           = None  # lgpio chip handle
        self._ser: Optional[serial.Serial] = None  # serial port handle

        # ── Global speed scale ────────────────────────────────────────────────
        # A safety gate that scales every servo's effective speed before it
        # reaches hardware. Range: 0.0 (all servos frozen) to 1.0 (full speed).
        #
        # Default is 0.0 — nothing moves until you explicitly enable it.
        # Set via ServoController.set_speed_scale(), or from the web dashboard.
        #
        # How it works:
        #   actual_speed = int(requested_speed * speed_scale)
        #   If actual_speed < SPEED_MIN (1), the command is suppressed entirely.
        #
        # This affects JointServo.move_to() and WheelServo.spin() — it does NOT
        # affect stop(), disable_torque(), or any read operations.
        self.speed_scale: float = 0.0

    # ── context manager ───────────────────────────────────────────────────────

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def open(self):
        """Open GPIO and serial port. Called automatically by __enter__."""
        if not ON_PI:
            logger.warning("CANCELLED: Not running on Raspberry Pi/Linux GPIO environment")
            return

        self._h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self._h, DIRECTION_PIN)
        self._ser = serial.Serial(
            port=SERIAL_PORT,
            baudrate=BAUDRATE,
            timeout=0.1,
        )
        logger.info("ServoController opened — port %s @ %d baud", SERIAL_PORT, BAUDRATE)

    def close(self):
        """Close serial port and GPIO. Called automatically by __exit__."""
        if self._ser and self._ser.is_open:
            self._ser.close()
        if self._h is not None:
            lgpio.gpiochip_close(self._h)
        logger.info("ServoController closed")

    # ── speed scale ───────────────────────────────────────────────────────────

    def set_speed_scale(self, scale: float) -> None:
        """
        Set the global speed scale for all servos on this bus.

        Parameters
        ----------
        scale : float
            0.0 = all servos frozen (no movement commands sent to hardware).
            1.0 = full speed (servo's own max_speed / speed setting applies).
            Values outside [0.0, 1.0] are clamped.

        This is the master safety switch. It is set to 0.0 on startup so
        nothing can move until you explicitly allow it — either from code
        or from the web dashboard slider.

        Called automatically by the web dashboard when the slider moves.
        Safe to call from any thread.
        """
        self.speed_scale = max(0.0, min(1.0, scale))
        logger.info("Global speed scale set to %.0f%%", self.speed_scale * 100)

    def scale_speed(self, requested_speed: int) -> int:
        """
        Apply speed_scale to a requested speed value.

        Returns the scaled integer speed, or 0 if the scale is so low
        that the result would fall below SPEED_MIN.
        Used internally by JointServo and WheelServo before every move command.

        Parameters
        ----------
        requested_speed : int   The unscaled speed (1–1023).

        Returns
        -------
        int   Scaled speed ready to write to hardware, or 0 meaning "don't move".
        """
        scaled = int(requested_speed * self.speed_scale)
        return scaled if scaled >= SPEED_MIN else 0

    # ── packet helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _checksum(data: list[int]) -> int:
        """
        Dynamixel checksum: bitwise NOT of the lower byte of the sum of all
        data bytes between the header and the checksum itself.
        """
        return (~sum(data)) & 0xFF

    def _send(self, packet: bytes):
        """
        Transmit a raw packet on the RS-485 bus.

        Sets the direction pin HIGH (TX mode) before writing and LOW (RX mode)
        immediately after. The 1 ms sleep gives the servo time to respond
        before we switch back to listening.
        """
        lgpio.gpio_write(self._h, DIRECTION_PIN, 1)
        self._ser.write(packet)
        self._ser.flush()
        time.sleep(0.001)
        lgpio.gpio_write(self._h, DIRECTION_PIN, 0)

    # ── public register access ────────────────────────────────────────────────

    def write_byte(self, servo_id: int, address: int, value: int):
        """
        Write a single byte to a register.

        Parameters
        ----------
        servo_id : int   Target servo's Dynamixel ID.
        address  : int   Register address (use REG_* constants above).
        value    : int   Byte value to write (0–255).
        """
        data = [servo_id, 4, INST_WRITE, address, value & 0xFF]
        self._send(bytes([0xFF, 0xFF] + data + [self._checksum(data)]))

    def write_word(self, servo_id: int, address: int, value: int):
        """
        Write a 16-bit word (two bytes, little-endian) to a register.

        Used for position (0–1023) and speed (0–2047) values which exceed
        one byte. Splits value into low byte and high byte automatically.

        Parameters
        ----------
        servo_id : int   Target servo's Dynamixel ID.
        address  : int   Register address (use REG_* constants above).
        value    : int   Word value to write (0–2047 depending on register).
        """
        low  = value & 0xFF
        high = (value >> 8) & 0xFF
        data = [servo_id, 5, INST_WRITE, address, low, high]
        self._send(bytes([0xFF, 0xFF] + data + [self._checksum(data)]))

    def read_word(self, servo_id: int, address: int) -> Optional[int]:
        """
        Read a 16-bit word from a register.

        Parameters
        ----------
        servo_id : int   Target servo's Dynamixel ID.
        address  : int   Register address (use REG_* constants above).

        Returns
        -------
        int or None
            The 16-bit value, or None if the servo did not respond or the
            response checksum was invalid (e.g. servo not powered / wrong ID).
        """
        data = [servo_id, 4, INST_READ, address, 2]
        self._send(bytes([0xFF, 0xFF] + data + [self._checksum(data)]))

        raw = self._ser.read(8)  # response: 0xFF 0xFF ID LEN ERR LOW HIGH CHK
        if len(raw) < 8 or raw[0] != 0xFF or raw[1] != 0xFF:
            logger.warning(
                "read_word: no/bad response from servo %d register %d", servo_id, address
            )
            return None
        return raw[5] | (raw[6] << 8)


# ─────────────────────────────────────────────────────────────────────────────
# BASE SERVO  (internal — not used directly)
# ─────────────────────────────────────────────────────────────────────────────

class _BaseServo:
    """
    Internal base class with functionality shared by JointServo and WheelServo.

    Not intended to be instantiated on its own. Use JointServo or WheelServo.

    Attributes
    ----------
    ctrl : ServoController
        The shared bus controller this servo communicates through.
    id : int
        The Dynamixel ID of this servo (1–253, set in hardware with Dynamixel Wizard).
    label : str
        Optional human-readable name stored in persistent state (e.g. "left_wheel").
    _torque_on : bool
        In-memory flag tracking whether torque is currently enabled.
        Kept in sync with the hardware and saved to disk on every change.
    """

    DEFAULT_SPEED = 150  # Conservative default speed — safe for most setups

    def __init__(self, controller: ServoController, servo_id: int, label: str = ""):
        self.ctrl  = controller
        self.id    = servo_id
        self.label = label
        self._torque_on = False

    # ── torque ────────────────────────────────────────────────────────────────

    def enable_torque(self):
        """
        Power the motor. Required before any movement command will work.
        State is saved to disk so we know on restart whether it was on or off.
        """
        self.ctrl.write_byte(self.id, REG_TORQUE_ENABLE, 1)
        self._torque_on = True
        self.ctrl.store.update(self.id, torque_on=True)
        logger.debug("[servo %d] torque ON", self.id)

    def disable_torque(self):
        """
        Cut motor power. Shaft can now spin freely by hand.
        Always call this (via safe_shutdown) before powering off the Pi.
        """
        self.ctrl.write_byte(self.id, REG_TORQUE_ENABLE, 0)
        self._torque_on = False
        self.ctrl.store.update(self.id, torque_on=False)
        logger.debug("[servo %d] torque OFF", self.id)

    # ── status queries ────────────────────────────────────────────────────────

    def get_position(self) -> Optional[int]:
        """
        Read the current shaft position directly from the servo (0–1023).
        Returns None if the servo does not respond.
        """
        return self.ctrl.read_word(self.id, REG_PRESENT_POSITION)

    def get_load(self) -> Optional[int]:
        """
        Read the current load on the motor (0–2047).
        0–1023 = CCW load; 1024–2047 = CW load.
        Useful for detecting stalls or collisions.
        Returns None if the servo does not respond.
        """
        return self.ctrl.read_word(self.id, REG_PRESENT_LOAD)

    def is_moving(self) -> Optional[bool]:
        """
        Return True if the servo is still travelling to its goal.
        Returns None if the servo does not respond.
        """
        val = self.ctrl.read_word(self.id, REG_MOVING)
        return bool(val) if val is not None else None

    def wait_until_stopped(self, timeout: float = 5.0, poll: float = 0.05):
        """
        Block until the servo reports it has stopped moving.

        Parameters
        ----------
        timeout : float   Maximum seconds to wait before giving up.
        poll    : float   How often (seconds) to ask the servo if it's still moving.
        """
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_moving():
                return
            time.sleep(poll)
        logger.warning("[servo %d] wait_until_stopped timed out after %.1fs", self.id, timeout)

    def stop(self):
        """Stop the servo. Each subclass implements this differently."""
        raise NotImplementedError


# ─────────────────────────────────────────────────────────────────────────────
# JOINT SERVO  (rotates between a fixed min and max position)
# ─────────────────────────────────────────────────────────────────────────────

class JointServo(_BaseServo):
    """
    A servo that sweeps between a defined minimum and maximum position.
    Use this for arms, grippers, tilting heads, etc.

    SAFETY: Every position command is clamped to [min_pos, max_pos].
    If outside limits, it is silently clamped and a warning is logged.
    This prevents mechanical damage from out-of-range commands.

    ⚠️  IMPORTANT: Before deploying, set min_pos and max_pos to a NARROW
    range around your expected movement. Widen only after confirming the
    hardware handles it safely. Starting with the full 0–1023 range is
    asking for broken linkages.

    PERSISTENT STATE:
    min_pos, max_pos, speed, and last_pos are saved to disk.
    If a servo with this ID already exists in the state file AND no new
    limits are provided, the saved limits are restored automatically.

    Parameters
    ----------
    controller : ServoController
        The shared bus controller.
    servo_id   : int
        Dynamixel ID (1–253) — must match the ID burned into the servo.
    min_pos    : int or None
        Minimum allowed position (0–1023). If None, loads from saved state.
        If no saved state exists, defaults to POSITION_MIN_ABSOLUTE (0).
    max_pos    : int or None
        Maximum allowed position (0–1023). If None, loads from saved state.
        If no saved state exists, defaults to POSITION_MAX_ABSOLUTE (1023).
    speed      : int
        Default movement speed (1–1023). Loaded from state if not provided.
    label      : str
        Optional name saved to disk (e.g. "elbow"). Purely informational.

    Attributes
    ----------
    min_pos  : int   Lower bound of safe movement range.
    max_pos  : int   Upper bound of safe movement range.
    _speed   : int   Current speed setting.
    """

    def __init__(
        self,
        controller: ServoController,
        servo_id:   int,
        min_pos:    Optional[int] = None,
        max_pos:    Optional[int] = None,
        speed:      int           = _BaseServo.DEFAULT_SPEED,
        label:      str           = "",
    ):
        super().__init__(controller, servo_id, label)

        # Load saved state (if any) and apply as defaults
        saved = controller.store.get(servo_id) or {}

        self.min_pos = min_pos if min_pos is not None else saved.get("min_pos", POSITION_MIN_ABSOLUTE)
        self.max_pos = max_pos if max_pos is not None else saved.get("max_pos", POSITION_MAX_ABSOLUTE)
        self._speed  = saved.get("speed", speed)

        if not (POSITION_MIN_ABSOLUTE <= self.min_pos < self.max_pos <= POSITION_MAX_ABSOLUTE):
            raise DynamixelError(
                f"[servo {servo_id}] Invalid range [{self.min_pos}, {self.max_pos}]. "
                f"Must be within {POSITION_MIN_ABSOLUTE}–{POSITION_MAX_ABSOLUTE} with min < max."
            )

        # Persist this servo's configuration
        controller.store.update(
            servo_id,
            mode    = "joint",
            min_pos = self.min_pos,
            max_pos = self.max_pos,
            speed   = self._speed,
            label   = label or saved.get("label", ""),
        )

        self.enable_torque()
        self.ctrl.write_word(self.id, REG_MOVING_SPEED, self._speed)

        logger.info(
            "[servo %d '%s'] JointServo ready | range [%d–%d] speed %d",
            self.id, self.label or "unlabelled", self.min_pos, self.max_pos, self._speed,
        )

    # ── speed ─────────────────────────────────────────────────────────────────

    def set_speed(self, speed: int):
        """
        Change movement speed. Persisted to disk.

        Parameters
        ----------
        speed : int
            1–1023. Higher = faster. 0 means full uncontrolled speed — not allowed here.
            Clamped to [1, 1023].
        """
        self._speed = max(SPEED_MIN, min(SPEED_MAX, speed))
        self.ctrl.write_word(self.id, REG_MOVING_SPEED, self._speed)
        self.ctrl.store.update(self.id, speed=self._speed)

    # ── movement ──────────────────────────────────────────────────────────────

    def _clamp(self, pos: int) -> int:
        """Clamp pos to [min_pos, max_pos], logging a warning if clamping occurred."""
        clamped = max(self.min_pos, min(self.max_pos, pos))
        if clamped != pos:
            logger.warning(
                "[servo %d] Requested position %d is outside limits [%d–%d], clamped to %d",
                self.id, pos, self.min_pos, self.max_pos, clamped,
            )
        return clamped

    def move_to(self, position: int, wait: bool = False):
        """
        Move to an absolute position. Clamped to [min_pos, max_pos].

        If speed_scale on the controller is 0.0, the command is suppressed and
        the servo stays where it is. Non-zero scales are applied to _speed before
        writing to the hardware.

        Parameters
        ----------
        position : int    Target position (0–1023).
        wait     : bool   If True, blocks until the servo has stopped moving.
        """
        scaled_speed = self.ctrl.scale_speed(self._speed)
        if scaled_speed == 0:
            logger.debug("[servo %d] move_to suppressed — speed_scale is 0", self.id)
            return

        pos = self._clamp(position)
        # Apply scaled speed before moving so the servo actually uses it
        self.ctrl.write_word(self.id, REG_MOVING_SPEED, scaled_speed)
        self.ctrl.write_word(self.id, REG_GOAL_POSITION, pos)
        self.ctrl.store.update(self.id, last_pos=pos)
        logger.debug("[servo %d] move_to %d (speed %d)", self.id, pos, scaled_speed)
        if wait:
            self.wait_until_stopped()

    def move_to_percent(self, percent: float, wait: bool = False):
        """
        Move to a position expressed as a fraction of the allowed range.

        Parameters
        ----------
        percent : float
            0.0 = min_pos (full CCW limit)
            0.5 = center of range
            1.0 = max_pos (full CW limit)
            Values outside 0.0–1.0 are clamped.
        wait    : bool   If True, blocks until the servo has stopped.
        """
        percent = max(0.0, min(1.0, percent))
        pos = int(self.min_pos + percent * (self.max_pos - self.min_pos))
        self.move_to(pos, wait=wait)

    def move_to_center(self, wait: bool = False):
        """Move to the midpoint of [min_pos, max_pos]."""
        self.move_to_percent(0.5, wait=wait)

    def stop(self):
        """
        Hold position by commanding the servo to stay where it currently is.
        Falls back to center if the position cannot be read.
        """
        pos = self.get_position()
        if pos is not None:
            self.move_to(pos)
        else:
            logger.warning("[servo %d] Could not read position, moving to center", self.id)
            self.move_to_center()

    # ── limit adjustment ──────────────────────────────────────────────────────

    def set_limits(self, min_pos: int, max_pos: int):
        """
        Update the safe range at runtime and persist to disk.

        Use this if you physically reconfigure the hardware and need
        different limits without restarting.

        Parameters
        ----------
        min_pos : int   New lower limit (0–1023).
        max_pos : int   New upper limit (0–1023, must be > min_pos).
        """
        if not (POSITION_MIN_ABSOLUTE <= min_pos < max_pos <= POSITION_MAX_ABSOLUTE):
            raise DynamixelError(f"Invalid limits [{min_pos}, {max_pos}]")
        self.min_pos = min_pos
        self.max_pos = max_pos
        self.ctrl.store.update(self.id, min_pos=min_pos, max_pos=max_pos)
        logger.info("[servo %d] Limits updated to [%d–%d]", self.id, min_pos, max_pos)

    # ── one-time EEPROM setup ─────────────────────────────────────────────────

    def set_joint_mode_eeprom(self):
        """
        Write CW/CCW angle limits to the servo's EEPROM so it boots in joint mode.

        ⚠️  EEPROM has a limited write cycle lifetime (~100,000 writes).
        Call this ONCE during hardware setup, not in any loop or regular code.
        After this, the servo will enforce these limits in hardware on every boot.
        """
        self.ctrl.write_word(self.id, REG_CW_ANGLE_LIMIT,  self.min_pos)
        self.ctrl.write_word(self.id, REG_CCW_ANGLE_LIMIT, self.max_pos)
        logger.info(
            "[servo %d] EEPROM written: joint mode [%d–%d]",
            self.id, self.min_pos, self.max_pos,
        )


# ─────────────────────────────────────────────────────────────────────────────
# WHEEL SERVO  (continuous 360° rotation)
# ─────────────────────────────────────────────────────────────────────────────

class WheelServo(_BaseServo):
    """
    A servo that spins continuously in either direction (wheel / drive mode).
    Use this for drive wheels, conveyor rollers, turntables, etc.

    In wheel mode, the goal position register is ignored. The speed register
    encodes both speed and direction:
      - Bit 10 = 0 → counter-clockwise  (values 0–1023)
      - Bit 10 = 1 → clockwise          (values 1024–2047, where 1024 = stop)

    ⚠️  WHEEL MODE MUST BE CONFIGURED ON THE SERVO BEFORE USE.
    Either use Dynamixel Wizard to set both angle limits to 0, OR call
    set_wheel_mode_eeprom() once during initial hardware setup.
    Without this, the servo will ignore speed commands and behave like a joint.

    PERSISTENT STATE:
    max_speed and last running state are saved to disk.

    Parameters
    ----------
    controller : ServoController
        The shared bus controller.
    servo_id   : int
        Dynamixel ID (1–253).
    max_speed  : int
        Hard cap on speed (1–1023). Commands above this are clamped.
        Start low (e.g. 200–300) and increase carefully.
        Loaded from saved state if not provided explicitly.
    label      : str
        Optional human-readable name (e.g. "left_wheel").

    Attributes
    ----------
    max_speed : int    Speed ceiling. Any spin() call is clamped to this.
    _running  : bool   Whether the wheel is currently spinning (in-memory only).
    """

    def __init__(
        self,
        controller: ServoController,
        servo_id:   int,
        max_speed:  Optional[int] = None,
        label:      str           = "",
    ):
        super().__init__(controller, servo_id, label)

        saved = controller.store.get(servo_id) or {}

        self.max_speed = max_speed if max_speed is not None else saved.get("max_speed", 300)
        self._running  = False

        if not (SPEED_MIN <= self.max_speed <= SPEED_MAX):
            raise DynamixelError(
                f"[servo {servo_id}] max_speed {self.max_speed} out of range [{SPEED_MIN}–{SPEED_MAX}]"
            )

        controller.store.update(
            servo_id,
            mode      = "wheel",
            max_speed = self.max_speed,
            label     = label or saved.get("label", ""),
        )

        self.enable_torque()
        logger.info(
            "[servo %d '%s'] WheelServo ready | max_speed %d",
            self.id, self.label or "unlabelled", self.max_speed,
        )

    # ── movement ──────────────────────────────────────────────────────────────

    def spin(self, speed: int, clockwise: bool = True):
        """
        Start spinning at the given speed and direction.

        If speed_scale on the controller is 0.0, the command is suppressed and
        the wheel stays stopped.

        Parameters
        ----------
        speed     : int
            Desired speed (1–max_speed). Clamped if above max_speed.
        clockwise : bool
            True = clockwise, False = counter-clockwise.
            "Clockwise" is from the perspective of looking at the output shaft.
        """
        speed = max(SPEED_MIN, min(self.max_speed, speed))
        scaled = self.ctrl.scale_speed(speed)
        if scaled == 0:
            logger.debug("[servo %d] spin suppressed — speed_scale is 0", self.id)
            return
        # Bit 10 set → CW direction in Dynamixel wheel mode encoding
        raw = scaled | 0x400 if clockwise else scaled
        self.ctrl.write_word(self.id, REG_MOVING_SPEED, raw)
        self._running = True
        self.ctrl.store.update(self.id, running=True, last_speed=scaled, last_direction="CW" if clockwise else "CCW")
        logger.debug("[servo %d] spinning %s speed %d (scaled from %d)", self.id, "CW" if clockwise else "CCW", scaled, speed)

    def stop(self):
        """
        Stop the wheel immediately by setting speed to 0.
        The servo holds its position under load (torque still on).
        """
        self.ctrl.write_word(self.id, REG_MOVING_SPEED, 0)
        self._running = False
        self.ctrl.store.update(self.id, running=False)
        logger.debug("[servo %d] stopped", self.id)

    def spin_for(self, seconds: float, speed: int, clockwise: bool = True):
        """
        Spin for a fixed duration, then stop automatically.

        Parameters
        ----------
        seconds   : float   How long to spin.
        speed     : int     Speed (1–max_speed).
        clockwise : bool    Direction.
        """
        self.spin(speed, clockwise)
        time.sleep(seconds)
        self.stop()

    def set_max_speed(self, max_speed: int):
        """
        Update the speed ceiling at runtime and persist to disk.

        Parameters
        ----------
        max_speed : int   New ceiling (1–1023).
        """
        if not (SPEED_MIN <= max_speed <= SPEED_MAX):
            raise DynamixelError(f"max_speed {max_speed} out of range [{SPEED_MIN}–{SPEED_MAX}]")
        self.max_speed = max_speed
        self.ctrl.store.update(self.id, max_speed=max_speed)
        logger.info("[servo %d] max_speed updated to %d", self.id, max_speed)

    # ── one-time EEPROM setup ─────────────────────────────────────────────────

    def set_wheel_mode_eeprom(self):
        """
        Write both angle limits to 0 in EEPROM, permanently enabling wheel mode.

        ⚠️  EEPROM has a limited write cycle lifetime (~100,000 writes).
        Call this ONCE during initial hardware setup, never in a loop.
        The servo will boot in wheel mode from this point forward.
        """
        self.ctrl.write_word(self.id, REG_CW_ANGLE_LIMIT,  0)
        self.ctrl.write_word(self.id, REG_CCW_ANGLE_LIMIT, 0)
        logger.info("[servo %d] EEPROM written: wheel mode enabled", self.id)


# ─────────────────────────────────────────────────────────────────────────────
# SAFE SHUTDOWN  (always call this in a finally block)
# ─────────────────────────────────────────────────────────────────────────────

def safe_shutdown(*servos: _BaseServo, delay: float = 0.2):
    """
    Stop and disable torque on every servo passed in.

    Designed to be called in a finally block to guarantee clean shutdown
    even if the code crashes or CTRL+C is pressed.

    What it does
    ------------
    1. Calls stop() on every servo (holds position or cuts wheel speed).
    2. Waits `delay` seconds for the servos to settle.
    3. Calls disable_torque() on every servo (shafts become free-spinning).
    4. State is saved to disk for each servo.

    Errors during shutdown are logged but do not prevent other servos
    from shutting down.

    Parameters
    ----------
    *servos : _BaseServo   Any number of JointServo / WheelServo instances.
    delay   : float        Seconds to wait between stop and torque-off. Default 0.2.

    Example
    -------
    try:
        arm.move_to(600)
        wheel.spin(200)
    finally:
        safe_shutdown(arm, wheel)
    """
    for servo in servos:
        try:
            servo.stop()
        except Exception as e:
            logger.error("[servo %d] stop() failed during shutdown: %s", servo.id, e)

    time.sleep(delay)

    for servo in servos:
        try:
            servo.disable_torque()
        except Exception as e:
            logger.error("[servo %d] disable_torque() failed during shutdown: %s", servo.id, e)

    logger.info("safe_shutdown complete for %d servo(s)", len(servos))