"""
AS5600 Angle Sensor Manager

Examples:
    python angles.py scan
    python angles.py sensor1
    python angles.py sensor2
    python angles.py both
    python angles.py monitor
    python angles.py monitor --interval 0.1
"""

import argparse
import time

import board
import adafruit_tca9548a
from smbus2 import SMBus


AS5600_ADDR = 0x36

RAW_ANGLE_MSB = 0x0C
RAW_ANGLE_LSB = 0x0D


class AngleSensorManager:

    def __init__(self, multiplexer_address=0x70):
        self.i2c = board.I2C()

        self.tca = adafruit_tca9548a.TCA9548A(
            self.i2c,
            address=multiplexer_address
        )

        self.bus = SMBus(1)

        print(
            f"[INFO] Multiplexer initialized at "
            f"{hex(multiplexer_address)}"
        )

    # -------------------------------------------------
    # Debug utility
    # -------------------------------------------------

    def scan_all_channels(self):
        print("\n--- TCA9548A CHANNEL SCAN ---")

        for channel in range(8):

            ch = self.tca[channel]

            if ch.try_lock():
                try:
                    devices = [
                        hex(addr)
                        for addr in ch.scan()
                        if addr != 0x70
                    ]

                    print(
                        f"Channel {channel}: "
                        f"{devices}"
                    )

                finally:
                    ch.unlock()

        print("-----------------------------\n")

    # -------------------------------------------------
    # Internal helpers
    # -------------------------------------------------

    def _select_channel(self, channel):

        if channel < 0 or channel > 7:
            raise ValueError(
                "Channel must be between 0 and 7"
            )

        self.bus.write_byte(
            0x70,
            1 << channel
        )

    def _read_raw_angle(self, channel):

        self._select_channel(channel)

        high = self.bus.read_byte_data(
            AS5600_ADDR,
            RAW_ANGLE_MSB
        )

        low = self.bus.read_byte_data(
            AS5600_ADDR,
            RAW_ANGLE_LSB
        )

        return ((high << 8) | low) & 0x0FFF

    @staticmethod
    def _raw_to_degrees(raw):
        return raw * 360.0 / 4096.0

    # -------------------------------------------------
    # Public API
    # -------------------------------------------------

    def read_sensor_1(self):
        raw = self._read_raw_angle(0)

        return {
            "raw": raw,
            "degrees": self._raw_to_degrees(raw)
        }

    def read_sensor_2(self):
        raw = self._read_raw_angle(1)

        return {
            "raw": raw,
            "degrees": self._raw_to_degrees(raw)
        }

    def read_both_sensors(self):
        return {
            "sensor1": self.read_sensor_1(),
            "sensor2": self.read_sensor_2()
        }


# -------------------------------------------------
# CLI
# -------------------------------------------------

def print_sensor(name, data):
    print(
        f"{name}: "
        f"{data['degrees']:.2f}° "
        f"({data['raw']})"
    )


def cmd_scan(sensor_mgr):
    sensor_mgr.scan_all_channels()


def cmd_sensor1(sensor_mgr):
    print_sensor(
        "S1",
        sensor_mgr.read_sensor_1()
    )


def cmd_sensor2(sensor_mgr):
    print_sensor(
        "S2",
        sensor_mgr.read_sensor_2()
    )


def cmd_both(sensor_mgr):
    data = sensor_mgr.read_both_sensors()

    print_sensor(
        "S1",
        data["sensor1"]
    )

    print_sensor(
        "S2",
        data["sensor2"]
    )


def cmd_monitor(sensor_mgr, interval):

    try:
        while True:

            data = sensor_mgr.read_both_sensors()

            print(
                f"S1: {data['sensor1']['degrees']:.2f}° "
                f"({data['sensor1']['raw']}) | "
                f"S2: {data['sensor2']['degrees']:.2f}° "
                f"({data['sensor2']['raw']})"
            )

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\nStopped.")


def main():

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(
        dest="command"
    )

    subparsers.add_parser("scan")
    subparsers.add_parser("sensor1")
    subparsers.add_parser("sensor2")
    subparsers.add_parser("both")

    monitor = subparsers.add_parser(
        "monitor"
    )

    monitor.add_argument(
        "--interval",
        type=float,
        default=0.25
    )

    args = parser.parse_args()

    sensor_mgr = AngleSensorManager()

    if args.command == "scan":
        cmd_scan(sensor_mgr)

    elif args.command == "sensor1":
        cmd_sensor1(sensor_mgr)

    elif args.command == "sensor2":
        cmd_sensor2(sensor_mgr)

    elif args.command == "both":
        cmd_both(sensor_mgr)

    elif args.command == "monitor":
        cmd_monitor(
            sensor_mgr,
            args.interval
        )

    else:
        # No command supplied
        cmd_monitor(
            sensor_mgr,
            0.25
        )


if __name__ == "__main__":
    main()