"""
AS5600 Angle Sensor Manager
Uses:
    - TCA9548A multiplexer
    - Direct AS5600 register reads
    - No AS5600 library required

Dependencies:
    pip install adafruit-blinka
    pip install adafruit-circuitpython-tca9548a
    pip install smbus2
"""

import board
import adafruit_tca9548a
from smbus2 import SMBus

# AS5600 I2C address
AS5600_ADDR = 0x36

# AS5600 angle registers
RAW_ANGLE_MSB = 0x0C
RAW_ANGLE_LSB = 0x0D


class AngleSensorManager:

    def __init__(self, multiplexer_address=0x70):
        self.i2c = board.I2C()

        self.tca = adafruit_tca9548a.TCA9548A(
            self.i2c,
            address=multiplexer_address
        )

        # Raspberry Pi I2C bus
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
        """
        Select multiplexer channel.
        """

        if channel < 0 or channel > 7:
            raise ValueError(
                "Channel must be between 0 and 7"
            )

        # TCA9548A channel select register
        self.bus.write_byte(
            0x70,
            1 << channel
        )

    def _read_raw_angle(self, channel):
        """
        Read raw 12-bit AS5600 angle.

        Returns:
            int (0-4095)
        """

        self._select_channel(channel)

        high = self.bus.read_byte_data(
            AS5600_ADDR,
            RAW_ANGLE_MSB
        )

        low = self.bus.read_byte_data(
            AS5600_ADDR,
            RAW_ANGLE_LSB
        )

        raw = ((high << 8) | low) & 0x0FFF

        return raw

    def _raw_to_degrees(self, raw):
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
# Example
# -------------------------------------------------

if __name__ == "__main__":

    sensors = AngleSensorManager()

    sensors.scan_all_channels()

    while True:

        data = sensors.read_both_sensors()

        print(
            f"S1: {data['sensor1']['degrees']:.2f}° "
            f"({data['sensor1']['raw']})"
        )

        print(
            f"S2: {data['sensor2']['degrees']:.2f}° "
            f"({data['sensor2']['raw']})"
        )

        print("-" * 40)