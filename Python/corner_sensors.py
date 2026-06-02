from smbus2 import SMBus
import time

TCA_ADDR = 0x70
AS5600_ADDR = 0x36

RAW_MSB = 0x0C
RAW_LSB = 0x0D

class CornerSensorManager:

    def __init__(self, bus_num=13, channels=(0, 1, 2)):

        self.bus = SMBus(bus_num)
        self.channels = channels

        print(f"[INFO] Using /dev/i2c-{bus_num}")
        print(f"[INFO] Channels mapped: {channels}")

# -------------------------
# Select TCA channel
# -------------------------
def _select(self, ch):

    if ch < 0 or ch > 7:
        raise ValueError("TCA channel must be 0–7")

    self.bus.write_byte(TCA_ADDR, 1 << ch)

# -------------------------
# Read one sensor
# -------------------------
    def _read_raw(self, ch):

        self._select(ch)

        high = self.bus.read_byte_data(AS5600_ADDR, RAW_MSB)
        low  = self.bus.read_byte_data(AS5600_ADDR, RAW_LSB)

        return ((high << 8) | low) & 0x0FFF

    @staticmethod
    def _deg(raw):
        return raw * 360.0 / 4096.0

# -------------------------
# Public API
# -------------------------
    def read_all(self):

        result = {}

        for i, ch in enumerate(self.channels):

            raw = self._read_raw(ch)

            result[f"sensor_{i+1}"] = {
               "channel": ch,
               "raw": raw,
               "deg": self._deg(raw)
            }

        return result

    def monitor(self, delay=0.2):

        try:
            while True:

                data = self.read_all()

                line = " | ".join(
                     f"S{i+1}: {d['deg']:.2f}°"
                    for i, d in enumerate(data.values())
                )

                print(line)

                time.sleep(delay)

        except KeyboardInterrupt:
            print("\nStopped")
