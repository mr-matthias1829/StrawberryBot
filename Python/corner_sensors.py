from smbus2 import SMBus
import time

TCA_ADDR   = 0x70
AS5600_ADDR = 0x36

RAW_MSB = 0x0C
RAW_LSB = 0x0D


class CornerSensorManager:

    def __init__(self, bus_num=1):
        self.bus = SMBus(bus_num)
        print(f"[INFO] Using /dev/i2c-{bus_num}")
        self.channels = self._discover_channels()
        print(f"[INFO] Found sensors on channels: {self.channels}")

    def _select(self, ch):
        if ch < 0 or ch > 7:
            raise ValueError("TCA channel must be 0-7")
        self.bus.write_byte(TCA_ADDR, 1 << ch)

    def _discover_channels(self):
        found = []
        for ch in range(8):
            try:
                self._select(ch)
                self.bus.read_byte_data(AS5600_ADDR, RAW_MSB)
                print(f"[INFO] AS5600 found on channel {ch}")
                found.append(ch)
            except OSError:
                pass
        return tuple(found)

    def _read_raw(self, ch):
        self._select(ch)
        high = self.bus.read_byte_data(AS5600_ADDR, RAW_MSB)
        low  = self.bus.read_byte_data(AS5600_ADDR, RAW_LSB)
        return ((high << 8) | low) & 0x0FFF

    @staticmethod
    def _deg(raw):
        return raw * 360.0 / 4096.0

    def read_all(self):
        result = {}
        for i, ch in enumerate(self.channels):
            try:
                raw = self._read_raw(ch)
                result[f"sensor_{i+1}"] = {
                    "channel": ch,
                    "raw":     raw,
                    "deg":     self._deg(raw),
                }
            except OSError as e:
                print(f"[WARN] Channel {ch} disappeared: {e}")
        return result

    def monitor(self, delay=0.2):
        try:
            while True:
                data = self.read_all()
                line = " | ".join(
                    f"S{i+1}(CH{d['channel']}): {d['deg']:.2f}°"
                    for i, d in enumerate(data.values())
                )
                print(line)
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\nStopped")


if __name__ == "__main__":
    sensors = CornerSensorManager(bus_num=1)
    sensors.monitor()