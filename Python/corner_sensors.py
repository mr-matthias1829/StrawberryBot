from smbus2 import SMBus
import time

TCA_ADDR    = 0x70 # multiplexer
AS5600_ADDR = 0x36 # sensor
RAW_MSB     = 0x0C
RAW_LSB     = 0x0D

# threshold (degrees) to detect a wrap: if the jump between two readings is
# larger than this we treat it as a 0°/360° crossing.
WRAP_THRESHOLD = 180.0

# how many times to retry an I2C read before giving up and falling back to
# the last known-good value. Keeps gaps between successful samples small so
# wrap detection (_update_laps) doesn't get fed a stale _prev_deg.
READ_RETRIES = 3
RETRY_DELAY  = 0.002  # seconds between retries


class CornerSensorManager:
    def __init__(self, bus_num=1):
        self.bus = SMBus(bus_num)
        print(f"[INFO] Using /dev/i2c-{bus_num}")
        self.channels = self._discover_channels()
        print(f"[INFO] Found sensors on channels: {self.channels}")

        # lap tracking — keyed by channel number
        self._laps:      dict[int, int]   = {ch: 0 for ch in self.channels}
        self._prev_deg:  dict[int, float] = {}   # last known angle per channel
        self._direction: dict[int, int]   = {ch: 0 for ch in self.channels}  # +1 / -1 / 0

        # last known-good reading per channel — returned (marked stale) if
        # the live read fails after all retries, so callers always get the
        # most reliable latest value instead of an exception/None.
        self._last_good: dict[int, dict] = {}


    def _select(self, ch: int) -> None:
        if ch < 0 or ch > 7:
            raise ValueError("TCA channel must be 0-7")
        self.bus.write_byte(TCA_ADDR, 1 << ch)

    def _discover_channels(self) -> tuple:
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

    def _read_raw(self, ch: int) -> int:
        self._select(ch)
        high = self.bus.read_byte_data(AS5600_ADDR, RAW_MSB)
        low  = self.bus.read_byte_data(AS5600_ADDR, RAW_LSB)
        return ((high << 8) | low) & 0x0FFF

    def _read_raw_retry(self, ch: int) -> int | None:
        """
        try to read the raw value up to READ_RETRIES times.
        returns the raw int on success, or None if every attempt failed.
        """
        last_err = None
        for attempt in range(READ_RETRIES):
            try:
                return self._read_raw(ch)
            except OSError as e:
                last_err = e
                if attempt < READ_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        print(f"[WARN] channel {ch} read failed after {READ_RETRIES} attempts: {last_err}")
        return None

    @staticmethod
    def _deg(raw: int) -> float: # converts raw to proper deg
        return raw * 360.0 / 4096.0

    def _update_laps(self, ch: int, new_deg: float) -> None:
        """detect a 0°/360° wrap and increment (or decrement) the lap counter."""
        if ch not in self._prev_deg:
            self._prev_deg[ch] = new_deg
            return

        prev = self._prev_deg[ch]
        delta = new_deg - prev

        if delta > WRAP_THRESHOLD: # wrapped backwards  (360 to 0)
            self._laps[ch] -= 1
            self._direction[ch] = -1
        elif delta < -WRAP_THRESHOLD: # wrapped forwards   (0 to 360)
            self._laps[ch] += 1
            self._direction[ch] = 1
        else:
            self._direction[ch] = 1 if delta > 0 else (-1 if delta < 0 else 0)

        self._prev_deg[ch] = new_deg



    def channel_has_sensor(self, ch: int) -> bool:
        """return True if a sensor was discovered on the given TCA channel."""
        return ch in self.channels

    def read_sensor(self, ch: int) -> dict:
        """
        read one sensor by TCA channel number.

        Returns a dict with keys:
            channel  – TCA channel
            raw      – 12-bit raw value (0-4095)
            deg      – angle in degrees (0.0-359.9…)
            laps     – signed lap count since start / last reset
            stale    – True if this is a repeat of the last known-good
                       reading because the live read failed even after
                       retries; False if it's a fresh sample.

        Raises ValueError if no sensor is present on that channel
        """
        if not self.channel_has_sensor(ch):
            raise ValueError(f"No sensor on channel {ch}")

        raw = self._read_raw_retry(ch)

        if raw is None:
            # all retries failed this cycle — fall back to last known-good
            # reading rather than returning nothing. crucially, _prev_deg /
            # _laps are NOT touched here, so the next successful read still
            # compares against the correct previous angle.
            if ch in self._last_good:
                stale = dict(self._last_good[ch])
                stale["stale"] = True
                return stale
            raise OSError(f"Channel {ch}: no reading available (sensor never responded)")

        deg = self._deg(raw)
        self._update_laps(ch, deg)
        result = {
            "channel": ch,
            "raw":     raw,
            "deg":     deg,
            "laps":    self._laps[ch],
            "stale":   False,
        }
        self._last_good[ch] = result
        return result

    def read_sensor_by_index(self, index: int) -> dict:
        """
        read sensor by discovery index (0-based).
        convenience wrapper around read_sensor() for code that uses
        positional numbering rather than channel numbers.
        """
        if index < 0 or index >= len(self.channels):
            raise IndexError(f"sensor index {index} out of range (found {len(self.channels)} sensors)")
        return self.read_sensor(self.channels[index])

    def get_laps(self, ch: int) -> int:
        """return the current lap count for a channel (positive = CW, negative = CCW)."""
        if not self.channel_has_sensor(ch):
            raise ValueError(f"No sensor on channel {ch}")
        return self._laps[ch]

    def reset_laps(self, ch: int | None = None) -> None:
        """
        reset lap counters.
        pass a channel number to reset just one sensor, or None to reset all.
        """
        targets = self.channels if ch is None else (ch,)
        for c in targets:
            self._laps[c] = 0

    def sensor_count(self) -> int:
        """return the number of discovered sensors."""
        return len(self.channels)

    def read_all(self) -> dict:
        """
        read every discovered sensor.

        returns a dict keyed by "sensor_1", "sensor_2", … each containing:
            channel, raw, deg, laps, stale
        """
        result = {}
        for i, ch in enumerate(self.channels):
            try:
                result[f"sensor_{i+1}"] = self.read_sensor(ch)
            except OSError as e:
                print(f"[WARN] Channel {ch} unavailable: {e}")
        return result

    def monitor(self, delay: float = 0.2) -> None:
        """continuously print all sensor readings until Ctrl-C."""
        try:
            while True:
                data = self.read_all()
                line = " | ".join(
                    f"S{i+1}(CH{d['channel']}): {d['deg']:>7.2f}°  laps={d['laps']:+d}"
                    + (" [stale]" if d.get("stale") else "")
                    for i, d in enumerate(data.values())
                )
                print(line)
                time.sleep(delay)
        except KeyboardInterrupt:
            print("\nstopped")

    @staticmethod
    def total_position(reading: dict) -> float:
        """give the full degrees amount by (laps * degrees), includes correction for negative numbers"""
        laps = reading["laps"]
        deg = reading["deg"]
        if laps >= 0:
            return laps * 360.0 + deg
        else:
            return laps * 360.0 - (360.0 - deg)



# standalone
# reads all discovered sensors and then monitors them
if __name__ == "__main__":
    sensors = CornerSensorManager(bus_num=1)
    sensors.monitor()