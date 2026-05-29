# Basic TCA9548A multiplexer setup for Raspberry Pi
# Designed to be easy to expand later.
#
# Install first:
# pip3 install adafruit-blinka
# pip3 install adafruit-circuitpython-tca9548a

import board
import adafruit_tca9548a


class MultiplexerManager:
    def __init__(self, address=0x70):
        """
        Create the I2C bus and multiplexer object.
        Default TCA9548A address = 0x70
        """

        # Main Raspberry Pi I2C bus
        self.i2c = board.I2C()

        # Multiplexer object
        self.tca = adafruit_tca9548a.TCA9548A(
            self.i2c,
            address=address
        )

        print(f"[INFO] Multiplexer initialized at address {hex(address)}")

    def get_channel(self, channel: int):
        """
        Get a multiplexer channel.

        Example:
            channel0 = mux.get_channel(0)
        """

        if channel < 0 or channel > 7:
            raise ValueError("Channel must be between 0 and 7")

        return self.tca[channel]

    def scan_channel(self, channel: int):
        """
        Scan a single channel for connected I2C devices.
        """

        ch = self.get_channel(channel)

        if ch.try_lock():
            try:
                addresses = ch.scan()

                # Remove multiplexer address from results
                addresses = [
                    addr for addr in addresses
                    if addr != 0x70
                ]

                return addresses

            finally:
                ch.unlock()

        return []

    def scan_all_channels(self):
        """
        Scan all 8 channels and print results.
        """

        print("\n--- TCA9548A CHANNEL SCAN ---")

        for channel in range(8):
            devices = self.scan_channel(channel)

            if devices:
                hex_devices = [hex(d) for d in devices]
                print(f"Channel {channel}: {hex_devices}")
            else:
                print(f"Channel {channel}: No devices found")

        print("-----------------------------\n")


# Example usage
if __name__ == "__main__":

    mux = MultiplexerManager()

    # Scan all channels
    mux.scan_all_channels()

    # Example:
    # sensor_channel = mux.get_channel(0)
    #
    # Later:
    # sensor = YourSensorLibrary(sensor_channel)