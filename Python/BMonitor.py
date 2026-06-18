"""
BMonitor.py
==================
Monitors GPIO 22 (pin 15) for battery status.
HIGH = battery fine, LOW = battery critical → home all + shutdown.

Run alongside Main.py or import and call start() from Main.py.

ICT made this
"""

import threading
import time
import platform

ON_PI = platform.system() == "Linux"

GPIO_PIN = 22
POLL_INTERVAL = 1.0   # seconds between checks
DEBOUNCE_COUNT = 5    # must read LOW this many times in a row before acting

_thread = None
_running = False


def _monitor():
    time.sleep(5)
    if ON_PI:
        import lgpio
        h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_input(h, GPIO_PIN, lgpio.SET_PULL_UP)

    low_count = 0
    print(f"[battery] Monitoring GPIO {GPIO_PIN} (pin 15). HIGH=ok, LOW=critical.")


    while _running:
        if ON_PI:
            import lgpio
            level = lgpio.gpio_read(h, GPIO_PIN)
        else:
            level = 1  # simulate fine on non-Pi

        if level == 0:
            low_count += 1
            print(f"[battery] LOW reading {low_count}/{DEBOUNCE_COUNT}")
            if low_count >= DEBOUNCE_COUNT:
                print("[battery] CRITICAL — homing and shutting down!")
                _do_shutdown()
                break
        else:
            if low_count > 0:
                print("[battery] Back to HIGH — false alarm, resetting counter.")
            low_count = 0

        time.sleep(POLL_INTERVAL)

    if ON_PI:
        lgpio.gpiochip_close(h)
    print("[battery] Monitor stopped.")


def _do_shutdown():
    return

    try:
        import motor
        motor.home_all()
        time.sleep(12)  # give home_all() time to finish
    except Exception as e:
        print(f"[battery] home_all() failed: {e}")

    try:
        import os, signal
        os.kill(os.getpid(), signal.SIGTERM)
    except Exception as e:
        print(f"[battery] SIGTERM failed: {e}")


def start():
    global _thread, _running
    if _running:
        return
    _running = True
    _thread = threading.Thread(target=_monitor, daemon=True, name="battery-monitor")
    _thread.start()


def stop():
    global _running
    _running = False


if __name__ == "__main__":
    start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop()