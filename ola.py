#!/usr/bin/env python3
import time

import requests

class OlaDMXUniverse:

    def __init__(self, universe:int=0, host:str="http://127.0.0.1:9090"):
        self.universe = universe
        self.host = host
        self.data = [0] * 512

    def set1(self, channel: int, value: int):
        """Set DMX channel (1–512) to value (0–255)."""
        if not 1 <= channel <= 512:
            raise ValueError("Channel out of range (1–512)")
        self.data[channel - 1] = max(0, min(255, int(value)))

    def set(self, channel: int, values: list[int]):
        """Set DMX channel (1–512) to values (0–255)."""
        for n in range(len(values)):
            self.set1(channel + n, values[n])

    def get(self, channel: int) -> int:
        """Return current value for a channel."""
        return self.data[channel - 1]

    def blackout(self):
        """Set all channels to 0 and send."""
        self.data = [0] * 512
        self.send()

    def send(self):
        """Send the current DMX buffer to OLA."""
        payload = {
            "u": self.universe,
            "d": ",".join(map(str, self.data))
        }
        requests.post(f"{self.host}/set_dmx", data=payload, timeout=2)

    def stream(self, fps=30):
        """Continuously send the current buffer until interrupted."""
        delay = 1.0 / fps
        try:
            while True:
                self.send()
                time.sleep(delay)
        except KeyboardInterrupt:
            self.blackout()
