#!/usr/bin/env python3
import colorsys
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


def hsv_to_rgb_bytes(h: float, s: float, v: float) -> tuple[int, int, int]:
    """Convert HSV (0–1 floats) to RGB (0–255 ints)."""
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

class Light:
    """An WRGB light."""

    def __init__(self, dmx: OlaDMXUniverse, channel: int):
        self.dmx = dmx
        self.channel = channel

    def rgbf(self, r: int, g: int, b: int, f: int = 255):
        """Takes 0-255 ints for r, g, b, f.
        R is red, G is green, B is blue.
        F is the fader channel, and determines overall brightness.
        """
        self.dmx.set(self.channel, [f, r, g, b])
        self.dmx.send()

    def hsv(self, h: float, s: float, v: float):
        """Takes 0-1 float for h, s, v.
        H is the hue, s is the saturation, v is the value.
        For hue, 0 is red, 1/6 is green, 1/3 is blue, 1/2 is yellow, 2/3 is cyan, 5/6 is magenta, 1 is red again.
        """
        hue = h % 1.0
        r,g,b = hsv_to_rgb_bytes(hue, s, v)
        f = 255 * v

        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        f = max(0, min(255, f))

        self.rgbf(r, g, b, f)

    def color(self, color:str):
        if color == "red":
            self.rgbf(255, 0, 0)
        elif color == "green":
            self.rgbf(0, 255, 0)
        elif color == "blue":
            self.rgbf(0, 0, 255)
        elif color == "white":
            self.rgbf(255, 255, 255)
        elif color == "yellow":
            self.rgbf(255, 255, 0)
        elif color == "cyan":
            self.rgbf(0, 255, 255)
        elif color == "magenta":
            self.rgbf(255, 0, 255)
        elif color == "orange":
            self.rgbf(255, 165, 0)
        elif color == "purple":
            self.rgbf(128, 0, 128)
        elif color == "pink":
            self.rgbf(255, 66, 66)
        elif color == "brown":
            self.rgbf(165, 42, 42)
        elif color == "gray":
            self.rgbf(128, 128, 128)
        elif color == "black":
            self.rgbf(0, 0, 0)
        else:
            raise ValueError(f"Invalid color: {color}")