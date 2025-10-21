#!/usr/bin/env python3
import colorsys
import time

import requests

class OlaDMXUniverse:

    def __init__(self, universe:int=0, host:str="http://127.0.0.1:9090", max_fps:float=40.0):
        self.universe = universe
        self.host = host
        self.data = [0] * 512
        self.max_fps = max_fps
        self.last_send_time = time.time()
        self.universe_start_time = time.time()
        self.send_cnt = 0

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
        self.send(force=True)

    def send(self, force: bool = False):
        """Send the current DMX buffer to OLA."""
        if not force:
            now = time.time()
            if now - self.last_send_time < 1.0 / self.max_fps:
                return
        self.last_send_time = now
        payload = {
            "u": self.universe,
            "d": ",".join(map(str, self.data))
        }
        requests.post(f"{self.host}/set_dmx", data=payload, timeout=2)
        self.send_cnt += 1
        if self.send_cnt % 1000 == 0:
            #print(f"Sent {self.send_cnt} frames in {now - self.universe_start_time:.2f}s")
            pass


class Colors:
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 0, 255)
    white = (255, 255, 255)
    yellow = (255, 255, 0)
    cyan = (0, 255, 255)
    magenta = (255, 0, 255)
    orange = (255, 50, 0)
    purple = (128, 0, 128)
    pink = (255, 66, 66)
    brown = (165, 42, 42)
    gray = (128, 128, 128)
    black = (0, 0, 0)

    @staticmethod
    def by_name(color: str) -> tuple[int, int, int]:
        return getattr(Colors, color)

    @staticmethod
    def hsv_to_rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
        """Convert HSV (0–1 floats) to RGB (0–255 ints)."""
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return int(r * 255), int(g * 255), int(b * 255)


class Light:
    """An WRGB light."""

    def __init__(self, dmx: OlaDMXUniverse, channel: int, name: str = ""):
        self.dmx = dmx
        self.channel = channel
        self._rgbf_val = [0, 0, 0, 0]
        if name:
            self.name = name
        else:
            self.name = f"Channel {channel}"

    def __str__(self):
        return f"Light(name={self.name})"

    def rgbf(self, r: int, g: int, b: int, f: int = 255):
        """Takes 0-255 ints for r, g, b, f.
        R is red, G is green, B is blue.
        F is the fader channel, and determines overall brightness.
        """
        self._rgbf_val = [f, r, g, b]
        self.dmx.set(self.channel, self._rgbf_val)
        self.dmx.send()

    def hsv(self, h: float, s: float, v: float):
        """Takes 0-1 float for h, s, v.
        H is the hue, s is the saturation, v is the value.
        For hue, 0 is red, 1/6 is green, 1/3 is blue, 1/2 is yellow, 2/3 is cyan, 5/6 is magenta, 1 is red again.
        """
        hue = h % 1.0
        r,g,b = Colors.hsv_to_rgb(hue, s, v)
        f = 255 * v

        r = max(0, min(255, r))
        g = max(0, min(255, g))
        b = max(0, min(255, b))
        f = max(0, min(255, f))

        self.rgbf(r, g, b, f)

    def color(self, color:str):
        rgb = Colors.by_name(color)
        self.rgbf(rgb[0], rgb[1], rgb[2])

    def get_hsv(self) -> tuple[float, float, float]:
        r, g, b = self._rgbf_val[1:4]
        h, s, v = colorsys.rgb_to_hsv(r/255, g/255, b/255)
        return h, s, v