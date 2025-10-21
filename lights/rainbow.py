#!/usr/bin/env python3
import time

import requests

from ola import Light
from ola import OlaDMXUniverse

dmx = OlaDMXUniverse(universe=0)
light = Light(dmx, 1)
light2 = Light(dmx, 9)
light3 = Light(dmx, 17)
light4 = Light(dmx, 25)

try:
    while True:
        for h in range(0, 360, 1):
            light.hsv(h/360, 1.0, 1.0)
            light2.hsv((h+30)/360, 1.0, 1.0)
            light3.hsv((h+60)/360, 1.0, 1.0)
            light4.hsv((h+90)/360, 1.0, 1.0)
            time.sleep(0.03)
        print(f"Cycled all colors")
finally:
    dmx.blackout()

