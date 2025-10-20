#!/usr/bin/env python3
import time

import requests

from ola import Light
from ola import OlaDMXUniverse

dmx = OlaDMXUniverse(universe=0)
light = Light(dmx, 1)
light2 = Light(dmx, 9)

try:
    while True:
        for h in range(0, 360, 1):
            light.hsv(h/360, 1.0, 1.0)
            light2.hsv((h+90)/360, 1.0, 1.0)
            time.sleep(0.01)
finally:
    dmx.blackout()

