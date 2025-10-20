#!/usr/bin/env python3
import time

import requests

from ola import OlaDMXUniverse

dmx = OlaDMXUniverse(universe=0)

try:
    while True:
        for argb in [(255,255,0,0),(255,0,255,0),(255,0,0,255)]:
            dmx.set(1, argb)
            dmx.send()
            time.sleep(1)
finally:
    dmx.blackout()

