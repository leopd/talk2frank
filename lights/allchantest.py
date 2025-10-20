#!/usr/bin/env python3
import time

import requests
import tqdm

from ola import OlaDMXUniverse

dmx = OlaDMXUniverse(universe=0)

print(f"Setting all channels to 0")
dmx.blackout()
time.sleep(2)

print(f"Setting all channels to 255")
for ch in tqdm.trange(1,512):
    dmx.set1(ch, 255)
    dmx.send()
    time.sleep(0.1)
time.sleep(2)

print(f"Setting all channels to 128")
for ch in tqdm.trange(1,512):
    dmx.set1(ch, 128)
    dmx.send()
    time.sleep(0.1)
time.sleep(2)

printf("Resetting all to 0")
dmx.blackout()