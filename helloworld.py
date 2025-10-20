#!/usr/bin/env python3
import time
from DMXEnttecPro import Controller

dmx = Controller('/dev/ttyUSB0', auto_submit=True)  # adjust port as needed


try:
    offset = 0
    while True:
        # red, green, blue
        for r, g, b in [(255,0,0), (0,255,0), (0,0,255)]:
            dmx.set_channel(offset + 1, r)
            dmx.set_channel(offset + 2, g)
            dmx.set_channel(offset + 3, b)
            print(f"Setting channels at {offset=} to {r},{g},{b}")
            time.sleep(0.1)
            offset += 4
finally:
    # blackout
    for ch in (1,2,3):
        dmx.set_channel(ch, 0)
    dmx.submit()
