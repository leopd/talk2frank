import time

from ola import Light, OlaDMXUniverse
from constellation import Constellation

dmx = OlaDMXUniverse()
constellation = Constellation(dmx=dmx)

for name, light in constellation.lights.items():
    print(f"Setting {name} to pink")
    light.color("pink")
    time.sleep(1)

print("Done")
