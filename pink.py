from ola import OlaDMXUniverse
from ola import Light

dmx = OlaDMXUniverse()
light1 = Light(dmx, 1)
light2 = Light(dmx, 9)

light1.color("pink")
light2.color("pink")
