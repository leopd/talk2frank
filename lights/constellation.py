from dataclasses import dataclass
import yaml

from ola import Light, OlaDMXUniverse

@dataclass
class LightDefn:
    name: str
    channel: int


class Constellation:
    """A collection of lights, configured in a yaml file."""

    def __init__(self, config_fn: str = "constellation.yaml", dmx: OlaDMXUniverse = None):
        self.light_defs = {}
        self.dmx = dmx
        self.lights = {}
        self.load(config_fn)

    def load(self, config_fn: str):
        with open(config_fn, 'r') as f:
            self.raw_config = yaml.safe_load(f)

        for light_json in self.raw_config['lights']:
            defn = LightDefn(light_json['name'], light_json['channel'])
            self.light_defs[defn.name] = defn
            print(f"Loaded light {defn.name} on channel {defn.channel}")
            self.lights[defn.name] = Light(self.dmx, defn.channel, name=defn.name)

        print(f"Loaded {len(self.light_defs)} lights from {config_fn}")

