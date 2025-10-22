"""Twinkler
"""
import logging
import random
import sys
import threading
import time
from queue import Empty, Queue

import yaml

from ola import Light, OlaDMXUniverse
from constellation import Constellation

logger = logging.getLogger(__name__)

class Twinkler:
    def __init__(self):
        self.dmx = OlaDMXUniverse()
        self.constellation = Constellation(dmx=self.dmx)

def clip(x: float) -> float:
    return max(0.0, min(1.0, x))

def hsv_to_str(hsv: tuple[float, float, float]) -> str:
    return f"HSV({hsv[0]:.3f}, {hsv[1]:.3f}, {hsv[2]:.3f})"

class TwinkleLight:
    """A light that twinkles.
    It periodically picks a new target HSV value, based on the variances you specify.
    It picks a timescale to fade to that new value.
    """
    def __init__(self, light: Light, config: dict):
        self.light = light
        self._lock = threading.Lock()
        self.config = self.validate_config(config)
        with self._lock:
            base_hsv = self.mid_hsv()
            self._old_hsv = base_hsv
            self._target_hsv = base_hsv
            self._start_time = time.time()
            self._target_time = self._start_time
            self._schedule_new_target_locked()

    def validate_config(self, config: dict) -> dict:
        if not isinstance(config, dict):
            raise ValueError("Config must be a dictionary")
        for k in ["hue", "sat", "val", "tau"]:
            if not k in config:
                raise ValueError(f"Config must contain a '{k}' key")
            if (not isinstance(config[k], list)) or (len(config[k]) != 2):
                raise ValueError(f"Config value for '{k}' must be a list of two floats")
        return config
    
    def mid_hsv(self) -> tuple[float, float, float]:
        return (
            (self.config["hue"][0] + self.config["hue"][1]) / 2,
            (self.config["sat"][0] + self.config["sat"][1]) / 2,
            (self.config["val"][0] + self.config["val"][1]) / 2,
        )

    def _schedule_new_target_locked(self):
        """Select a new HSV target and fade duration for this light."""
        hue_range = self.config["hue"]
        sat_range = self.config["sat"]
        val_range = self.config["val"]
        tau_range = self.config["tau"]

        self._old_hsv = self._target_hsv
        next_hsv = (
            random.uniform(hue_range[0], hue_range[1]),
            random.uniform(sat_range[0], sat_range[1]),
            random.uniform(val_range[0], val_range[1]),
        )
        self._target_hsv = next_hsv
        self._start_time = time.time()
        tdelta = random.uniform(tau_range[0], tau_range[1])
        self._target_time = self._start_time + tdelta
        logger.debug(f"New target for {self.light.name} to {hsv_to_str(self._target_hsv)} in {tdelta:.2f}s")

    def tick(self):
        """Advance the light toward its current target HSV value."""
        with self._lock:
            now = time.time()
            duration = self._target_time - self._start_time
            if duration <= 0.0:
                current_hsv = self._target_hsv
                schedule_new_target = True
            else:
                progress = clip((now - self._start_time) / duration)
                current_hsv = (
                    self._old_hsv[0] + (self._target_hsv[0] - self._old_hsv[0]) * progress,
                    self._old_hsv[1] + (self._target_hsv[1] - self._old_hsv[1]) * progress,
                    self._old_hsv[2] + (self._target_hsv[2] - self._old_hsv[2]) * progress,
                )
                schedule_new_target = progress >= 1.0
            if schedule_new_target:
                self._schedule_new_target_locked()
        self.light.hsv(current_hsv[0], current_hsv[1], current_hsv[2])

    def update_config(self, config: dict):
        """Update the twinkle configuration and reset the fade targets."""
        with self._lock:
            validated_config = self.validate_config(config)
            self.config = validated_config
            # Don't change the light's current value immediately - wait until the next cycle.

class TwinkleConfigs:
    def __init__(self, config_fn: str = "twinkles.yaml"):
        with open(config_fn, 'r') as f:
            self.raw_config = yaml.safe_load(f)
        self.configs = {}
        for name, config in self.raw_config.items():
            self.configs[name] = config

    def get_config(self, name: str) -> dict:
        return self.configs[name]


def command_listener(command_queue: Queue[str], stop_event: threading.Event):
    """Read commands from stdin and forward them to the main loop."""
    while not stop_event.is_set():
        line = sys.stdin.readline()
        if not line:
            stop_event.set()
            command_queue.put("quit")
            return
        command = line.strip()
        if not command:
            continue
        command_queue.put(command)
        if command.lower() in {"quit", "exit"}:
            stop_event.set()
            return

def handle_command(
    command: str,
    twinkle_lights: list[TwinkleLight],
    configs: TwinkleConfigs,
    stop_event: threading.Event,
):
    """Apply a command read from stdin to the running twinkle show."""
    normalized = command.strip().lower()
    # Strip trailing punctuation
    normalized = normalized.rstrip(".,!?")

    if normalized in {"quit", "exit"}:
        print("Stopping twinkler loop.")
        stop_event.set()
        return

    if normalized == "list":
        available = ", ".join(sorted(configs.configs.keys()))
        print(f"Available scenes: {available}")
        return

    if normalized not in configs.configs:
        print(f"Unknown scene '{normalized}'. Type 'list' to see available scenes.")
        return

    scene_config = configs.get_config(normalized)
    for light in twinkle_lights:
        light.update_config(scene_config)
    print(f"Switched to scene '{normalized}'.")


def main(scene: str = "", *args: str):
    """Run the twinkler loop while listening for scene change commands."""
    twinkler = Twinkler()
    tconfigs = TwinkleConfigs()
    twinkle_lights: list[TwinkleLight] = []

    if not scene:
        scene = "fire"
    if scene not in tconfigs.configs:
        print(f"Unknown start scene '{scene}'. Falling back to 'fire'.")
        scene = "fire"

    conf = tconfigs.get_config(scene)
    print(f"Using config for {scene}: {conf}")

    for light in twinkler.constellation.lights.values():
        twinkle_light = TwinkleLight(light, config=conf)
        twinkle_lights.append(twinkle_light)

    command_queue: Queue[str] = Queue()
    stop_event = threading.Event()
    listener = threading.Thread(
        target=command_listener,
        args=(command_queue, stop_event),
        daemon=True,
    )
    listener.start()

    print("Type a scene name to switch, 'list' to show options, or 'quit' to exit.")

    try:
        while True:
            while True:  # tight loop to handle all pending commands
                try:
                    pending_command = command_queue.get_nowait()
                except Empty:
                    break
                handle_command(pending_command, twinkle_lights, tconfigs, stop_event)
            if stop_event.is_set():
                break

            for twinkle_light in twinkle_lights:
                twinkle_light.tick()
            time.sleep(0.01)
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop_event.set()
        listener.join(timeout=1.0)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(*sys.argv[1:])