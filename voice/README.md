# Voices

To try different voices, run commands like:

```
uv run python -m voice.tryvoice "I will eat your candy!" -r 0.75 -p -4 -g 1.8 -s p299
uv run python -m voice.tryvoice "I will eat your candy!" -r 0.75 -p -4 -g 1.8 -s p260
uv run python -m voice.tryvoice "I will eat your candy!" -r 0.75 -p -4 -g 1.8 -s p263
uv run python -m voice.tryvoice "I will eat your candy!" -r 0.65 -p -4 -g 1.8 -s p263
uv run python -m voice.tryvoice "I will eat your candy!" -r 0.65 -p -4 -g 0.2 -s p263
uv run python -m voice.tryvoice "I will eat your candy!" -r 0.65 -p -4 -g 2.5 -s p263
uv run python -m voice.tryvoice "I will eat your candy!" -r 0.65 -p -4 -g 5 -s p263
uv run python -m voice.tryvoice "I will eat your candy!" -r 0.75 -p -5 -g 5 -s p263
uv run python -m voice.tryvoice "I will eat your candy!" -r 0.75 -p -5 -g 5 -s p260
```


## Debugging linux sound.

OMG.  What hassle.  Check your volume:

```
alsamixer
```

Check your devices and pick a good one, e.g.

```
pactl list short sinks
pactl set-default-sink alsa_output.usb-M-Audio_AIR_192_4-00.analog-stereo
```
