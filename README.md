# Talk to frank

Several components:

- `lights/` - DMX lights
- `ears/` - Whisper for speech recognition
- `voice/` - Voice generation

## DMX setup

See [lights/README.md](lights/README.md) for more details.


## Ears

See [ears/README.md](ears/README.md) for more details.

Building whisper-cpp needs CUDA and a lot of RAM.  (At least 24GB - turn on swap!)

```
cd ears
./setup.sh
```


## Ears + Lights Demo

```
uv run ../ears/mic_stream_whisper.py | uv run twinkler.py black
```

and then try saying "fire" or "black" or "water".