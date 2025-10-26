# Talk to frank

Several components:

- `lights/` - DMX lights
- `ears/` - Whisper for speech recognition
- `voice/` - Voice generation
- `snark/` - LLM brains for snarking at passersby

## DMX setup

See [lights/README.md](lights/README.md) for more details.


## Ears

See [ears/README.md](ears/README.md) for more details.

Building whisper-cpp needs CUDA and a lot of RAM.  At least 24GB (CPU RAM) - turn on swap!

```
cd ears
./setup.sh
```


## Ears + Lights Demo

```
./earlight.sh
```

and then try saying "fire" or "black" or "water".