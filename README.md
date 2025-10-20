# Talk to frank

Several components:

- `lights/` - DMX lights
- `ears/` - Whisper for speech recognition
- `voice/` - Voice generation

## DMX setup

See [lights/README.md](lights/README.md) for more details.


## Ears

Currently uses `whisper-cpp`.  
TODO: Replace with faster-whisper.

Building whisper-cpp needs CUDA and a lot of RAM.  (At least 24GB - turn on swap!)

```
cd ears
./setup.sh
```

Test it out:
```
# live mic -> text (PulseAudio/ALSA default device)
./build/bin/whisper-cli -m ./models/ggml-base.en.bin --capture --real-time --print-colors
```