# Ears

This uses [faster-whisper](https://github.com/guillaumekln/faster-whisper) to transcribe audio from a microphone.

We use PipeWire as the microphone backend, which speaks the PulseAudio protocol.
(This all relies on ALSA, which is the kernel level stuff, but we won't touch it directly.)

## Setup

```
./setup.sh
```

Note: Building whisper-cpp needs CUDA and a lot of RAM.  At least 24GB (CPU RAM) - turn on swap!

## Run

```
python mic_stream_whisper.py
```

### Snark mode (client)

`ears_snark.sh` captures phrases and sends them to a running snark server, then plays the returned audio.

Minimal usage:

```
export SNARK_SERVER_URL="http://localhost:8123"
./ears_snark.sh --model tiny.en --device cuda
```

For server endpoint details and curl examples, see `../snark/README.md`.

## Troubleshooting

**`sounddevice.PortAudioError: Error querying device -1`**

You need to configure a microphone.  Use the `--list-devices` option to see what's available.

```
python mic_stream_whisper.py --list-devices
```



