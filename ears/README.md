# Ears

This uses [faster-whisper](https://github.com/guillaumekln/faster-whisper) to transcribe audio from a microphone.

We use PipeWire as the microphone backend, which speaks the PulseAudio protocol.
(This all relies on ALSA, which is the kernel level stuff, but we won't touch it directly.)

## Setup

```
./setup.sh
```

## Run

```
python mic_stream_whisper.py
```

### Snark mode

`ears_snark.sh` captures phrases and sends them to a remote snark server, then plays the returned audio.

Environment variable:

- `SNARK_SERVER_URL`: Base URL of the snark server, e.g. `https://snark.example.com`

Example:

```
export SNARK_SERVER_URL="https://snark.example.com"
./ears_snark.sh --model tiny.en --device cuda
```

Protocol:

- Client POSTs to `${SNARK_SERVER_URL}/api/tts` with JSON `{ "text": "your phrase" }`.
- Response body: raw 32-bit float PCM samples.
- Response header `X-Audio-Sample-Rate`: playback sample rate in Hz.

## Troubleshooting

**`sounddevice.PortAudioError: Error querying device -1`**

You need to configure a microphone.  Use the `--list-devices` option to see what's available.

```
python mic_stream_whisper.py --list-devices
```



