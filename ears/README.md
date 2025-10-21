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

## Troubleshooting

**`sounddevice.PortAudioError: Error querying device -1`**

You need to configure a microphone.  Use the `--list-devices` option to see what's available.

```
python mic_stream_whisper.py --list-devices
```

