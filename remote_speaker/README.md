# Remote Speaker

Simple FastAPI service that plays uploaded WAV files synchronously.

## Run

```bash
uv run uvicorn remote_speaker.spserver:app --host 0.0.0.0 --port 8124
```

## API

- POST `/play/wav` with multipart form-data:
  - `wav_file`: the WAV file to play

Response is JSON and only returns after playback completes.

### Example

```bash
curl -X POST http://$REMOTE_SPEAKER_HOST:8124/play/wav \
  -F 'wav_file=@voice/monster.wav' \
  -H 'Expect:'
```

The `Expect:` header helps avoid 100-continue delays with some proxies.


