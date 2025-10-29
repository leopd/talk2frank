# Talk2Frank

An interactive Halloween installation: a creepy pumpkin-headed character ("Jacqueline" / "Jack") that watches, listens, and responds to passersby with rude/snarky remarks, voice, and lights.

Core components live in subdirectories and can be developed and run independently:

- `snark/` — Vision+Language model server and logic (the "brains")
- `voice/` — Text‑to‑speech synthesis and post‑processing (the "voice")
- `ears/` — Microphone capture and speech‑to‑text (the "ears")
- `eyes/` — Camera utilities and visual demos (the "eyes")
- `lights/` — DMX lighting control via OLA (the "lights")

The system prompt/persona for the character is defined in `snark/prompt.txt`.

## Prereqs and tooling

- Python managed with `uv` only. See `AGENT.md`.
- Python 3.11 (per `pyproject.toml`).
- CUDA GPU strongly recommended for VLM, TTS, and ASR performance.

Install dependencies:

```
uv sync
```

## Quickstart

1) Start the snark server (VLM + optional TTS responses):

```
./snark-server.sh
```

By default uses `Qwen/Qwen2.5-VL-7B-Instruct`. To change size (e.g., `14B`, `32B`):

```
export VLM_SIZE="14B"
./snark-server.sh
```

2) Try text or image inference (see `snark/README.md` for more):

```
curl -X POST http://localhost:8123/infer/text -F 'prompt=Say hello' -F 'max_new_tokens=50'
curl -X POST http://localhost:8123/infer/text -F 'prompt=Growl at me' -F 'response_format=wav' --output reply.wav
```

3) Run the microphone capture client in snark mode to round‑trip TTS:

```
cd ears
export SNARK_SERVER_URL="http://localhost:8123"
./ears_snark.sh --model tiny.en --device cuda
```

If configured, you can also run a simple ears+lights demo:

```
./earlight.sh
```

## Project layout and pointers

- `snark/` — FastAPI server exposing `/infer/text` and `/infer/image` endpoints. Reads persona from `snark/prompt.txt` at startup and on each request.
  - Start: `uv run uvicorn snark.server:app --host 0.0.0.0 --port 8123` or `./snark-server.sh`
  - Benchmark: `uv run python -m snark.benchmark_vlm`
  - Details and curl examples: see `snark/README.md`

- `voice/` — TTS via `TTS` library with creepy post‑processing (pitch down, mild distortion). Try different voices with:
  - `uv run python -m voice.tryvoice "I will eat your candy!" -r 0.75 -p -4 -g 1.8 -s p260`
  - More in `voice/README.md`

- `ears/` — ASR using faster‑whisper; PipeWire/PulseAudio backend.
  - Setup: `cd ears && ./setup.sh`
  - Mic stream: `uv run python ears/mic_stream_whisper.py`
  - Snark client: `./ears_snark.sh` (uses `${SNARK_SERVER_URL}/infer/text` with `response_format=wav`)
  - More in `ears/README.md`

- `eyes/` — Camera helpers and a simple “insultcam” demo.
  - `uv run framegrab autodiscover`
  - `uv run python eyes/insultcam.py`
  - More in `eyes/README.md`

- `lights/` — DMX control driven by OLA.
  - Install OLA with `lights/setup.sh`, then test with `uv run python lights/allchantest.py`
  - Define fixtures in `lights/constellation.yaml`
  - Troubleshooting in `lights/README.md`

## Persona and content

The character is intentionally mean‑spirited and scary for Halloween. The system prompt in `snark/prompt.txt` defines behavior and tone (e.g., threats, spooky recipes, playful “Boo!”, “Trick or treat!”, etc.). Adjust that file to change the vibe. You can hot‑reload by editing the file; the server reloads the prompt on requests.

## Notes on performance

- Flash‑Attention can improve VLM speed; it is optional and can require careful installation. See `snark/README.md` for guidance.
- GPU memory needs rise with model size. Start with `7B` unless you know your hardware can handle more.
- TTS and ASR also benefit from GPU.

## Development

- Install dev deps and run tests:

```
uv sync --dev
uv run pytest
```

- This repository uses `uv` for everything; do not manage venvs manually. See `AGENT.md`.