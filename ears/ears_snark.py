#!/usr/bin/env python3
import argparse
import io
import os
import sys
import traceback
from typing import Optional

import numpy as np
import requests
import sounddevice as sd
import soundfile as sf

from .ear_guts import WhisperStreamer


def parse_args():
    parser = argparse.ArgumentParser(description="Stream mic, send phrases to snark server, play reply.")
    parser.add_argument("--model", default="tiny.en", help="Whisper model name")
    parser.add_argument("--device", default="cuda", help="Inference device: cuda or cpu")
    parser.add_argument("--dtype", default="float16", help="Compute precision for Whisper")
    parser.add_argument("--sample-rate", type=int, default=16000, help="Mic sample rate in Hz")
    parser.add_argument("--block-ms", type=int, default=500, help="Audio block length in ms")
    parser.add_argument("--window-ms", type=int, default=2000, help="Inference window length in ms")
    parser.add_argument("--input-device", type=int, help="Input device index for PortAudio")
    parser.add_argument("--min-phrase-len", type=int, default=3, help="Minimum characters before sending")
    return parser.parse_args()


def post_phrase_and_play(base_url: str, text: str):
    url = f"{base_url.rstrip('/')}/infer/text"
    response = requests.post(
        url,
        data={
            "prompt": text,
            "max_new_tokens": 200,
            "response_format": "wav",
        },
        timeout=60,
    )
    if not response.ok:
        details = response.text[:256]
        ct = response.headers.get("Content-Type", "")
        raise RuntimeError(f"HTTP {response.status_code} from {url} (Content-Type: {ct}): {details}")
    print(f"[snark] got response {response.status_code}")

    audio_bytes = response.content
    wav_io = io.BytesIO(audio_bytes)
    audio_array, sample_rate = sf.read(wav_io, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    audio_length_seconds = audio_array.shape[0] / sample_rate
    print(f"[snark] playing audio {audio_length_seconds:.1f} seconds")
    sd.play(audio_array, samplerate=sample_rate)
    sd.wait()
    print(f"[snark] done playing audio")


def main():
    args = parse_args()

    snark_url = os.environ.get("SNARK_SERVER_URL")
    if not snark_url:
        print("SNARK_SERVER_URL must be set to the base URL of the snark server, e.g. https://snark.example.com", file=sys.stderr)
        sys.exit(2)

    streamer = WhisperStreamer(
        model_name=args.model,
        device=args.device,
        compute_type=args.dtype,
        sample_rate=args.sample_rate,
        block_ms=args.block_ms,
        window_ms=args.window_ms,
        input_device=args.input_device,
    )

    print("Listening. Ctrl+C to stop.")
    for event in streamer.stream_events():
        if not event.transcript:
            continue
        text = event.transcript.strip()
        if len(text) < args.min_phrase_len:
            continue
        try:
            print(f"[snark] captured: {text}")
            post_phrase_and_play(snark_url, text)
        except Exception as exc:
            traceback.print_exc()
            print(f"Error contacting snark server: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()


