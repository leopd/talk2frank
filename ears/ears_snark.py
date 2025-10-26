#!/usr/bin/env python3
import argparse
import os
import sys
from typing import Optional

import numpy as np
import requests
import sounddevice as sd

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


def post_phrase_and_play(snark_url: str, text: str):
    response = requests.post(
        f"{snark_url.rstrip('/')}/api/tts",
        json={"text": text},
        timeout=30,
    )
    response.raise_for_status()

    audio_bytes = response.content
    audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
    sr_header: Optional[str] = response.headers.get("X-Audio-Sample-Rate")
    if not sr_header:
        raise RuntimeError("Snark server did not include X-Audio-Sample-Rate header")
    sample_rate = int(sr_header)

    sd.play(audio_array, samplerate=sample_rate)
    sd.wait()


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
            post_phrase_and_play(snark_url, text)
        except Exception as exc:
            print(f"Error contacting snark server: {exc}", file=sys.stderr)


if __name__ == "__main__":
    main()


