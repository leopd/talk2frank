#!/usr/bin/env python3
import argparse
import logging
import sys
from typing import Any, Dict, List

import sounddevice as sd

from .ear_guts import (
    CHECKED_SAMPLE_RATES,
    StreamingResult,
    WhisperStreamer,
    describe_device,
    list_devices,
)

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Whisper microphone streaming demo."""

    parser = argparse.ArgumentParser(description="Stream microphone audio into Whisper.")
    parser.add_argument("--model", default="tiny.en", help="Name of the Whisper model to load.")
    parser.add_argument("--device", default="cuda", help="Device for inference, for example 'cuda' or 'cpu'.")
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Compute precision, such as 'float16', 'int8_float16', or 'float32'.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate in Hertz.")
    parser.add_argument("--block-ms", type=int, default=500, help="Block size in milliseconds.")
    parser.add_argument("--window-ms", type=int, default=2000, help="Sliding window size in milliseconds.")
    parser.add_argument("--input-device", type=int, help="Input device index for PortAudio.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available PortAudio devices and exit.",
    )
    return parser.parse_args()


def describe_device(device_index: int) -> Dict[str, Any]:
    return None  # Backwards compat shim; functionality moved to ear_guts


def list_devices():
    return None  # Backwards compat shim; functionality moved to ear_guts


def main():
    args = parse_args()

    if args.list_devices:
        list_devices()
        return

    try:
        streamer = WhisperStreamer(
            model_name=args.model,
            device=args.device,
            compute_type=args.dtype,
            sample_rate=args.sample_rate,
            block_ms=args.block_ms,
            window_ms=args.window_ms,
            input_device=args.input_device,
        )
    except Exception as exc:
        print(f"Error loading model: {exc}")
        print(
            "Available models: tiny, tiny.en, base, base.en, small, small.en, medium, medium.en, large-v3-turbo"
        )
        print("Maybe others?  Check docs at https://github.com/guillaumekln/faster-whisper")
        sys.exit(1)

    if streamer.resample_warning:
        print(streamer.resample_warning)

    logger.info("Listening. Ctrl+C to stop.")
    for result in streamer.stream_events():
        if result.inference_ms is None:
            logger.info(
                f"Capture time: {result.capture_ms:.2f}ms   "
                f"Buffering... {result.buffered_samples} / {result.required_samples}"
            )
            continue

        message = (
            f"Capture time: {result.capture_ms:.2f}ms   "
            f"Transcription time: {result.inference_ms:.2f}ms   "
        )
        if result.transcript:
            logger.debug(message)
            print(f"{result.transcript}", flush=True)
        else:
            volume = "(silent - no audio)"
            if result.rms_db is not None:
                volume = f"{result.rms_db:.2f}dB"
            logger.debug(f"{message} Nothing transcribed.  Volume: {volume}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
    main()