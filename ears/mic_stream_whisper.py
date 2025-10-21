#!/usr/bin/env python3
import argparse
import queue
import sys

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from typing import Any


class WhisperListener:

    def __init__(self, model: str, device: str = "cuda", dtype: str = "float16"):
        self.model = WhisperModel(model, device=device, compute_type=dtype)
        self.q = queue.Queue()

    def on_audio(
        self,
        indata: np.ndarray,
        frames: int,
        time: Any,
        status: Any,
    ):
        """Receive audio buffers from PortAudio and queue them for transcription."""

        if status:
            print(status, file=sys.stderr)
        self.q.put(indata.copy())

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the Whisper microphone streaming demo."""

    parser = argparse.ArgumentParser(description="Stream microphone audio into Whisper.")
    parser.add_argument("--model", default="small", help="Name of the Whisper model to load.")
    parser.add_argument("--device", default="cuda", help="Device for inference, for example 'cuda' or 'cpu'.")
    parser.add_argument(
        "--dtype",
        default="float16",
        help="Compute precision, such as 'float16', 'int8_float16', or 'float32'.",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Audio sample rate in Hertz.")
    parser.add_argument("--block-ms", type=int, default=30, help="Block size in milliseconds.")
    parser.add_argument("--window-ms", type=int, default=2000, help="Sliding window size in milliseconds.")
    parser.add_argument("--input-device", type=int, help="Input device index for PortAudio.")
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List available PortAudio devices and exit.",
    )
    return parser.parse_args()


def main():
    """Capture microphone audio and stream transcription results to stdout."""

    args = parse_args()
    listener = WhisperListener(args.model, args.device, args.dtype)

    if args.list_devices:
        print(sd.query_devices())
        return

    buf = np.zeros((0, 1), dtype=np.float32)
    with sd.InputStream(
        channels=1,
        samplerate=args.sample_rate,
        blocksize=int(args.sample_rate * args.block_ms / 1000),
        dtype="float32",
        callback=listener.on_audio,
        device=args.input_device,
    ):
        print("Listening. Ctrl+C to stop.")
        try:
            need = int(args.sample_rate * args.window_ms / 1000)
            while True:
                newbuf = listener.q.get()
                buf = np.concatenate([buf, newbuf])
                if buf.size >= need:
                    chunk = buf[-need:]
                    segments, _ = listener.model.transcribe(
                        chunk, beam_size=1, vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=250)
                    )
                    text = "".join(s.text for s in segments).strip()
                    if text:
                        print(text, flush=True)
                    else:
                        print(f"Nothing.  Newbuf: {newbuf.min()}, {newbuf.mean()}, {newbuf.max()}")
                else:
                    print(f"Buffering... {buf.size} / {need}")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()