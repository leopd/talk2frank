#!/usr/bin/env python3
import argparse
import queue
import sys
import time

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from typing import Any, Dict, Iterable, List, Tuple

WHISPER_SAMPLE_RATE = 16000

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
    parser.add_argument("--model", default="tiny.en", help="Name of the Whisper model to load.")
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


def describe_device(device_index: int) -> Dict[str, Any]:
    """Return metadata for a PortAudio device including supported sample rates."""

    info = sd.query_devices(device_index)

    supported: List[int] = []
    CHECKED_SAMPLE_RATES: Tuple[int, ...] = (
        8000,
        11025,
        16000,
        22050,
        24000,
        32000,
        44100,
        48000,
        88200,
        96000,
    )

    for rate in CHECKED_SAMPLE_RATES:
        try:
            sd.check_input_settings(device=device_index, channels=info["max_input_channels"], samplerate=rate)
        except Exception:
            continue
        supported.append(rate)

    return {
        "index": device_index,
        "name": info["name"],
        "hostapi": sd.query_hostapis(info["hostapi"]) ["name"],
        "max_input_channels": info["max_input_channels"],
        "max_output_channels": info["max_output_channels"],
        "default_samplerate": info["default_samplerate"],
        "supported_samplerates": supported,
    }


def list_devices() -> None:
    """Print a table of PortAudio devices and their supported sample rates."""

    devices: Iterable[int] = range(len(sd.query_devices()))
    rows = [describe_device(i) for i in devices]

    header = "Index  Inputs  Outputs  Default Hz  Supported Rates  Name"
    print(header)
    print("-" * len(header))
    for row in rows:
        supported = ", ".join(str(rate) for rate in row["supported_samplerates"]) or "(none)"
        print(
            f"{row['index']:>5}  {row['max_input_channels']:>6}  {row['max_output_channels']:>7}  "
            f"{int(row['default_samplerate']):>10}  {supported:<16}  {row['name']}"
        )


def main():
    """Capture microphone audio and stream transcription results to stdout."""

    args = parse_args()
    listener = WhisperListener(args.model, args.device, args.dtype)

    if args.list_devices:
        list_devices()
        return

    buf = np.zeros(0, dtype=np.float32)
    with sd.InputStream(
        channels=1,
        samplerate=args.sample_rate,
        blocksize=int(args.sample_rate * args.block_ms / 1000),
        dtype="float32",
        callback=listener.on_audio,
        device=args.input_device,
    ):
        print("Listening. Ctrl+C to stop.")
        if args.sample_rate != WHISPER_SAMPLE_RATE:
            audio_resample_factor = int(round(args.sample_rate / WHISPER_SAMPLE_RATE))
            adjusted_rate = args.sample_rate / audio_resample_factor
            if abs(adjusted_rate - WHISPER_SAMPLE_RATE) > 1:
                print(
                    f"Warning! Downsampling by an integer stride gives {adjusted_rate:.1f} Hz, "
                    f"not the expected {WHISPER_SAMPLE_RATE} Hz"
                )
        else:
            audio_resample_factor = 1

        try:
            need = int(WHISPER_SAMPLE_RATE * args.window_ms / 1000)
            while True:
                start_time = time.time()
                newbuf = listener.q.get()
                elapsed = time.time() - start_time

                newbuf = np.asarray(newbuf, dtype=np.float32).reshape(-1)
                if audio_resample_factor > 1:
                    newbuf = newbuf[::audio_resample_factor]

                print(f"Capture time: {elapsed*1000:.2f}ms   ", end="")

                buf = np.concatenate((buf, newbuf))

                if buf.size >= need:
                    chunk = buf[-need:]
                    start_time = time.time()
                    segments, _ = listener.model.transcribe(
                        chunk,
                        beam_size=1,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=250),
                    )
                    elapsed = time.time() - start_time
                    print(f"Transcription time: {elapsed*1000:.2f}ms   ", end="")

                    text = "".join(s.text for s in segments).strip()
                    if text:
                        print(text, flush=True)
                    else:
                        mean_square = np.mean(np.square(chunk))
                        if mean_square <= 0:
                            volume = "(silent - no audio)"
                        else:
                            rms_db = 20 * np.log10(mean_square ** 0.5)
                            volume = f"{rms_db:.2f}dB"
                        print(f"Nothing transcribed.  Volume: {volume}")

                    buf = buf[-need:]
                else:
                    print(f"Buffering... {buf.size} / {need}")
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()