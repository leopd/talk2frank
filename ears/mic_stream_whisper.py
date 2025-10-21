#!/usr/bin/env python3
import argparse
import logging
import queue
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

WHISPER_SAMPLE_RATE = 16000
CHECKED_SAMPLE_RATES = (
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


@dataclass
class StreamingResult:
    capture_ms: float
    buffered_samples: int
    required_samples: int
    transcript: Optional[str]
    inference_ms: Optional[float]
    rms_db: Optional[float]


class WhisperStreamer:

    def __init__(
        self,
        model_name: str,
        device: str,
        compute_type: str,
        sample_rate: int,
        block_ms: int,
        window_ms: int,
        input_device: Optional[int],
    ):
        """Initialize the audio streamer and Whisper model."""

        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
        self.sample_rate = sample_rate
        self.block_ms = block_ms
        self.window_ms = window_ms
        self.input_device = input_device
        self.audio_queue: queue.Queue = queue.Queue()
        self.window_samples = max(1, int(WHISPER_SAMPLE_RATE * window_ms / 1000))
        self.block_frames = max(1, int(sample_rate * block_ms / 1000))
        self.downsample_stride, self.resample_warning = self._determine_stride(sample_rate)

    def _determine_stride(self, sample_rate: int) -> Tuple[int, Optional[str]]:
        """Pick an integer stride for crude downsampling and note any mismatch warnings."""

        if sample_rate <= 0:
            return 1, "Invalid sample rate; using Whisper default stride."

        if sample_rate == WHISPER_SAMPLE_RATE:
            return 1, None

        stride = max(1, int(round(sample_rate / WHISPER_SAMPLE_RATE)))
        effective_rate = sample_rate / stride
        if abs(effective_rate - WHISPER_SAMPLE_RATE) <= 1:
            return stride, None

        warning = (
            f"Downsampling by stride {stride} feeds {effective_rate:.1f} Hz to Whisper; "
            f"it expects {WHISPER_SAMPLE_RATE} Hz."
        )
        return stride, warning

    def on_audio(self, indata: np.ndarray, frames: int, time_info: Any, status: Any):
        """Receive audio buffers from PortAudio and queue them for transcription."""

        if status:
            print(status, file=sys.stderr)
        self.audio_queue.put(indata.copy())

    def _downsample(self, samples: np.ndarray) -> np.ndarray:
        """Apply integer-stride downsampling when the input rate exceeds Whisper's rate."""

        if self.downsample_stride <= 1:
            return samples
        return samples[::self.downsample_stride]

    def stream_events(self) -> Iterable[StreamingResult]:
        """Yield streaming results that include timing data and transcription text."""

        buffer = np.zeros(0, dtype=np.float32)
        try:
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_frames,
                dtype="float32",
                callback=self.on_audio,
                device=self.input_device,
            ):
                while True:
                    capture_start = time.perf_counter()
                    raw_chunk = self.audio_queue.get()
                    capture_ms = (time.perf_counter() - capture_start) * 1000.0

                    samples = np.asarray(raw_chunk, dtype=np.float32).reshape(-1)
                    samples = self._downsample(samples)
                    buffer = np.concatenate((buffer, samples))

                    if buffer.size < self.window_samples:
                        yield StreamingResult(
                            capture_ms=capture_ms,
                            buffered_samples=buffer.size,
                            required_samples=self.window_samples,
                            transcript=None,
                            inference_ms=None,
                            rms_db=None,
                        )
                        continue

                    buffer = buffer[-self.window_samples:]
                    inference_start = time.perf_counter()
                    segments, _ = self.model.transcribe(
                        buffer,
                        beam_size=1,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=250),
                    )
                    inference_ms = (time.perf_counter() - inference_start) * 1000.0

                    text = "".join(segment.text for segment in segments).strip()
                    rms_db = None
                    if not text:
                        mean_square = float(np.mean(np.square(buffer)))
                        if mean_square > 0:
                            rms_db = 20 * np.log10(mean_square ** 0.5)

                    yield StreamingResult(
                        capture_ms=capture_ms,
                        buffered_samples=buffer.size,
                        required_samples=self.window_samples,
                        transcript=text if text else None,
                        inference_ms=inference_ms,
                        rms_db=rms_db,
                    )
        except KeyboardInterrupt:
            return


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
    """Return metadata for a PortAudio device including supported sample rates."""

    info = sd.query_devices(device_index)

    supported: List[int] = []
    if info["max_input_channels"] > 0:
        for rate in CHECKED_SAMPLE_RATES:
            try:
                sd.check_input_settings(
                    device=device_index,
                    channels=info["max_input_channels"],
                    samplerate=rate,
                )
            except Exception:
                continue
            supported.append(rate)

    host_api = sd.query_hostapis(info["hostapi"])["name"]
    return {
        "index": device_index,
        "name": info["name"],
        "hostapi": host_api,
        "max_input_channels": info["max_input_channels"],
        "max_output_channels": info["max_output_channels"],
        "default_samplerate": info["default_samplerate"],
        "supported_samplerates": supported,
    }


def list_devices():
    """Print a table of PortAudio devices and their supported sample rates."""

    rows = [describe_device(i) for i in range(len(sd.query_devices()))]

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