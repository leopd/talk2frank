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


