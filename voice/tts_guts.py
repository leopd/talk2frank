from typing import Iterable, Optional

import numpy as np
import torch
import soundfile as sf
import torchaudio.functional as audio_fn
from io import BytesIO

try:
    from TTS.api import TTS  # type: ignore
except Exception:  # pragma: no cover - import tested indirectly
    TTS = None  # type: ignore


PREFERRED_SPEAKERS: list[str] = [
    "p260",
    #"p299",
]


class TtsSynthesizer:
    def __init__(
        self,
        model_name: str = "tts_models/en/vctk/vits",
        #pitch_down_steps: int = -5,
        #distortion_gain: float = 5.0,
        #output_sample_rate_factor: float = 0.75,
        pitch_down_steps: int = -2,
        distortion_gain: float = 5.0,
        output_sample_rate_factor: float = 0.85,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if TTS is None:
            raise RuntimeError("TTS library not available")
        self._tts = TTS(model_name=model_name, progress_bar=False).to(device)
        self._pitch_down_steps = pitch_down_steps
        self._distortion_gain = distortion_gain
        self._output_sample_rate_factor = output_sample_rate_factor

    def pick_speaker(self, speakers: Optional[Iterable[str]]) -> Optional[str]:
        if not speakers:
            return None
        speaker_set = set(s.lower() for s in speakers)
        for preferred in PREFERRED_SPEAKERS:
            if preferred in speaker_set:
                return preferred
        return None

    def _process_audio(self, audio_array: np.ndarray, sample_rate: int) -> np.ndarray:
        """Apply creepy post-processing: pitch down and mild distortion, normalize to [-1, 1]."""
        waveform = torch.from_numpy(audio_array).unsqueeze(0)
        pitched_waveform = audio_fn.pitch_shift(waveform, int(sample_rate), n_steps=self._pitch_down_steps)
        distorted_waveform = torch.tanh(pitched_waveform * float(self._distortion_gain))
        processed_audio = distorted_waveform.squeeze(0).numpy()
        max_amplitude = float(np.max(np.abs(processed_audio)))
        if max_amplitude > 1.0 and max_amplitude != 0.0:
            processed_audio = processed_audio / max_amplitude
        return processed_audio.astype(np.float32, copy=False)

    def synthesize_array(self, text: str, speaker: Optional[str] = None) -> tuple[np.ndarray, int]:
        """Return processed audio array and header sample rate used for slowdown.

        This centralizes speaker selection, post-processing, and slowdown rate.
        """
        effective_speaker = speaker or self.pick_speaker(getattr(self._tts, "speakers", None)) or None
        tts_args = {
            "text": text,
        }
        if effective_speaker:
            tts_args["speaker"] = effective_speaker
        audio = self._tts.tts(**tts_args)
        audio_array = np.asarray(audio, dtype=np.float32)
        sample_rate = int(self._tts.synthesizer.output_sample_rate)
        processed = self._process_audio(audio_array, sample_rate)
        header_rate = max(1, int(sample_rate * float(self._output_sample_rate_factor)))
        return processed, header_rate

    def synthesize_wav(self, text: str) -> bytes:
        audio_array, header_rate = self.synthesize_array(text)
        buf = BytesIO()
        sf.write(buf, audio_array, samplerate=header_rate, format="WAV")
        return buf.getvalue()

    @property
    def speakers(self) -> list[str]:
        raw = getattr(self._tts, "speakers", None)
        return list(raw or [])


