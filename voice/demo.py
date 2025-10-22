#!/usr/bin/env python3

from TTS.api import TTS
import numpy as np
import sounddevice as sd
import torch
import torchaudio.functional as audio_fn


def demo_speakers(model_name: str, pitch_down_steps: int, sample_rate_factor: float):
    """Load a TTS model and play through all speakers with audio processing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"downloading model {model_name}")
    tts = TTS(model_name=model_name, progress_bar=False).to(device)

    sample_rate = tts.synthesizer.output_sample_rate
    text = "I am the monster hiding under your bed."

    for speaker in tts.speakers:
        print(f"Playing speaker: {speaker}")
        
        audio = tts.tts(text=text, speaker=speaker)
        audio_array = np.array(audio, dtype=np.float32)
        
        waveform = torch.from_numpy(audio_array).unsqueeze(0)
        pitched_waveform = audio_fn.pitch_shift(waveform, sample_rate, n_steps=pitch_down_steps)
        distorted_waveform = torch.tanh(pitched_waveform * 1.4)
        processed_audio = distorted_waveform.squeeze(0).numpy()
        
        max_amplitude = float(np.max(np.abs(processed_audio)))
        if max_amplitude > 1:
            processed_audio = processed_audio / max_amplitude
        
        sd.play(processed_audio, samplerate=sample_rate * sample_rate_factor)
        sd.wait()


if __name__ == "__main__":
    demo_speakers(
        model_name="tts_models/en/vctk/vits",
        pitch_down_steps=-3,
        sample_rate_factor=0.7,
    )

