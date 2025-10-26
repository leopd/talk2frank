#!/usr/bin/env python3
import random

import numpy as np
import sounddevice as sd
import torch
from .tts_guts import TtsSynthesizer


def demo_speakers(model_name: str, pitch_down_steps: int, sample_rate_factor: float, speed:float):
    """Load a TTS model and play through all speakers with audio processing."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"downloading model {model_name}")
    synth = TtsSynthesizer(
        model_name=model_name,
        pitch_down_steps=pitch_down_steps,
        distortion_gain=1.4,
        output_sample_rate_factor=sample_rate_factor,
    )
    text = "I am the monster hiding under your bed."

    speaker_list = synth.speakers
    random.shuffle(speaker_list)
    for speaker in speaker_list:
        print(f"Playing speaker: {speaker}")
        audio_array, header_rate = synth.synthesize_array(text=text, speaker=speaker)
        sd.play(audio_array, samplerate=header_rate)
        sd.wait()


if __name__ == "__main__":
    demo_speakers(
        model_name="tts_models/en/vctk/vits",  # Pretty good with p298, p260, p255
        #model_name="tts_models/en/ek1/vits",
        pitch_down_steps=+5,
        sample_rate_factor=0.4,
        speed=0.2,
    )

