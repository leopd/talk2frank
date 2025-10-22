#!/usr/bin/env python3

from TTS.api import TTS
import numpy as np
import sounddevice as sd
import torch
import torchaudio.functional as audio_fn

#model = "tts_models/en/ljspeech/tacotron2-DDC"
model = "tts_models/en/vctk/vits"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"downloading model {model}")
tts = TTS(model_name=model, progress_bar=False).to(device)
print(tts.speakers)

audio = tts.tts(text="I am the monster hiding under your bed.",
    speaker="p227")
sample_rate = tts.synthesizer.output_sample_rate
audio_array = np.array(audio, dtype=np.float32)

waveform = torch.from_numpy(audio_array).unsqueeze(0)
pitched_waveform = audio_fn.pitch_shift(waveform, sample_rate, n_steps=-3)
distorted_waveform = torch.tanh(pitched_waveform * 1.4)
processed_audio = distorted_waveform.squeeze(0).numpy()

max_amplitude = float(np.max(np.abs(processed_audio)))
if max_amplitude > 1:
    processed_audio = processed_audio / max_amplitude

sd.play(processed_audio, samplerate=sample_rate * 0.7)
sd.wait()

