#!/usr/bin/env python3

from TTS.api import TTS
import torch

model = "tts_models/en/ljspeech/tacotron2-DDC"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
print(f"downloading model {model}")
tts = TTS(model_name=model, progress_bar=False).to(device)
tts.tts_to_file(text="I am the monster hiding under your bed.", file_path="monster.wav")

