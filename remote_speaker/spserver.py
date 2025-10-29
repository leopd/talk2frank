from typing import Optional

import io
import numpy as np  # type: ignore
import sounddevice as sd  # type: ignore
import soundfile as sf  # type: ignore
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse


MAX_AUDIO_LENGTH_SECONDS = 30


app = FastAPI(title="Remote Speaker", version="0.1.0")


@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.post("/play/wav")
async def play_wav(wav_file: UploadFile = File(...)):
    """Accept a WAV upload and play it synchronously.

    The request will not return until audio playback completes.
    """
    data = await wav_file.read()

    wav_io = io.BytesIO(data)
    audio_array, sample_rate = sf.read(wav_io, dtype="float32")
    if audio_array.ndim > 1:
        audio_array = audio_array[:, 0]

    audio_length_seconds = audio_array.shape[0] / sample_rate if sample_rate else 0.0
    if audio_length_seconds > MAX_AUDIO_LENGTH_SECONDS:
        audio_array = audio_array[: int(sample_rate * MAX_AUDIO_LENGTH_SECONDS)]

    print(f"[remote_speaker] playing audio {audio_length_seconds:.1f} seconds")
    sd.play(audio_array, samplerate=sample_rate)
    sd.wait()
    print("[remote_speaker] done playing audio")

    return JSONResponse(
        {
            "status": "ok",
            "received_seconds": audio_length_seconds,
        }
    )


