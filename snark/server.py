from io import BytesIO
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response

from snark.vlm_guts import VisionLanguageModel
from voice.tts_guts import TtsSynthesizer

# Lazy global for single-GPU usage
_vlm: Optional[VisionLanguageModel] = None
_tts = None  # Lazy-initialized TTS instance


def get_vlm() -> VisionLanguageModel:
    global _vlm
    if _vlm is None:
        # Single-threaded, single instance due to CUDA
        _vlm = VisionLanguageModel()
    return _vlm


def get_tts():
    """Lazy-load and return a TTS synthesizer. Returns an object with synthesize_wav(text)->bytes."""
    global _tts
    if _tts is not None:
        return _tts

    synth = TtsSynthesizer(model_name="tts_models/en/vctk/vits")
    _tts = synth
    return _tts


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm VLM: small text-only inference
    vlm = get_vlm()
    # Load system prompt from snark/prompt.txt if present
    try:
        prompt_path = Path(__file__).parent / "prompt.txt"
        if prompt_path.exists():
            vlm.system_prompt = prompt_path.read_text(encoding="utf-8").strip()
    except Exception:
        pass
    try:
        _ = vlm.infer("Hello", image_path=None, max_new_tokens=1)
    except Exception:
        pass

    # Warm TTS: synthesize a tiny wav
    try:
        _ = get_tts().synthesize_wav("Hello")
    except Exception:
        pass

    yield


app = FastAPI(title="VLM Server", version="0.1.0", lifespan=lifespan)

@app.post("/infer/text")
async def infer_text(
    prompt: str = Form(...),
    max_new_tokens: int = Form(200),
    response_format: str = Form("text"),  # "text" or "wav"
):
    vlm = get_vlm()
    result = vlm.infer(prompt, image_path=None, max_new_tokens=max_new_tokens)
    if response_format == "wav":
        wav_bytes = get_tts().synthesize_wav(result)
        return Response(content=wav_bytes, media_type="audio/wav")
    return JSONResponse({"result": result})


@app.post("/infer/image")
async def infer_image(
    prompt: str = Form(...),
    max_new_tokens: int = Form(200),
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
    response_format: str = Form("text"),
):
    if not image_url and not image_file:
        return JSONResponse({"error": "Provide image_url or image_file"}, status_code=400)

    image_path: Optional[str] = None
    if image_url:
        image_path = image_url
    else:
        # Save uploaded file to a temp path
        data = await image_file.read()
        filename = image_file.filename or "upload.bin"
        tmp_path = Path("/tmp") / filename
        tmp_path.write_bytes(data)
        image_path = str(tmp_path)

    vlm = get_vlm()
    result = vlm.infer(prompt, image_path=image_path, max_new_tokens=max_new_tokens)
    if response_format == "wav":
        wav_bytes = get_tts().synthesize_wav(result)
        return Response(content=wav_bytes, media_type="audio/wav")
    return JSONResponse({"result": result})


