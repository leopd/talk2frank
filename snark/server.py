from pathlib import Path
from typing import Optional

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse

from vlm_guts import VisionLanguageModel


app = FastAPI(title="VLM Server", version="0.1.0")

# Lazy global for single-GPU usage
_vlm: Optional[VisionLanguageModel] = None


def get_vlm() -> VisionLanguageModel:
    global _vlm
    if _vlm is None:
        # Single-threaded, single instance due to CUDA
        _vlm = VisionLanguageModel()
    return _vlm


@app.post("/infer/text")
async def infer_text(prompt: str = Form(...), max_new_tokens: int = Form(200)) -> JSONResponse:
    vlm = get_vlm()
    result = vlm.infer(prompt, image_path=None, max_new_tokens=max_new_tokens)
    return JSONResponse({"result": result})


@app.post("/infer/image")
async def infer_image(
    prompt: str = Form(...),
    max_new_tokens: int = Form(200),
    image_url: Optional[str] = Form(None),
    image_file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    if not image_url and not image_file:
        return JSONResponse({"error": "Provide image_url or image_file"}, status_code=400)

    image_path: Optional[str] = None
    if image_url:
        image_path = image_url
    else:
        # Save uploaded file to a temp path
        data = await image_file.read()
        tmp_path = Path("/tmp") / image_file.filename
        tmp_path.write_bytes(data)
        image_path = str(tmp_path)

    vlm = get_vlm()
    result = vlm.infer(prompt, image_path=image_path, max_new_tokens=max_new_tokens)
    return JSONResponse({"result": result})


