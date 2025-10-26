from typing import Optional

import types
import tempfile

from fastapi.testclient import TestClient

from snark import server


class StubVLM:
    def infer(self, prompt: str, image_path: Optional[str] = None, max_new_tokens: int = 200) -> str:
        if image_path:
            return f"IMG:{prompt[:10]}:{max_new_tokens}"
        return f"TXT:{prompt[:10]}:{max_new_tokens}"


class StubTTS:
    def synthesize_wav(self, text: str) -> bytes:
        # Return a tiny valid WAV header with no data (RIFF minimal)
        return (
            b"RIFF" + (36).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little") + (1).to_bytes(2, "little")
            + (1).to_bytes(2, "little") + (16000).to_bytes(4, "little") + (32000).to_bytes(4, "little")
            + (2).to_bytes(2, "little") + (16).to_bytes(2, "little") + b"data" + (0).to_bytes(4, "little")
        )


def setup_function(_):
    # Patch the global getter to avoid loading real model
    server._vlm = StubVLM()  # type: ignore[assignment]
    server._tts = StubTTS()  # type: ignore[assignment]


def test_infer_text_endpoint():
    client = TestClient(server.app)
    resp = client.post("/infer/text", data={"prompt": "hello world", "max_new_tokens": "33"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"].startswith("TXT:hello worl:33")


def test_infer_image_url_endpoint():
    client = TestClient(server.app)
    resp = client.post(
        "/infer/image",
        data={"prompt": "describe", "max_new_tokens": "12", "image_url": "https://example.com/img.jpg"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"].startswith("IMG:describe:12")


def test_infer_image_upload_endpoint(tmp_path):
    client = TestClient(server.app)
    tmp_file = tmp_path / "x.jpg"
    tmp_file.write_bytes(b"fake")
    with tmp_file.open("rb") as f:
        resp = client.post(
            "/infer/image",
            data={"prompt": "upload", "max_new_tokens": "7"},
            files={"image_file": ("x.jpg", f, "image/jpeg")},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body["result"].startswith("IMG:upload:7")


def test_infer_image_missing_params():
    client = TestClient(server.app)
    resp = client.post("/infer/image", data={"prompt": "x"})
    assert resp.status_code == 400
    body = resp.json()
    assert "Provide image_url or image_file" in body["error"]


def test_infer_text_wav_response():
    client = TestClient(server.app)
    resp = client.post("/infer/text", data={"prompt": "hello", "response_format": "wav"})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content.startswith(b"RIFF")


def test_infer_image_wav_response():
    client = TestClient(server.app)
    resp = client.post(
        "/infer/image",
        data={"prompt": "describe", "image_url": "https://example.com/img.jpg", "response_format": "wav"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("audio/wav")
    assert resp.content.startswith(b"RIFF")


