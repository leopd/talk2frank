#!/usr/bin/env python3
import argparse
import traceback
import io
import os
import sys
import time
from typing import Optional, Dict, Any, Tuple, List

from framegrab import FrameGrabber  # type: ignore
from PIL import Image  # type: ignore
import imgcat  # type: ignore
import yaml  # type: ignore
import numpy as np  # type: ignore
import requests
import sounddevice as sd  # type: ignore
import soundfile as sf  # type: ignore
import torch  # type: ignore
from torchvision.models.detection import fasterrcnn_resnet50_fpn  # type: ignore
import torchvision.transforms as T  # type: ignore


MAX_AUDIO_LENGTH_SECONDS = 30

def clear_iterm_scrollback():
    sys.stdout.write("\033]1337;ClearScrollback\a")
    sys.stdout.flush()


class InsultCam:
    def __init__(
        self,
        base_url: str,
        prompt_file: str = "prompt-viz.txt",
        max_new_tokens: int = 200,
        temperature: float = 1.1,
        top_p: float = 0.95,
        config_path: str = "camera.yaml",
        preview: bool = True,
        request_timeout_s: float = 60.0,
        remote_speaker_url: Optional[str] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.prompt_file = prompt_file
        self.prompt = ""
        self.reload_prompt()
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.config_path = config_path
        self.preview = bool(preview)
        self.request_timeout_s = float(request_timeout_s)
        self.remote_speaker_url = remote_speaker_url.rstrip("/") if remote_speaker_url else None
        self._detector: Optional[Tuple[torch.nn.Module, str]] = None
        self.MIN_TIME_BETWEEN_MESSAGES_SECONDS = 10

    def reload_prompt(self):
        old_prompt = self.prompt
        self.prompt = open(self.prompt_file).read().strip()
        if old_prompt != self.prompt:
            print(f"[insultcam] new prompt loaded: {self.prompt}")

    def _get_grabber(self) -> FrameGrabber:
        cameras = FrameGrabber.from_yaml(self.config_path)
        return cameras[0]

    def _encode_frame_to_jpeg(self, frame: np.ndarray) -> bytes:
        if not isinstance(frame, np.ndarray):
            raise RuntimeError("framegrab returned a non-numpy frame")
        if frame.ndim == 2:
            mode = "L"
            img = Image.fromarray(frame, mode=mode)
        elif frame.ndim == 3 and frame.shape[2] == 3:
            # Image comes in as BGR.  Swap it to RGB.
            img = Image.fromarray(frame[:, :, ::-1])
        else:
            raise RuntimeError(f"Unexpected frame shape: {frame.shape}")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=90)
        return buf.getvalue()

    def _post_image_and_get_wav(self, jpeg_bytes: bytes) -> bytes:
        url = f"{self.base_url}/infer/image"
        files = {
            "image_file": ("frame.jpg", io.BytesIO(jpeg_bytes), "image/jpeg"),
        }
        data = {
            "prompt": self.prompt,
            "max_new_tokens": str(self.max_new_tokens),
            "temperature": str(self.temperature),
            "top_p": str(self.top_p),
            "response_format": "wav",
        }

        start_time = time.time()
        resp = requests.post(url, data=data, files=files, timeout=self.request_timeout_s)
        elapsed = time.time() - start_time
        if not resp.ok:
            details = resp.text[:256]
            ct = resp.headers.get("Content-Type", "")
            raise RuntimeError(
                f"HTTP {resp.status_code} from {url} (Content-Type: {ct}): {details}"
            )
        print(f"[snark] response took {elapsed:.2f} seconds")
        return resp.content

    def _play_wav_bytes(self, wav_bytes: bytes) -> None:
        wav_io = io.BytesIO(wav_bytes)
        audio_array, sample_rate = sf.read(wav_io, dtype="float32")
        if audio_array.ndim > 1:
            audio_array = audio_array[:, 0]
        audio_length_seconds = audio_array.shape[0] / sample_rate if sample_rate else 0.0
        if audio_length_seconds > MAX_AUDIO_LENGTH_SECONDS:
            audio_array = audio_array[: int(sample_rate * MAX_AUDIO_LENGTH_SECONDS)]
        print(f"[snark] playing audio {audio_length_seconds:.1f} seconds")
        sd.play(audio_array, samplerate=sample_rate)
        sd.wait()
        print("[snark] done playing audio")

    def _post_wav_to_remote_speaker(self, wav_bytes: bytes) -> None:
        if not self.remote_speaker_url:
            raise RuntimeError("remote_speaker_url not configured")
        url = f"{self.remote_speaker_url}/play/wav"
        files = {
            "wav_file": ("audio.wav", io.BytesIO(wav_bytes), "audio/wav"),
        }
        resp = requests.post(url, files=files, timeout=self.request_timeout_s)
        if not resp.ok:
            details = resp.text[:256]
            ct = resp.headers.get("Content-Type", "")
            raise RuntimeError(
                f"HTTP {resp.status_code} from {url} (Content-Type: {ct}): {details}"
            )

    def _play_or_send_wav(self, wav_bytes: bytes) -> None:
        if self.remote_speaker_url:
            self._post_wav_to_remote_speaker(wav_bytes)
            return
        self._play_wav_bytes(wav_bytes)

    def _maybe_preview(self, frame: np.ndarray) -> None:
        if not self.preview:
            return
        try:
            jpeg_bytes = self._encode_frame_to_jpeg(frame)
            imgcat.imgcat(jpeg_bytes)
        except Exception as e:
            print("Error previewing frame {e}", file=sys.stderr)

    def run_once(self) -> None:
        grabber = self._get_grabber()
        try:
            frame = grabber.grab()
        finally:
            # Best-effort cleanup
            try:
                if hasattr(grabber, "release"):
                    grabber.release()  # type: ignore[attr-defined]
                elif hasattr(grabber, "close"):
                    grabber.close()  # type: ignore[attr-defined]
            except Exception:
                print("Error releasing grabber", file=sys.stderr)
                traceback.print_exc()

        if frame is None:
            raise RuntimeError("Failed to capture a frame from the camera.")

        jpeg_bytes = self._encode_frame_to_jpeg(frame)
        self._maybe_preview(frame)
        wav_bytes = self._post_image_and_get_wav(jpeg_bytes)
        self._play_or_send_wav(wav_bytes)

    # --- Looping and detection ---
    def _load_detector(self) -> Tuple[torch.nn.Module, str]:
        if self._detector is not None:
            return self._detector
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = fasterrcnn_resnet50_fpn(weights="DEFAULT")  # type: ignore[arg-type]
        model.eval()
        model.to(device)
        self._detector = (model, device)
        return self._detector

    def _detect_people(self, frame: np.ndarray, score_thresh: float = 0.7) -> List[Tuple[np.ndarray, float]]:
        model, device = self._load_detector()
        # Convert numpy frame (H,W,C RGB uint8) to tensor
        pil = Image.fromarray(frame if frame.dtype == np.uint8 else frame.astype(np.uint8, copy=False))
        transform = T.Compose([T.ToTensor()])
        x = transform(pil).to(device)
        with torch.no_grad():
            outputs = model([x])
        out = outputs[0]
        boxes = out.get("boxes")
        labels = out.get("labels")
        scores = out.get("scores")
        if boxes is None or labels is None or scores is None:
            return []
        boxes = boxes.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        results = []
        for b, l, s in zip(boxes, labels, scores):
            if s < score_thresh:
                continue
            # COCO class 1 is person in torchvision (indexing starts at 1)
            if int(l) == 1:
                # box as (x1, y1, x2, y2)
                results.append((b, float(s)))
        return results

    def _is_centered(self, box: np.ndarray, width: int, height: int, edge_margin: int = 8, center_margin_frac: float = 0.15) -> bool:
        x1, y1, x2, y2 = [float(v) for v in box]
        # Reject if touches left/right edge within margin
        if x1 <= edge_margin or x2 >= (width - edge_margin):
            return False
        # Center check: center x in middle band
        cx = 0.5 * (x1 + x2)
        band_left = width * (0.5 - center_margin_frac)
        band_right = width * (0.5 + center_margin_frac)
        return band_left <= cx <= band_right

    def _draw_boxes(self, frame: np.ndarray, boxes_and_scores: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        # Draw rectangles using Pillow to keep deps minimal
        try:
            img = Image.fromarray(frame.astype(np.uint8, copy=False))
            from PIL import ImageDraw  # type: ignore
            draw = ImageDraw.Draw(img)
            for box, score in boxes_and_scores:
                x1, y1, x2, y2 = [float(v) for v in box]
                color = (0, 255, 0)
                draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
            return np.asarray(img)
        except Exception:
            return frame

    def run_loop(self) -> None:
        grabber = self._get_grabber()
        time_last_spoke = 0
        try:
            cnt = 0
            while True:
                try:
                    frame = grabber.grab()
                    if frame is None:
                        continue
                    cnt += 1
                    h, w = int(frame.shape[0]), int(frame.shape[1])
                    detections = self._detect_people(frame)
                    if not detections:
                        self._maybe_preview(frame)
                        print(f"[insultcam] no people detected")
                        continue
                    num_people = len(detections)
                    centered = [d for d in detections if self._is_centered(d[0], w, h)]
                    if self.preview:
                        if cnt % 1000 == 0:
                            # Avoid blowing up memory with lots of images in the scrollback buffer
                            # Ask me why I thought of this.
                            clear_iterm_scrollback()
                        boxed = self._draw_boxes(frame, detections)
                        jpeg_bytes = self._encode_frame_to_jpeg(boxed)
                        imgcat.imgcat(jpeg_bytes)

                    if not centered:
                        # People detected but not centered; keep looping
                        print(f"[insultcam] {num_people} people detected but not centered")
                        continue
                    print(f"[insultcam] {num_people} people detected and centered - sending to snark server")
                    # Found a centered person.  maybe time to play a message.

                    if time.time() - time_last_spoke < self.MIN_TIME_BETWEEN_MESSAGES_SECONDS:
                        print(f"[insultcam] too soon to speak again")
                        continue

                    self.reload_prompt()
                    jpeg_bytes = self._encode_frame_to_jpeg(frame)
                    wav_bytes = self._post_image_and_get_wav(jpeg_bytes)
                    self._play_or_send_wav(wav_bytes)
                    time_last_spoke = time.time()
                except Exception as e:
                    traceback.print_exc()
                    print(f"[insultcam] error: {e}.  Sleeping 5 seconds to recover.", file=sys.stderr)
                    time.sleep(5)
                    continue
        finally:
            try:
                if hasattr(grabber, "release"):
                    grabber.release()  # type: ignore[attr-defined]
                elif hasattr(grabber, "close"):
                    grabber.close()  # type: ignore[attr-defined]
            except Exception:
                print("Error releasing grabber", file=sys.stderr)
                traceback.print_exc()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture one frame using camera.yaml, send to snark /infer/image, play WAV response."
        )
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompt-viz.txt",
        help="Path to file containing standard prompt (default: prompt-viz.txt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="camera.yaml",
        help="Path to framegrab camera YAML config (default: camera.yaml)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=100,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--preview",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display the captured image using imgcat (default: on)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Request timeout (seconds)",
    )
    parser.add_argument(
        "--remote-speaker-url",
        type=str,
        default=None,
        help="If set, send WAV to remote speaker service instead of local playback",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Enable continuous loop with person detection before sending",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    snark_url = os.environ.get("SNARK_SERVER_URL")
    if not snark_url:
        print(
            "SNARK_SERVER_URL must be set to the base URL of the snark server, e.g. https://snark.example.com",
            file=sys.stderr,
        )
        return 2

    app = InsultCam(
        base_url=snark_url,
        prompt_file=args.prompt_file,
        config_path=args.config,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        preview=args.preview,
        request_timeout_s=args.timeout,
        remote_speaker_url=args.remote_speaker_url,
    )
    if args.loop:
        app.run_loop()
    else:
        app.run_once()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
