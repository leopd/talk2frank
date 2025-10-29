#!/usr/bin/env python3
import argparse
import sys
import os
from typing import List

try:
    import sounddevice as sd  # type: ignore
except Exception:  # pragma: no cover - optional runtime dep
    sd = None  # type: ignore

# Import that works whether executed as a module (python -m voice.tryvoice)
# or as a script (python voice/tryvoice.py)
try:
    from .tts_guts import TtsSynthesizer  # type: ignore
except Exception:  # pragma: no cover - fallback for direct script execution
    sys.path.append(os.path.dirname(__file__))
    from tts_guts import TtsSynthesizer  # type: ignore

try:  # Optional import: only needed for model listing
    from TTS.api import TTS  # type: ignore
except Exception:  # pragma: no cover
    TTS = None  # type: ignore


def list_available_models() -> List[str]:
    if TTS is None:
        return []
    try:
        # Prefer classmethod if available
        if hasattr(TTS, "list_models"):
            return list(TTS.list_models())  # type: ignore[attr-defined]
        # Fallback to instance method if present
        t = TTS()
        if hasattr(t, "list_models"):
            return list(t.list_models())  # type: ignore[attr-defined]
    except Exception:
        return []
    return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simple TTS demo with post-processing and speaker/model listing"
    )
    parser.add_argument(
        "text",
        nargs="?",
        default="I am the monster hiding under your bed.",
        help="Text to speak (default: a short sample)",
    )
    parser.add_argument(
        "-m",
        "--model",
        dest="model_name",
        default="tts_models/en/vctk/vits",
        help="TTS model name",
    )
    parser.add_argument(
        "-s",
        "--speaker",
        dest="speaker",
        default=None,
        help="Speaker id/name (model-dependent). If omitted, a preferred speaker may be auto-selected",
    )
    parser.add_argument(
        "-p",
        "--pitch-down-steps",
        dest="pitch_down_steps",
        type=int,
        default=5,
        help="Semitone steps to shift pitch down (integer)",
    )
    parser.add_argument(
        "-r",
        "--output-sample-rate-factor",
        dest="output_sample_rate_factor",
        type=float,
        default=0.6,
        help="Factor to scale header sample rate (slows playback)",
    )
    parser.add_argument(
        "-g",
        "--distortion-gain",
        dest="distortion_gain",
        type=float,
        default=1.4,
        help="Gain before tanh distortion (higher is harsher)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )
    parser.add_argument(
        "--list-speakers",
        action="store_true",
        help="List speakers for the selected model and exit",
    )
    return parser.parse_args()


def cmd_list_models() -> int:
    models = list_available_models()
    if not models:
        print("No models found or TTS library unavailable.")
        return 1
    for name in models:
        print(name)
    return 0


def cmd_list_speakers(model_name: str, pitch_down_steps: int, distortion_gain: float, output_sample_rate_factor: float) -> int:
    try:
        synth = TtsSynthesizer(
            model_name=model_name,
            pitch_down_steps=pitch_down_steps,
            distortion_gain=distortion_gain,
            output_sample_rate_factor=output_sample_rate_factor,
        )
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"Failed to load model '{model_name}': {e}")
        return 1
    speakers = sorted(synth.speakers)
    if not speakers:
        print("Model does not expose speakers (single-speaker model or unsupported).")
        return 0
    for spk in speakers:
        print(spk)
    return 0


def cmd_speak(text: str, model_name: str, speaker: str | None, pitch_down_steps: int, distortion_gain: float, output_sample_rate_factor: float) -> int:
    if sd is None:
        print("sounddevice is not installed or not available; cannot play audio.")
        return 1
    try:
        synth = TtsSynthesizer(
            model_name=model_name,
            pitch_down_steps=pitch_down_steps,
            distortion_gain=distortion_gain,
            output_sample_rate_factor=output_sample_rate_factor,
        )
    except Exception as e:  # pragma: no cover - environment dependent
        print(f"Failed to load model '{model_name}': {e}")
        return 1

    audio_array, header_rate = synth.synthesize_array(text=text, speaker=speaker)
    sd.play(audio_array, samplerate=header_rate)
    sd.wait()
    return 0


def main() -> int:
    args = parse_args()
    if args.list_models:
        return cmd_list_models()
    if args.list_speakers:
        return cmd_list_speakers(
            model_name=args.model_name,
            pitch_down_steps=args.pitch_down_steps,
            distortion_gain=args.distortion_gain,
            output_sample_rate_factor=args.output_sample_rate_factor,
        )
    return cmd_speak(
        text=args.text,
        model_name=args.model_name,
        speaker=args.speaker,
        pitch_down_steps=args.pitch_down_steps,
        distortion_gain=args.distortion_gain,
        output_sample_rate_factor=args.output_sample_rate_factor,
    )


if __name__ == "__main__":
    raise SystemExit(main())

