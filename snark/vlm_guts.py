from pathlib import Path
from typing import Optional

import torch
import PIL.Image as Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    import flash_attn  # noqa: F401
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class VisionLanguageModel:
    """Wrapper for a vision-language model like Qwen2.5-VL."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", flash_attn: str = "off"):
        """
        Load the Qwen2.5-VL model and processor.
        """
        print(f"Loading model {model_name}...")

        # Determine attention implementation
        if flash_attn == "on":
            if not FLASH_ATTN_AVAILABLE:
                raise RuntimeError(
                    "Flash attention requested but not installed. Install with: uv add flash-attn --no-build-isolation"
                )
            attn_impl = "flash_attention_2"
            print("Using Flash Attention 2")
        elif flash_attn == "off":
            attn_impl = "eager"
            print("Using standard attention (eager)")
        else:
            if FLASH_ATTN_AVAILABLE:
                attn_impl = "flash_attention_2"
                print("Flash Attention detected and enabled")
            else:
                attn_impl = "eager"
                print("Flash Attention not available, using standard attention (eager)")

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_impl,
        )
        print(f"Model loaded on device: {self.model.device}")

    def infer(self, prompt: str, image_path: Optional[str] = None, max_new_tokens: int = 200) -> str:
        """
        Run inference with text prompt and optional image.
        """
        # Build messages in Qwen2.5-VL format with optional system prompt
        messages = []
        system_prompt = getattr(self, "system_prompt", None)
        if system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": str(system_prompt)},
                    ],
                }
            )
        if image_path:
            # Resolve relative image paths to this module's directory for test stability
            image_path_obj = Path(image_path)
            if not image_path_obj.is_absolute() and not image_path_obj.exists():
                candidate = Path(__file__).parent / image_path
                if candidate.exists():
                    image_path = str(candidate)
            print(f"Loading image: {image_path}")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            )
        else:
            print("Text-only mode (no image)")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            )

        print("Preprocessing...")
        text_prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        vision_info = process_vision_info(messages)
        if isinstance(vision_info, tuple) and len(vision_info) == 3:
            image_inputs, video_inputs, audio_inputs = vision_info
        else:
            image_inputs, video_inputs = vision_info  # type: ignore[misc]
            audio_inputs = None

        processor_kwargs = {
            "text": [text_prompt],
            "images": image_inputs,
            "videos": video_inputs,
            "padding": True,
            "return_tensors": "pt",
        }
        if audio_inputs is not None:
            processor_kwargs["audios"] = audio_inputs
        inputs = self.processor(**processor_kwargs).to(self.model.device)

        print("Generating output...")
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        result = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return result


