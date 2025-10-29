from pathlib import Path
import re
import time
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

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", flash_attn: str = "auto"):
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
        # Report effective attention implementation and device mapping for transparency
        effective_attn = getattr(self.model, "attn_implementation", None)
        if not effective_attn:
            effective_attn = getattr(self.model.config, "attn_implementation", None)
        if not effective_attn:
            effective_attn = getattr(self.model.config, "_attn_implementation", None)
        print(f"Effective attention implementation: {effective_attn}")
        device_map = getattr(self.model, "hf_device_map", None)
        if device_map:
            print(f"Device map: {device_map}")
        try:
            from torch.backends.cuda import sdp_kernel  # type: ignore
            print(
                "SDPA enabled - flash:{}, mem_efficient:{}, math:{}".format(
                    sdp_kernel.is_flash_enabled(),
                    sdp_kernel.is_mem_efficient_enabled(),
                    sdp_kernel.is_math_enabled(),
                )
            )
        except Exception:
            print("SDPA not available")
        self.recent_responses = []
        self.system_prompt = ""
        self.MAX_RECENT_RESPONSES = 10
        self.MAX_WORDS_PER_RESPONSE = 50

    def infer(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        max_new_tokens: int = 200,
        temperature: float = 1.1,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """
        Run inference with text prompt and optional image.
        """
        # Build messages in Qwen2.5-VL format with optional system prompt
        start_time = time.time()
        messages = []
        if self.system_prompt:
            messages.append(
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": str(self.system_prompt)},
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
            print(f"VLM text input: {prompt}")
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            )

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

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_start = time.time()
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=float(temperature),
            top_p=float(top_p),
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        gen_elapsed = time.time() - gen_start

        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]

        # Track how many tokens were generated for downstream benchmarking
        try:
            self.last_generated_tokens = int(sum(len(g_ids) for g_ids in generated_ids))
        except Exception:
            self.last_generated_tokens = 0
        gen_tps = (self.last_generated_tokens / gen_elapsed) if gen_elapsed > 0 and self.last_generated_tokens > 0 else 0.0
        print(f"VLM generation: {self.last_generated_tokens} tok in {gen_elapsed:.2f}s = {gen_tps:.2f} tok/s")

        result = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        elapsed = time.time() - start_time
        print(f"VLM inference took {elapsed:.2f} seconds")
        output = self.postprocess(result)
        return output


    def postprocess(self, result: str) -> str:
        """
        Ad-hoc postprocessing of the result to remove things that shouldn't be there, but show up sometimes.
        """
        # Remove voice-directions inside asterisks like "*grumble* I'm not happy"
        result = re.sub(r'\*.*?\*', '', result)

        # For some reason Qwen outputs "addCriterion" sometimes, which is just nonsense.  Strip everything after that.
        result = re.sub(r'addCriterion.*', '', result)

        # Remove non-ascii characters.  Don't need it saying "chinese character"
        result = re.sub(r'[^\x00-\x7F]+', '', result)

        # Limit to MAX_WORDS_PER_RESPONSE words.
        words = result.split()
        if len(words) > self.MAX_WORDS_PER_RESPONSE:
            result = " ".join(words[:self.MAX_WORDS_PER_RESPONSE])
            print(f"Truncated response to {len(words)} words to {self.MAX_WORDS_PER_RESPONSE} words")

        # Null out duplicate responses.  (Common with stupid 7B model.)
        if result not in self.recent_responses:
            self.recent_responses.append(result)
            if len(self.recent_responses) > self.MAX_RECENT_RESPONSES:
                self.recent_responses.pop(0)
        else:
            print(f"Ignoring duplicate response: {result}")
            result = ""
        return result

