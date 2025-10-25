import argparse
import sys
from pathlib import Path
from typing import Optional

import torch
import PIL.Image as Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

try:
    import flash_attn
    FLASH_ATTN_AVAILABLE = True
except ImportError:
    FLASH_ATTN_AVAILABLE = False


class VisionLanguageModel:
    """Wrapper for a vision-language model like Qwen2.5-VL."""

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct", flash_attn: str = "off"):
        """
        Load the Qwen2.5-VL model and processor.
        
        Args:
            model_name: Model name or path
            flash_attn: Flash attention mode - "on", "off", or "auto"
        """
        print(f"Loading model {model_name}...")
        
        # Determine attention implementation
        if flash_attn == "on":
            if not FLASH_ATTN_AVAILABLE:
                raise RuntimeError("Flash attention requested but not installed. Install with: uv add flash-attn --no-build-isolation")
            attn_impl = "flash_attention_2"
            print(f"Using Flash Attention 2")
        elif flash_attn == "off":
            attn_impl = "eager"
            print(f"Using standard attention (eager)")
        else:
            if FLASH_ATTN_AVAILABLE:
                attn_impl = "flash_attention_2"
                print(f"Flash Attention detected and enabled")
            else:
                attn_impl = "eager"
                print(f"Flash Attention not available, using standard attention (eager)")
        
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
        
        Args:
            prompt: Text prompt for the model
            image_path: Optional path to the input image
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response from the model
        """
        # Build messages in Qwen2.5-VL format
        if image_path:
            # Resolve relative image paths to this module's directory for test stability
            image_path_obj = Path(image_path)
            if not image_path_obj.is_absolute() and not image_path_obj.exists():
                candidate = Path(__file__).parent / image_path
                if candidate.exists():
                    image_path = str(candidate)
            print(f"Loading image: {image_path}")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        else:
            print(f"Text-only mode (no image)")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
        
        print(f"Preprocessing...")
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
        
        print(f"Generating output...")
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        
        result = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return result


def main():
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL vision model inference")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to input image (uses standard prompt from file)",
    )
    parser.add_argument(
        "--textin",
        action="store_true",
        help="Read prompt text from stdin (text-only mode, no image)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompt.txt",
        help="Path to file containing standard prompt (default: prompt.txt)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 200)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        help="Model name or path (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )
    parser.add_argument(
        "--flash-attn",
        type=str,
        choices=["on", "off", "auto"],
        default="auto",
        help="Flash attention mode: on (require), off (disable), auto (use if available). Default: auto",
    )
    
    args = parser.parse_args()
    
    # Validate mode: either --image OR --textin
    if args.image and args.textin:
        parser.error("Cannot use both --image and --textin. Choose one mode.")
    if not args.image and not args.textin:
        parser.error("Must provide either --image or --textin")
    
    # Determine mode and get prompt
    if args.image:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            parser.error(f"Prompt file not found: {args.prompt_file}")
        prompt = prompt_path.read_text().strip()
        image_path = args.image
        print(f"Mode: Image processing")
        print(f"Image: {args.image}")
    else:
        print(f"Enter your prompt:")
        prompt = sys.stdin.readline().strip()
        print(f"Got prompt: {prompt}")
        image_path = None
        print(f"Mode: Text-only")
    
    print(f"Prompt: {prompt[:100]}...")
    print()
    
    # Load model and run inference
    vlm = VisionLanguageModel(model_name=args.model, flash_attn=args.flash_attn)
    result = vlm.infer(prompt, image_path=image_path, max_new_tokens=args.max_tokens)
    
    print("\n--- Output ---")
    print(result)
    print("\nDone.")


if __name__ == "__main__":
    main()
