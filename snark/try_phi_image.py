import argparse
import sys
from pathlib import Path

import torch
import PIL.Image as Image
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor


class VisionLanguageModel:
    """Wrapper for vision-language models for image understanding tasks."""

    def __init__(self, model_name: str = "microsoft/Phi-3.5-vision-instruct"):
        """Load the vision-language model and processor."""
        print(f"Loading model {model_name}...")
        
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config._attn_implementation = "eager"
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # disable flash attention
        )
        print(f"Model loaded on device: {self.model.device}")

    def infer(self, prompt: str, image_path: str = None, max_new_tokens: int = 200) -> str:
        """
        Run inference with text prompt and optional image.
        
        Args:
            prompt: Text prompt for the model
            image_path: Optional path to the input image
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response from the model
        """
        if image_path:
            img = Image.open(image_path)
            print(f"Loaded image resolution: {img.size}")
            
            if "<|image_1|>" not in prompt:
                prompt = f"<|image_1|>\n{prompt}"
            
            print(f"Preprocessing...")
            inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.model.device)
        else:
            print(f"Text-only mode (no image)")
            print(f"Preprocessing...")
            inputs = self.processor(text=prompt, return_tensors="pt").to(self.model.device)
        
        print(f"Generating output...")
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens, use_cache=False)
        
        if hasattr(output_ids, "sequences"):
            output_ids = output_ids.sequences
        
        output_ids = output_ids.cpu()
        output_ids[output_ids == -1] = 0
        
        result = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return result


def main():
    parser = argparse.ArgumentParser(description="Run Phi-3.5 Vision model inference")
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
        default="microsoft/Phi-3.5-vision-instruct",
        help="Model name or path (default: microsoft/Phi-3.5-vision-instruct)",
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
    vlm = VisionLanguageModel(model_name=args.model)
    result = vlm.infer(prompt, image_path=image_path, max_new_tokens=args.max_tokens)
    
    print("\n--- Output ---")
    print(result)
    print("\nDone.")


if __name__ == "__main__":
    main()
