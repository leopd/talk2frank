import argparse
from pathlib import Path

import torch
import PIL.Image as Image
from transformers import AutoModelForCausalLM, AutoProcessor


class VisionLanguageModel:
    """Wrapper for vision-language models for image understanding tasks."""

    def __init__(self, model_name: str = "microsoft/Phi-3.5-vision-instruct"):
        """Load the vision-language model and processor."""
        print(f"Loading model {model_name}...")
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",  # disable flash attention
        )
        print(f"Model loaded on device: {self.model.device}")

    def infer(self, image_path: str, prompt: str, max_new_tokens: int = 200) -> str:
        """
        Run inference on an image with the given prompt.
        
        Args:
            image_path: Path to the input image
            prompt: Text prompt for the model
            max_new_tokens: Maximum number of tokens to generate
            
        Returns:
            Generated text response from the model
        """
        img = Image.open(image_path)
        print(f"Loaded image resolution: {img.size}")
        
        print(f"Preprocessing...")
        inputs = self.processor(images=img, text=prompt, return_tensors="pt").to(self.model.device)
        
        print(f"Generating output...")
        output_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        
        result = self.processor.batch_decode(output_ids, skip_special_tokens=True)[0]
        return result


def main():
    parser = argparse.ArgumentParser(description="Run Phi-3.5 Vision model inference on an image")
    parser.add_argument("image", type=str, help="Path to input image")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for the model (if not provided, reads from prompt.txt)",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default="prompt.txt",
        help="Path to file containing prompt text (default: prompt.txt)",
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
    
    # Determine prompt source
    if args.prompt:
        prompt = args.prompt
    else:
        prompt_path = Path(args.prompt_file)
        if not prompt_path.exists():
            parser.error(f"Prompt file not found: {args.prompt_file}")
        prompt = prompt_path.read_text().strip()
    
    print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
    
    # Load model and run inference
    vlm = VisionLanguageModel(model_name=args.model)
    result = vlm.infer(args.image, prompt, max_new_tokens=args.max_tokens)
    
    print("\n--- Output ---")
    print(result)
    print("\nDone.")


if __name__ == "__main__":
    main()
