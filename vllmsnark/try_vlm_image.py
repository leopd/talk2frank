import argparse
import sys
from pathlib import Path

from snark.vlm_guts import VisionLanguageModel


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
        "--temperature",
        type=float,
        default=1.1,
        help="Sampling temperature (default: 1.1)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p (default: 0.95)",
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
    result = vlm.infer(
        prompt,
        image_path=image_path,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=True,
    )
    
    print("\n--- Output ---")
    print(result)
    print("\nDone.")


if __name__ == "__main__":
    main()
