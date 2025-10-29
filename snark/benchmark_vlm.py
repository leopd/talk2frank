import argparse
import os
import time

try:
    # Normal case: running as module (python -m snark.benchmark_vlm)
    from snark.vlm_guts import VisionLanguageModel
except ModuleNotFoundError:
    # Fallback: running as a plain script file; add repo root to sys.path
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from snark.vlm_guts import VisionLanguageModel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark VLM inference on a sample image")
    parser.add_argument(
        "--flash-attn",
        type=str,
        choices=["on", "off", "auto"],
        default="auto",
        help="Flash attention mode: on (require), off (disable), auto (use if available). Default: auto",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate (default: 200)",
    )
    return parser.parse_args()


def main():
    """Run a deterministic image caption with the VLM and report latency and output."""
    args = parse_args()

    size = os.environ.get("VLM_SIZE", "7B")
    model_name = f"Qwen/Qwen2.5-VL-{size}-Instruct"
    print(f"Model: {model_name}")
    print(f"Flash-attn: {args.flash_attn}")

    vlm = VisionLanguageModel(model_name=model_name, flash_attn=args.flash_attn)

    prompt = "Provide a detailed description of this image."
    image_path = "sample.jpeg"  # resolved relative to snark/ by vlm_guts

    start = time.time()
    result = vlm.infer(
        prompt,
        image_path=image_path,
        max_new_tokens=args.max_tokens,
        temperature=0.0,
        top_p=1.0,
        do_sample=False,
    )
    elapsed = time.time() - start

    print("\n--- Response ---")
    print(result)
    print("\n--- Benchmark ---")
    gen_tokens = getattr(vlm, "last_generated_tokens", 0)
    tps = (gen_tokens / elapsed) if elapsed > 0 and gen_tokens > 0 else 0.0
    print(f"Wall time: {elapsed:.2f} seconds")
    print(f"Generated tokens: {gen_tokens}")
    print(f"Tokens/sec: {tps:.2f}")


if __name__ == "__main__":
    main()


