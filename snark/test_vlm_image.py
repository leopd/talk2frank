import subprocess
import sys

from try_vlm_image import VisionLanguageModel


def test_vlm_image_mode_with_file_prompt():
    """Test VLM in image mode with standard prompt from file."""
    vlm = VisionLanguageModel()
    
    prompt = "Describe what you see in this image."
    result = vlm.infer(prompt, image_path="sample.jpeg", max_new_tokens=50)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result.strip()) > 0


def test_vlm_text_only_mode():
    """Test VLM in text-only mode (no image)."""
    vlm = VisionLanguageModel()
    
    prompt = "What is the capital of France?"
    result = vlm.infer(prompt, image_path=None, max_new_tokens=50)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result.strip()) > 0


def test_cli_image_mode():
    """Test CLI with --image (uses prompt file)."""
    result = subprocess.run(
        [
            sys.executable,
            "try_phi_image.py",
            "--image",
            "sample.jpeg",
            "--prompt-file",
            "prompt.txt",
            "--max-tokens",
            "50",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    
    assert result.returncode == 0
    assert "Mode: Image processing" in result.stdout
    assert "--- Output ---" in result.stdout
    output_lines = result.stdout.split("--- Output ---")[1].strip()
    assert len(output_lines) > 0


def test_cli_textin_mode():
    """Test CLI with --textin (stdin prompt, no image)."""
    prompt = "What is 2 + 2?"
    
    result = subprocess.run(
        [sys.executable, "try_phi_image.py", "--textin", "--max-tokens", "50"],
        input=prompt,
        text=True,
        capture_output=True,
        timeout=60,
    )
    
    assert result.returncode == 0
    assert "Mode: Text-only" in result.stdout
    assert "--- Output ---" in result.stdout
    output_lines = result.stdout.split("--- Output ---")[1].strip()
    assert len(output_lines) > 0

