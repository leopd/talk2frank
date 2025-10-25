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

