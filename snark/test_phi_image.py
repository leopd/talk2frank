from try_phi_image import VisionLanguageModel


def test_vlm_inference_with_sample_image():
    """Test that VLM can process sample.jpeg and return non-empty output."""
    vlm = VisionLanguageModel()
    
    prompt = "Describe what you see in this image."
    result = vlm.infer("sample.jpeg", prompt, max_new_tokens=50)
    
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0
    assert len(result.strip()) > 0

