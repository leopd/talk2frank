# Snark - the so-called "brains" of the operation

This works at 30tps!

```
VLLM_GPU_MEM_UTILIZATION=0.55 VLLM_TP_SIZE=2 CUDA_VISIBLE_DEVICES=1,3 VLM_SIZE=32B time uv run python benchmark_vlm.py --max-tokens 600
```


## Testing

```
uv run pytest
```

## Snark server

Start the FastAPI server (single-threaded GPU usage):

```
uv run uvicorn snark.server:app --host 0.0.0.0 --port 8123
```

It defaults to using the 7B model.  To use a different model, set the `VLM_SIZE` environment variable.
```
export VLM_SIZE="32B"
```

I think the other valid size is 72B, but you'll need multiple large GPUs to run it.


To use the 14B model, set the `VLM_SIZE` environment variable to `14B`.
```
export VLM_SIZE="14B"
uv run uvicorn snark.server:app --host 0.0.0.0 --port 8123
```
### Example requests

- Text-only inference:

```
curl -X POST http://$SNARK_SERVER_IP:8123/infer/text \
  -F 'prompt=What is the capital of France?' \
  -F 'max_new_tokens=50'
```

- Text-to-speech (WAV response):

```
curl -X POST http://$SNARK_SERVER_IP:8123/infer/text \
  -F 'prompt=Say hello to my little friend' \
  -F 'response_format=wav' \
  --output reply.wav
```

- Image via URL:

```
curl -X POST http://$SNARK_SERVER_IP:8123/infer/image \
  -F 'prompt=Describe this image' \
  -F 'image_url=https://example.com/sample.jpg' \
  -F 'max_new_tokens=50'
```

- Image via URL to WAV:

```
curl -X POST http://$SNARK_SERVER_IP:8123/infer/image \
  -F 'prompt=Describe this image' \
  -F 'image_url=https://example.com/sample.jpg' \
  -F 'response_format=wav' \
  --output reply.wav
```

- Image via upload:

```
curl -X POST http://$SNARK_SERVER_IP:8123/infer/image \
  -F 'prompt=Describe this image' \
  -F 'image_file=@snark/sample.jpeg' \
  -F 'max_new_tokens=50'
```

Notes:
- Set `response_format=wav` to receive an `audio/wav` response synthesized via TTS.
- The server lazily loads TTS on first WAV request; initial call may be slower.
- The model defaults to `Qwen/Qwen2.5-VL-7B-Instruct`.
- File uploads require `python-multipart` (already listed as a dependency).

## To flash-attn or NOT

It's a trade-off.  Flash attention is faster, but it's not always available.  And can be difficult to install.
If you don't want it, comment it out of the `pyproject.toml` file.

If you do want it, you likely will need to rebuild it, which can take
hours if you're not careful, or on a slow machine.  To get it to go faster,
make sure you use all the cores, and don't build for CUDA architectures you 
don't need.

```
uv pip install ninja cmake
export USE_NINJA=1  # without ninja, it'll only build on 1 or 2 cores, and take forever.
export MAX_JOBS=$(nproc)                 # torch C++/CUDA build
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export TORCH_CUDA_ARCH_LIST="8.0"  # 8.0 for A100; 8.6 for RTX 3090; 8.9 for RTX 4090
# This is still likely to take a long time and dominate your machine, so nice it.
time nice uv add flash-attn --no-build-isolation  # ~10 minutes on 48 cores for A100.
```


You can check if it's installed correctly by running:
```
uv run python -c "import flash_attn; print(flash_attn.__file__)"
```
