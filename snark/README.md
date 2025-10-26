# Snark - the so-called "brains" of the operation


## Testing

```
uv run pytest
```

## Server

Start the FastAPI server (single-threaded GPU usage):

```
uv run uvicorn snark.server:app --host 0.0.0.0 --port 8000
```

### Example requests

- Text-only inference:

```
curl -X POST http://<server-ip>:8000/infer/text \
  -F 'prompt=What is the capital of France?' \
  -F 'max_new_tokens=50'
```

- Image via URL:

```
curl -X POST http://<server-ip>:8000/infer/image \
  -F 'prompt=Describe this image' \
  -F 'image_url=https://example.com/sample.jpg' \
  -F 'max_new_tokens=50'
```

- Image via upload:

```
curl -X POST http://<server-ip>:8000/infer/image \
  -F 'prompt=Describe this image' \
  -F 'image_file=@snark/sample.jpeg' \
  -F 'max_new_tokens=50'
```

Notes:
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
export USE_NINJA=1
export MAX_JOBS=$(nproc)                 # torch C++/CUDA build
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export TORCH_CUDA_ARCH_LIST="8.0"  # 8.0 for A100; 8.6 for RTX 3090; 8.9 for RTX 4090
time uv add flash-attn --no-build-isolation  # ~10 minutes on 48 cores for A100.
```
