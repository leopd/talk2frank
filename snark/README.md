# Snark - the so-called "brains" of the operation


## Testing

```
uv run pytest
```

## Installing flash-attn

This can be annoying.  

```
uv pip install ninja cmake
export USE_NINJA=1
export MAX_JOBS=$(nproc)                 # torch C++/CUDA build
export CMAKE_BUILD_PARALLEL_LEVEL=$(nproc)
export TORCH_CUDA_ARCH_LIST="8.0"  # 8.0 for A100; 8.6 for RTX 3090; 8.9 for RTX 4090
time uv add flash-attn --no-build-isolation
```
