#!/bin/bash

set -e

cd "$(dirname "$0")"

sudo apt-get install -y build-essential cmake git wget
if [ ! -d "whisper-cpp" ]; then
    git clone https://github.com/ggml-org/whisper.cpp whisper-cpp
fi
cd whisper-cpp

cmake -B build -DGGML_CUDA=1 && cmake --build build -j  # drop -DGGML_CUDA=1 if no CUDA
# get a model (good+fast): 
bash ./models/download-ggml-model.sh base.en

