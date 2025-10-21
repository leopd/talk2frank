#!/bin/bash
# Setup faster-whisper with GPU on Ubuntu + CUDA, plus a live-mic demo.
# Usage: bash setup_faster_whisper.sh

set -e

cd "$(dirname "$0")"

set -euo pipefail

sudo apt-get update
sudo apt-get install -y \
    ffmpeg \
    libportaudio2 \
    libsndfile1 \
    portaudio19-dev

sudo apt-get install -y pipewire-audio-client-libraries pulseaudio-utils

systemctl --user start pipewire.service pipewire-pulse.service
systemctl --user status pipewire.service pipewire-pulse.service
systemctl --user enable --now pipewire.socket pipewire-pulse.socket wireplumber.service

sudo usermod -aG audio $USER

sudo apt install -y libcudnn9-cuda-13 libcudnn9-dev-cuda-13

