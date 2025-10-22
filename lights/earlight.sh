#!/bin/bash

uv sync
uv run ../ears/mic_stream_whisper.py --model small.en | uv run twinkler.py black

