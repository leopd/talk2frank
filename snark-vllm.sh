#!/bin/bash

cd "$(dirname $0)"
cd vllmsnark
uv run uvicorn server:app --host 0.0.0.0 --port 8123

