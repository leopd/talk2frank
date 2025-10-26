#!/bin/bash

cd "$(dirname $0)"
uv run uvicorn snark.server:app --host 0.0.0.0 --port 8123

