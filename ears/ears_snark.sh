#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SNARK_SERVER_URL:-}" ]]; then
  echo "SNARK_SERVER_URL must be set to the base URL of the snark server (e.g., https://snark.example.com)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run -m ears.ears_snark "$@"
