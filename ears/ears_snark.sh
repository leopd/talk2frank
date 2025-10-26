#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${SNARK_SERVER_URL:-}" ]]; then
  echo "SNARK_SERVER_URL must be set to the base URL of the snark server (e.g., https://localhost:8123)" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

uv run -m ears.ears_snark "$@"
