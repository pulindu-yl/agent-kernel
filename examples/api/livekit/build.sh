#!/bin/bash

set -euo pipefail
if command -v pyenv >/dev/null 2>&1; then
  uv venv --python "$(pyenv which python)" --allow-existing
else
  uv venv --allow-existing
fi

uv sync --all-extras
