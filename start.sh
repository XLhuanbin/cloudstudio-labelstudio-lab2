#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi
if [[ -x .venv/bin/python ]]; then
  python_bin=.venv/bin/python
else
  python_bin="${PYTHON_BIN:-python3}"
fi
exec "$python_bin" -u start.py "$@"
