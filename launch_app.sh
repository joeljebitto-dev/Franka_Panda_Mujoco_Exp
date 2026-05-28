#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
RELOAD="${RELOAD:-0}"
BUILD_FRONTEND="${BUILD_FRONTEND:-auto}"
PYTHON_BIN="${PYTHON_BIN:-}"

usage() {
  cat <<USAGE
Launch the Franka Panda MuJoCo web dashboard.

Usage:
  ./launch_app.sh [--host HOST] [--port PORT] [--reload] [--no-build]

Environment:
  HOST=127.0.0.1          Backend bind address.
  PORT=8000               Backend port.
  RELOAD=1                Enable uvicorn reload.
  BUILD_FRONTEND=0        Skip frontend build.
  PYTHON_BIN=/path/python Override Python executable.

Open after launch:
  http://HOST:PORT
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host)
      HOST="${2:?Missing value for --host}"
      shift 2
      ;;
    --port)
      PORT="${2:?Missing value for --port}"
      shift 2
      ;;
    --reload)
      RELOAD=1
      shift
      ;;
    --no-build)
      BUILD_FRONTEND=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

cd "$ROOT_DIR"

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
  else
    echo "Python was not found. Create .venv or set PYTHON_BIN." >&2
    exit 1
  fi
fi

if [[ "$BUILD_FRONTEND" != "0" ]]; then
  if [[ ! -f "$ROOT_DIR/frontend/dist/index.html" || "$BUILD_FRONTEND" == "1" ]]; then
    if ! command -v npm >/dev/null 2>&1; then
      echo "frontend/dist is missing and npm was not found. Install npm or run with --no-build." >&2
      exit 1
    fi

    echo "Building frontend..."
    if [[ ! -d "$ROOT_DIR/frontend/node_modules" ]]; then
      npm --prefix "$ROOT_DIR/frontend" install
    fi
    npm --prefix "$ROOT_DIR/frontend" run build
  fi
fi

export PYTHONPATH="$ROOT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Starting Franka Panda dashboard at http://$HOST:$PORT"
if [[ "$RELOAD" == "1" ]]; then
  exec "$PYTHON_BIN" -m franka_mujoco.app --host "$HOST" --port "$PORT" --reload
fi

exec "$PYTHON_BIN" -m franka_mujoco.app --host "$HOST" --port "$PORT"
