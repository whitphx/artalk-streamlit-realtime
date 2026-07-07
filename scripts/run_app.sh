#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REPO_ROOT="$PWD"
if [[ -z "${ARTALK_STREAMLIT_PYTHON:-}" && -x "$REPO_ROOT/.mamba-env/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.mamba-env/bin/python"
else
  PYTHON_BIN="${ARTALK_STREAMLIT_PYTHON:-python}"
fi
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/huggingface}"

if [[ -z "${ST_REMOTE_BIN:-}" ]]; then
  PYTHON_PATH="$(command -v "$PYTHON_BIN" || true)"
  if [[ -n "$PYTHON_PATH" && -x "$(dirname "$PYTHON_PATH")/st-remote" ]]; then
    ST_REMOTE_BIN="$(dirname "$PYTHON_PATH")/st-remote"
  else
    ST_REMOTE_BIN="st-remote"
  fi
fi

if [[ -n "${ARTALK_ASSET_DIR:-}" ]]; then
  export GAGAVATAR_MODEL_PATH="${GAGAVATAR_MODEL_PATH:-$ARTALK_ASSET_DIR/GAGAvatar/GAGAvatar.pt}"
  export GAGAVATAR_TRACKED_PATH="${GAGAVATAR_TRACKED_PATH:-$ARTALK_ASSET_DIR/GAGAvatar/tracked.pt}"
  export GAGAVATAR_FLAME_MODEL_PATH="${GAGAVATAR_FLAME_MODEL_PATH:-$ARTALK_ASSET_DIR/FLAME_with_eye.pt}"
fi

ST_REMOTE_ARGS=()
APP_ARGS=()
TARGET_ARGS=ST_REMOTE_ARGS
HAS_PROVIDER_ARG=0
HAS_NO_REMOTE_ARG=0
for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    TARGET_ARGS=APP_ARGS
    continue
  fi
  if [[ "$TARGET_ARGS" == "ST_REMOTE_ARGS" ]]; then
    case "$arg" in
      --provider|--provider=*) HAS_PROVIDER_ARG=1 ;;
      --no-remote) HAS_NO_REMOTE_ARG=1 ;;
    esac
    ST_REMOTE_ARGS+=("$arg")
  else
    APP_ARGS+=("$arg")
  fi
done

# Default to an HTTPS tunnel provider: browser microphone access needs a
# secure context, and st-remote's own default is "first available".
PROVIDER_ARGS=()
if [[ "$HAS_PROVIDER_ARG" -eq 0 && "$HAS_NO_REMOTE_ARG" -eq 0 ]]; then
  PROVIDER_ARGS=(--provider "${ST_REMOTE_PROVIDER:-ngrok}")
fi

COMMAND=(
  "$ST_REMOTE_BIN"
  --host "${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
  --port "${STREAMLIT_SERVER_PORT:-8501}"
  "${PROVIDER_ARGS[@]}"
  "${ST_REMOTE_ARGS[@]}"
  streamlit_app.py
)

if [[ "${#APP_ARGS[@]}" -gt 0 ]]; then
  COMMAND+=(-- -- "${APP_ARGS[@]}")
fi

exec "${COMMAND[@]}"
