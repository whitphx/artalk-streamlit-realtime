#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${ARTALK_STREAMLIT_PYTHON:-python}"
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
for arg in "$@"; do
  if [[ "$arg" == "--" ]]; then
    TARGET_ARGS=APP_ARGS
    continue
  fi
  if [[ "$TARGET_ARGS" == "ST_REMOTE_ARGS" ]]; then
    ST_REMOTE_ARGS+=("$arg")
  else
    APP_ARGS+=("$arg")
  fi
done

COMMAND=(
  "$ST_REMOTE_BIN"
  --host "${STREAMLIT_SERVER_ADDRESS:-0.0.0.0}"
  --port "${STREAMLIT_SERVER_PORT:-8501}"
  "${ST_REMOTE_ARGS[@]}"
  streamlit_app.py
)

if [[ "${#APP_ARGS[@]}" -gt 0 ]]; then
  COMMAND+=(-- -- "${APP_ARGS[@]}")
fi

exec "${COMMAND[@]}"
