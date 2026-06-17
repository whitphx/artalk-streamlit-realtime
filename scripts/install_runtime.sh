#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${ARTALK_STREAMLIT_PYTHON:-python}"

if [[ -z "${ARTALK_PACKAGE_DIR:-}" ]]; then
  cat >&2 <<'EOF'
ARTALK_PACKAGE_DIR is required.
Set it to the packaged ARTalk checkout, for example:
  ARTALK_PACKAGE_DIR=/path/to/ARTalk
EOF
  exit 2
fi

if [[ -z "${GAGAVATAR_PACKAGE_DIR:-}" ]]; then
  cat >&2 <<'EOF'
GAGAVATAR_PACKAGE_DIR is required.
Set it to the packaged GAGAvatar checkout, for example:
  GAGAVATAR_PACKAGE_DIR=/path/to/GAGAvatar
EOF
  exit 2
fi

uv pip install --python "$PYTHON_BIN" -e "$ARTALK_PACKAGE_DIR" --no-deps
uv pip install --python "$PYTHON_BIN" -e "$GAGAVATAR_PACKAGE_DIR" --no-deps
uv pip install --python "$PYTHON_BIN" av numpy openai streamlit streamlit-webrtc streamlit-remote
uv pip install --python "$PYTHON_BIN" -e . --no-deps
