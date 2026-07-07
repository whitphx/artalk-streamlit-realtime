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
export UV_CACHE_DIR="${UV_CACHE_DIR:-$REPO_ROOT/.uv-cache}"

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

uv pip install --python "$PYTHON_BIN" --no-build-isolation -e "$ARTALK_PACKAGE_DIR" --no-deps
uv pip install --python "$PYTHON_BIN" --no-build-isolation -e "$GAGAVATAR_PACKAGE_DIR" --no-deps
uv pip install --python "$PYTHON_BIN" av numpy openai streamlit streamlit-webrtc streamlit-remote
uv pip install --python "$PYTHON_BIN" --no-build-isolation -e . --no-deps
