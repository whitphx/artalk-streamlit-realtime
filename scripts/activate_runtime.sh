#!/usr/bin/env bash

if [[ -n "${BASH_SOURCE[0]:-}" ]]; then
  script_path="${BASH_SOURCE[0]}"
elif [[ -n "${(%):-%x}" ]]; then
  script_path="${(%):-%x}"
else
  script_path="$0"
fi

repo_root="$(cd "$(dirname "$script_path")/.." && pwd)"
runtime_dir="${ARTALK_STREAMLIT_ENV_DIR:-$repo_root/.mamba-env}"

if [[ ! -x "$runtime_dir/bin/python" ]]; then
  cat >&2 <<EOF
Runtime environment was not found at:
  $runtime_dir

Set ARTALK_STREAMLIT_ENV_DIR to another environment, or create .mamba-env first.
EOF
  return 2 2>/dev/null || exit 2
fi

export PATH="$runtime_dir/bin:$PATH"
export ARTALK_STREAMLIT_PYTHON="${ARTALK_STREAMLIT_PYTHON:-$runtime_dir/bin/python}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-$repo_root/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$repo_root/.pip-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$repo_root/.uv-cache}"
