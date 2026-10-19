#!/usr/bin/env bash
# Provision the GAGAvatar_track environment (avatar registration).
# Separate from the main env because its dependency set conflicts with the
# realtime runtime stack. Produces .envs/track and prints the env vars to use.
#   scripts/provision_track_env.sh [--cpu]
# --cpu installs CPU-only torch and the prebuilt CPU pytorch3d wheel, for
# hosts that track on CPU and have no CUDA toolkit (the Space image).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
ENV_DIR="${TRACK_ENV_DIR:-$REPO_ROOT/.envs/track}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.0;7.5;8.0+PTX}"
# The PyTorch index also serves common packages such as certifi, at older
# versions, and by default uv takes each package from the first index that
# has it at all, so most of the pins would not resolve.
export UV_INDEX_STRATEGY=unsafe-best-match

CPU=0
for arg in "$@"; do
  case "$arg" in
    --cpu) CPU=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

uv venv --python 3.12 "$ENV_DIR"

if [[ "$CPU" == 1 ]]; then
  # Derived from the CUDA pins rather than kept as a second list, so the two
  # environments cannot drift apart. The CPU wheels leave out the CUDA
  # libraries, several GB the image would carry without using.
  sed -e 's|/whl/cu118$|/whl/cpu|' -e 's|+cu118$|+cpu|' \
      -e '/^nvidia-/d' -e '/^triton==/d' \
      envs/track-requirements.txt >"$ENV_DIR/requirements-cpu.txt"
  uv pip install --python "$ENV_DIR/bin/python" -r "$ENV_DIR/requirements-cpu.txt"
  uv pip install --python "$ENV_DIR/bin/python" wheels/track-cpu/pytorch3d-*.whl
else
  uv pip install --python "$ENV_DIR/bin/python" -r envs/track-requirements.txt

  # pytorch3d has no binary for this torch/CUDA pair; build it fat so one shared
  # env works on every deployment GPU (a single-arch build fails with "no kernel
  # image" or, worse, silently corrupt output on the other hosts).
  PREBUILT=$(ls wheels/pytorch3d-*.whl 2>/dev/null | head -1 || true)
  if [[ -n "$PREBUILT" ]]; then
    uv pip install --python "$ENV_DIR/bin/python" "$PREBUILT"
  else
    echo "building pytorch3d 0.7.9 from source (fat: $ARCH_LIST); this takes a while"
    FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="$ARCH_LIST" \
      uv pip install --python "$ENV_DIR/bin/python" --no-build-isolation \
      "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.9"
  fi
fi

cat <<EOF
Track environment ready. Point the app at it:
  export GAGAVATAR_TRACK_PYTHON=$ENV_DIR/bin/python
  export GAGAVATAR_TRACK_DIR=<GAGAvatar checkout>/core/libs/GAGAvatar_track
The tracker's own model assets (~1.1 GB under its assets/) follow the
GAGAvatar_track README.
EOF
