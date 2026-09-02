#!/usr/bin/env bash
# Provision the GAGAvatar_track environment (avatar registration).
# Separate from the main env because its dependency set conflicts with the
# realtime runtime stack. Produces .envs/track and prints the env vars to use.
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
ENV_DIR="${TRACK_ENV_DIR:-$REPO_ROOT/.envs/track}"
ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-6.0;7.5;8.0+PTX}"

uv venv --python 3.12 "$ENV_DIR"
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
    "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.9"
fi

cat <<EOF
Track environment ready. Point the app at it:
  export GAGAVATAR_TRACK_PYTHON=$ENV_DIR/bin/python
  export GAGAVATAR_TRACK_DIR=<GAGAvatar checkout>/core/libs/GAGAvatar_track
The tracker's own model assets (~1.1 GB under its assets/) follow the
GAGAvatar_track README.
EOF
