#!/usr/bin/env bash
# Build the CPU-only pytorch3d wheel that `provision_track_env.sh --cpu`
# installs, against the torch pin in envs/track-requirements.txt. Rebuild it
# whenever that pin moves: the extension is ABI-tied to torch.
#   scripts/build_pytorch3d_cpu_wheel.sh <scratch-dir>
# The wheel links against the build host's glibc, so build on a host whose
# glibc is no newer than the image's (Ubuntu 24.04, glibc 2.39).
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"
SCRATCH="${1:?usage: $0 <scratch-dir>}"
PYTORCH3D_REV=v0.7.9

pin() { grep -E "^$1==" envs/track-requirements.txt | sed 's|+cu118$|+cpu|'; }

uv venv --clear --python 3.12 "$SCRATCH/venv"
uv pip install --python "$SCRATCH/venv/bin/python" \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "$(pin torch)" "$(pin setuptools)" "$(pin wheel)" "$(pin numpy)" ninja

rm -rf "$SCRATCH/pytorch3d"
git clone --depth 1 --branch "$PYTORCH3D_REV" \
  https://github.com/facebookresearch/pytorch3d.git "$SCRATCH/pytorch3d"

rm -rf "$REPO_ROOT/wheels/track-cpu"
# pytorch3d's setup.py compiles its CUDA kernels when torch sees a GPU or
# FORCE_CUDA=1; the CPU torch build sees none, and this rules out the latter.
# The interpreter's own build flags include -g, and the debug info would
# otherwise make up nearly all of the wheel. The prefix maps keep the build
# host's paths out of the strings the compiler embeds.
PREFIX_MAP="-ffile-prefix-map=$SCRATCH=/build -fmacro-prefix-map=$SCRATCH=/build"
FORCE_CUDA=0 LDFLAGS="-Wl,--strip-debug" CFLAGS="$PREFIX_MAP" CXXFLAGS="$PREFIX_MAP" uv build --wheel --no-build-isolation \
  --python "$SCRATCH/venv/bin/python" \
  --out-dir "$REPO_ROOT/wheels/track-cpu" "$SCRATCH/pytorch3d"
# uv build marks its output directory ignored, which would keep the wheel
# out of the commit.
rm "$REPO_ROOT/wheels/track-cpu/.gitignore"
