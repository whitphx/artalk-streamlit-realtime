#!/usr/bin/env bash
# Fetch the model weights the vendored GAGAvatar_track loads, except FLAME,
# into the tracker's assets/ (or $TRACK_ASSETS_DIR, linked from there) and the
# tracker env's download caches.
# FLAME is license-gated and never baked into an image: the tracker reads the
# app's own copy through a symlink, which resolves once
# `artalk-demo assets` has placed FLAME_with_eye.pt in the asset tree.
#   scripts/fetch_track_resources.sh   (after scripts/provision_track_env.sh)
set -euo pipefail

cd "$(dirname "$0")/.."
TRACK_DIR="${GAGAVATAR_TRACK_DIR:-vendor/GAGAvatar/gagavatar/libs/GAGAvatar_track}"
# The tracker loads its weights from its own assets/ only, so weights kept
# elsewhere (on a larger disk) are linked in there.
TRACK_ASSETS_DIR="${TRACK_ASSETS_DIR:-$TRACK_DIR/assets}"
TRACK_PYTHON="${TRACK_ENV_DIR:-.envs/track}/bin/python"
ASSET_DIR="${ARTALK_ASSET_DIR:-assets}"

# The archive that the tracker's build_resources.sh downloads, pinned.
RESOURCES_URL=https://huggingface.co/xg-chu/GAGAvatar_track/resolve/130e31731625d6ac98cd11802e71c77d617688d9/track_resources.tar
RESOURCES_SHA256=06489fe16ed3ab6c0546b5533c0016e261a694bde335500dddc7243e6bf29672

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
curl -fsSL -o "$tmp/track_resources.tar" "$RESOURCES_URL"
echo "$RESOURCES_SHA256  $tmp/track_resources.tar" | sha256sum -c -
# flame/ holds FLAME_with_eye.pt, byte-identical to the app's gated copy, and
# canonical.obj, which nothing in the tracker reads.
mkdir -p "$TRACK_ASSETS_DIR"
tar -xf "$tmp/track_resources.tar" -C "$TRACK_ASSETS_DIR" --exclude=flame
if [ "$(realpath "$TRACK_ASSETS_DIR")" != "$(realpath "$TRACK_DIR/assets")" ]; then
  # File by file, not one link per directory: the tracker checkout ignores
  # the weight files themselves, but would list directory links as untracked.
  cp -rs --remove-destination "$(realpath "$TRACK_ASSETS_DIR")/." "$TRACK_DIR/assets/"
fi
mkdir -p "$TRACK_DIR/assets/flame"
ln -sfr "$ASSET_DIR/FLAME_with_eye.pt" "$TRACK_DIR/assets/flame/FLAME_with_eye.pt"

# The face detector and the matting backbone download their weights on first
# use, which would otherwise happen during a visitor's first registration.
# Constructing the detector also compiles it, a cost scripts/track_avatar.py
# opts out of too.
TORCHDYNAMO_DISABLE=1 "$TRACK_PYTHON" - <<'EOF'
import face_alignment
from transformers import Mask2FormerForUniversalSegmentation

face_alignment.FaceAlignment(
    face_alignment.LandmarksType.TWO_D, device="cpu", face_detector="sfd"
)
Mask2FormerForUniversalSegmentation.from_pretrained(
    "facebook/mask2former-swin-tiny-coco-instance"
)
EOF
