#!/usr/bin/env bash
# Container start for a Hugging Face Space: complete the asset tree with the
# license-gated FLAME file (ARTALK_GATED_ASSETS_REPO, read with the HF_TOKEN
# secret), then launch on the Space's port.
set -euo pipefail
cd "$(dirname "$0")/.."
pixi run artalk-demo assets
exec pixi run artalk-demo up --profile spaces
