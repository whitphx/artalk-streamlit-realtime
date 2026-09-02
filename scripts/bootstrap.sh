#!/usr/bin/env bash
# Bring a fresh Linux CUDA host to a runnable state:
#   scripts/bootstrap.sh [--with-track]
# then: pixi run artalk-demo assets && pixi run artalk-demo up
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# Pinned revisions of the checkouts the app consumes as directories (the
# artalk/gagavatar Python packages themselves are pinned in pixi.toml).
FALLINGWATER_REPO=https://github.com/whitphx/Fallingwater.git
FALLINGWATER_REV=8fbde2eb99f5d7fd15fc0c74430c734cd0a9bbfd
# train_code for --motion-model artalk1s; same repo as the artalk package.
ARTALK_TRAIN_REPO=https://github.com/whitphx/ARTalk.git
ARTALK_TRAIN_REV=5126a307871c8ef100009423955a86763e6844d4
# Carries the vendored GAGAvatar_track (gagavatar/libs/GAGAvatar_track).
GAGAVATAR_REPO=https://github.com/whitphx/GAGAvatar.git
GAGAVATAR_REV=437dadea35d3069abdcc2026f661869fd608231d

WITH_TRACK=0
for arg in "$@"; do
  case "$arg" in
    --with-track) WITH_TRACK=1 ;;
    *) echo "unknown argument: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v pixi >/dev/null && [[ ! -x "$HOME/.pixi/bin/pixi" ]]; then
  echo "installing pixi into ~/.pixi"
  curl -fsSL https://pixi.sh/install.sh | bash
fi
PIXI="$(command -v pixi || echo "$HOME/.pixi/bin/pixi")"

"$PIXI" install --locked

fetch_rev() {
  local url="$1" rev="$2" dest="$3" submodules="${4:-0}"
  if [[ -e "$dest/.git" ]]; then
    if [[ "$(git -C "$dest" rev-parse HEAD)" == "$rev" ]]; then
      echo "$dest already at $rev"
      return
    fi
  else
    mkdir -p "$dest"
    git -C "$dest" init -q
    git -C "$dest" remote add origin "$url"
  fi
  git -C "$dest" fetch --depth 1 origin "$rev"
  git -C "$dest" checkout -q FETCH_HEAD
  if [[ "$submodules" == 1 ]]; then
    git -C "$dest" submodule update --init --depth 1
  fi
}

mkdir -p vendor
fetch_rev "$FALLINGWATER_REPO" "$FALLINGWATER_REV" vendor/Fallingwater
fetch_rev "$ARTALK_TRAIN_REPO" "$ARTALK_TRAIN_REV" vendor/ARTalk
fetch_rev "$GAGAVATAR_REPO" "$GAGAVATAR_REV" vendor/GAGAvatar 1

if [[ "$WITH_TRACK" == 1 ]]; then
  scripts/provision_track_env.sh
fi

cat <<EOF

Bootstrap done. Next:
  1. $PIXI run artalk-demo assets        # downloads model weights (FLAME stays manual)
  2. $PIXI run artalk-demo doctor        # validates the install
  3. $PIXI run artalk-demo up            # launches (profiles: lab, server, offline)
EOF
