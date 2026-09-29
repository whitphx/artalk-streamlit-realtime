#!/usr/bin/env bash
# Bring a fresh Linux CUDA host to a runnable state:
#   scripts/bootstrap.sh [--with-track] [--without-fallingwater]
# then: pixi run artalk-demo assets && pixi run artalk-demo up
set -euo pipefail

cd "$(dirname "$0")/.."
REPO_ROOT="$PWD"

# Pinned revisions of the checkouts the app consumes as directories (the
# artalk/gagavatar Python packages themselves are pinned in pixi.toml).
FALLINGWATER_REPO=https://github.com/whitphx/Fallingwater.git
FALLINGWATER_REV=8fbde2eb99f5d7fd15fc0c74430c734cd0a9bbfd
# train_code for --motion-model artalk1s; same repo as the artalk package.
# Deliberately behind the package pin in pixi.toml: later revisions add an
# audio-context argument to CrossAttention.forward, and the decoder constant
# cache in artalk_streamlit_realtime/artalk1s.py recognises the model by that
# signature, so it would quietly disable itself and the graph capture with it.
# Published checkpoints still load here; move both pins together.
ARTALK_TRAIN_REPO=https://github.com/whitphx/ARTalk.git
ARTALK_TRAIN_REV=8460342f4de2ec5ac9176bfd4409a4bb8872698f
# Carries the vendored GAGAvatar_track (gagavatar/libs/GAGAvatar_track).
GAGAVATAR_REPO=https://github.com/whitphx/GAGAvatar.git
GAGAVATAR_REV=437dadea35d3069abdcc2026f661869fd608231d

WITH_TRACK=0
WITH_FALLINGWATER=1
for arg in "$@"; do
  case "$arg" in
    --with-track) WITH_TRACK=1 ;;
    # The Fallingwater fork is private; hosts without GitHub credentials
    # (the Space image) run without that motion model.
    --without-fallingwater) WITH_FALLINGWATER=0 ;;
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
    # Upstream .gitmodules uses SSH URLs; stay on HTTPS so a fresh machine
    # without GitHub keys can clone.
    git -C "$dest" -c 'url.https://github.com/.insteadOf=git@github.com:' \
      submodule update --init --depth 1
  fi
}

mkdir -p vendor
if [[ "$WITH_FALLINGWATER" == 1 ]]; then
  fetch_rev "$FALLINGWATER_REPO" "$FALLINGWATER_REV" vendor/Fallingwater
fi
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
