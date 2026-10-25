#!/bin/bash
#SBATCH -J artalk-app
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --output=app-%j.log
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#
# Serves the avatar app on one GPU through an HTTPS tunnel, with the
# Space's visitor-facing settings (launch.toml [profiles.spaces]) and the lab
# profile's thread cap. Shortly before the walltime runs out, the job
# requeues itself, so it keeps serving until it is cancelled. Submit from the
# repo root:
#
#   sbatch scripts/app.sbatch
#   sbatch --chdir=/path/to/log-dir scripts/app.sbatch  # keep the log elsewhere
#
# The public URL appears in app-<jobid>.log once the tunnel is up. The job
# keeps its id across requeues, and only cancelling it stops it:
#
#   scancel <jobid>
#
# Local paths come from an optional env file, $APP_ENV (default
# ~/.config/artalk/app.env), for example:
#
#   export APP_DATA_DIR=/path/to/large-storage/artalk-app
#   export HF_HOME=/path/to/huggingface-cache
#   export ST_REMOTE_ARGS="--ngrok-binary /path/to/ngrok"
#
# The app runs on the repo's pixi env unless the env file sets
# ARTALK_STREAMLIT_PYTHON to another interpreter, such as a venv made with
# --system-site-packages over the pixi env that swaps in one package build:
#
#   export ARTALK_STREAMLIT_PYTHON=/path/to/overlay-venv/bin/python
#
# Cloudflare TURN credentials (CLOUDFLARE_TURN_KEY_ID,
# CLOUDFLARE_TURN_KEY_API_TOKEN) are read from the environment or from
# $TURN_ENV (default ~/.config/artalk/turn.env).
#
# The sidebar opens on Interactive mode with the OpenAI Live backend; the env
# file can set ARTALK_DEFAULT_MODE and ARTALK_DEFAULT_BACKEND instead.
#
# Avatar registration needs the tracker environment, provisioned once:
#
#   scripts/provision_track_env.sh --cpu
#   TRACK_ASSETS_DIR=/path/to/large-storage/track-assets \
#     scripts/fetch_track_resources.sh
#
# The job looks for it in $ARTALK_TRACK_ENV (default .envs/track in the repo)
# and tracks on CPU. Without it, registration stays off.
#
# Sign-in with Hugging Face:
#
#   1. Find the tunnel's hostname (the "Open in the browser" line of an
#      earlier run's log; it must stay the same across runs, such as an
#      ngrok static domain), and set it in the env file:
#        export ARTALK_PUBLIC_HOST=app.example.com
#   2. Create an OAuth app in the Hugging Face settings
#      (https://huggingface.co/docs/hub/oauth#creating-an-oauth-app), with
#      the redirect URI https://app.example.com/oauth2callback and the
#      scopes openid, profile, email and read-memberships.
#   3. Put its credentials in $OAUTH_ENV (default ~/.config/artalk/oauth.env),
#      readable only by you (chmod 600):
#        OAUTH_CLIENT_ID=...
#        OAUTH_CLIENT_SECRET=...
#        # Optional: keeps visitors logged in across restarts.
#        ARTALK_AUTH_COOKIE_SECRET=<output of `openssl rand -hex 32`>
#   4. Optionally restrict who may sign in, in the env file:
#        export ARTALK_LOGIN_ORGS=example-org
#        export ARTALK_LOGIN_USERS=alice,bob
#        export ARTALK_LOGIN_EMAIL_DOMAINS=example.com
#
# With login on and at least one of those lists set, signed-in visitors can
# use the operator's OpenAI key from $OPENAI_ENV (default
# ~/.config/artalk/openai.env, chmod 600), which exports OPENAI_API_KEY.
# ARTALK_OPENAI_KEY_FOR_ALL=1 in the env file hands that key to every
# visitor, signed in or not; set it only when the key's spend is capped.
set -euo pipefail

START=$SECONDS
REPO=${ARTALK_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$REPO/streamlit_app.py" ]; then
  echo "streamlit_app.py not found in $REPO; submit from the repo root or set ARTALK_REPO" >&2
  exit 1
fi
CONFIG_DIR=${XDG_CONFIG_HOME:-$HOME/.config}/artalk
LOCAL_ENV=${APP_ENV:-$CONFIG_DIR/app.env}
# shellcheck disable=SC1090
[ -f "$LOCAL_ENV" ] && . "$LOCAL_ENV"
TURN_ENV=${TURN_ENV:-$CONFIG_DIR/turn.env}
if [ -f "$TURN_ENV" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$TURN_ENV"
  set +a
fi

if [ -z "${ARTALK_STREAMLIT_PYTHON:-}" ] && [ -x "$REPO/.pixi/envs/default/bin/python" ]; then
  APP_PY=$REPO/.pixi/envs/default/bin/python
else
  APP_PY=${ARTALK_STREAMLIT_PYTHON:-python}
fi
if [ -n "${ST_REMOTE_BIN:-}" ]; then
  [ -x "$ST_REMOTE_BIN" ] || { echo "ST_REMOTE_BIN=$ST_REMOTE_BIN is not an executable" >&2; exit 1; }
  ST_REMOTE=("$ST_REMOTE_BIN")
else
  # Not the st-remote script next to the interpreter: an overlay venv has
  # none, and the base env's script would start Streamlit on the base env's
  # Python, since st-remote runs Streamlit with its own interpreter.
  ST_REMOTE=("$APP_PY" -c 'from streamlit_remote.cli import main; main()')
fi

DATA_DIR=${APP_DATA_DIR:-$REPO/.cache/app}
mkdir -p "$DATA_DIR/.streamlit"
# The app runs from the data dir, because its diagnostics dumps land in the
# working directory. Only the Streamlit config follows it there.
ln -sfn "$REPO/.streamlit/config.toml" "$DATA_DIR/.streamlit/config.toml"
# Streamlit reads only this job's own secrets file, never the repo's or the
# home directory's, since a key in those would be billed for every visitor.
# The home directory is shared, so the file stays on the node.
SECRETS_DIR=$(mktemp -d)
trap 'rm -rf "$SECRETS_DIR"' EXIT
SECRETS_FILE=$SECRETS_DIR/secrets.toml
: > "$SECRETS_FILE"
# The same goes for keys in the environment, which the app reads as a
# fallback for the secrets file.
while read -r name; do unset "$name"; done < <(compgen -e | grep '_API_KEY$' || true)

# A port derived from the job id, so that two jobs sharing a node do not collide.
APP_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))

export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export ARTALK_ASSET_DIR=${ARTALK_ASSET_DIR:-$REPO/assets}
export ARTALK_USER_AVATAR_DIR=${ARTALK_USER_AVATAR_DIR:-$DATA_DIR/user_avatars}
export ARTALK_SESSION_DATA_DIR=${ARTALK_SESSION_DATA_DIR:-$DATA_DIR/session_data}
# An A100 holds about two photoreal sessions.
export ARTALK_MAX_SESSIONS=${ARTALK_MAX_SESSIONS:-2}
export ARTALK_DEFAULT_MODE=${ARTALK_DEFAULT_MODE:-interactive}
export ARTALK_DEFAULT_BACKEND=${ARTALK_DEFAULT_BACKEND:-openai-live}

TRACK_ENV=${ARTALK_TRACK_ENV:-$REPO/.envs/track}
TRACK_DIR=${GAGAVATAR_TRACK_DIR:-$REPO/vendor/GAGAvatar/gagavatar/libs/GAGAvatar_track}
# The weights may be links to another disk, which -e follows.
if [ -x "$TRACK_ENV/bin/python" ] \
  && [ -e "$TRACK_DIR/assets/emica/EMICA-CVT_flame2020_notexture.pt" ] \
  && [ -e "$TRACK_DIR/assets/flame/FLAME_with_eye.pt" ]; then
  export GAGAVATAR_TRACK_PYTHON=$TRACK_ENV/bin/python GAGAVATAR_TRACK_DIR=$TRACK_DIR
  # The GPU belongs to the live sessions.
  export GAGAVATAR_TRACK_DEVICE=cpu
  REGISTRATION_STATUS="on, tracking on CPU with $TRACK_ENV"
else
  # Empty rather than unset, so that the launcher's own lookup below cannot
  # turn registration back on with half the tracker missing.
  export GAGAVATAR_TRACK_PYTHON=
  REGISTRATION_STATUS="off: no tracker environment in $TRACK_ENV or no weights under $TRACK_DIR/assets (see the header of this script)"
fi

OAUTH_ENV=${OAUTH_ENV:-$CONFIG_DIR/oauth.env}
OPENAI_ENV=${OPENAI_ENV:-$CONFIG_DIR/openai.env}
for name in ARTALK_LOGIN_ORGS ARTALK_LOGIN_USERS ARTALK_LOGIN_EMAIL_DOMAINS; do
  if [ -n "${!name:-}" ]; then export "${name?}"; fi
done
if [ ! -f "$OAUTH_ENV" ]; then
  LOGIN_STATUS="off: no $OAUTH_ENV"
elif [ -z "${ARTALK_PUBLIC_HOST:-}" ]; then
  LOGIN_STATUS="off: ARTALK_PUBLIC_HOST is not set"
# The client secret stays in this subshell, out of the app's environment.
elif (
  set -a
  # shellcheck disable=SC1090
  . "$OAUTH_ENV"
  set +a
  export OPENID_PROVIDER_URL=https://huggingface.co SPACE_HOST=$ARTALK_PUBLIC_HOST
  # The allow-lists check the e-mail and organizations claims, which Hugging
  # Face sends only for these scopes
  # (https://huggingface.co/docs/hub/oauth#currently-supported-scopes).
  export OAUTH_SCOPES=${OAUTH_SCOPES:-"openid profile email read-memberships"}
  cd "$REPO" && "$APP_PY" -c '
import sys
from pathlib import Path

from artalk_streamlit_realtime.login import write_hf_oauth_secrets

sys.exit(0 if write_hf_oauth_secrets(Path(sys.argv[1])) else 1)
' "$SECRETS_FILE"
); then
  export ARTALK_LOGIN=1
  LOGIN_STATUS="on, redirect URI https://$ARTALK_PUBLIC_HOST/oauth2callback"
else
  LOGIN_STATUS="off: $OAUTH_ENV lacks OAUTH_CLIENT_ID or OAUTH_CLIENT_SECRET"
fi
# The Space profile below turns login on, which without the section written
# above would only show visitors a configuration error.
export ARTALK_LOGIN=${ARTALK_LOGIN:-0}

KEY_STATUS="none; visitors enter their own"
if [ -f "$OPENAI_ENV" ]; then
  if [ "${ARTALK_OPENAI_KEY_FOR_ALL:-0}" = 1 ]; then
    # shellcheck disable=SC1090
    . "$OPENAI_ENV"
    KEY_STATUS="operator key from $OPENAI_ENV for every visitor (ARTALK_OPENAI_KEY_FOR_ALL=1)"
  elif [ "$ARTALK_LOGIN" != 1 ]; then
    KEY_STATUS="none; $OPENAI_ENV is used only with login on"
  elif [ -z "${ARTALK_LOGIN_ORGS:-}${ARTALK_LOGIN_USERS:-}${ARTALK_LOGIN_EMAIL_DOMAINS:-}" ]; then
    # Empty allow-lists admit every Hugging Face account.
    KEY_STATUS="none; $OPENAI_ENV is used only with an ARTALK_LOGIN_* allow-list"
  else
    # shellcheck disable=SC1090
    . "$OPENAI_ENV"
    KEY_STATUS="operator key from $OPENAI_ENV for signed-in visitors"
  fi
fi
# A compute node sits behind the lab's NAT, so media to a browser elsewhere
# needs a TURN relay.
if [ -n "${CLOUDFLARE_TURN_KEY_ID:-}" ] && [ -n "${CLOUDFLARE_TURN_KEY_API_TOKEN:-}" ]; then
  export ARTALK_ICE_PROVIDER=cloudflare
  TURN_STATUS="Cloudflare TURN relay"
else
  export ARTALK_ICE_PROVIDER=stun
  TURN_STATUS="STUN only"
  echo "WARNING: no Cloudflare TURN credentials (CLOUDFLARE_TURN_KEY_ID, CLOUDFLARE_TURN_KEY_API_TOKEN, or $TURN_ENV); WebRTC falls back to STUN, and visitors outside the cluster network will likely fail to connect"
fi

# The rest comes from launch.toml through the launcher's own resolution
# (Space profile, lab thread cap, GPU preset, vendor checkouts), which never
# overrides a variable already set above. `artalk-demo up` itself cannot run
# here: it fixes the port, and it has no way to pass st-remote's flags.
LAUNCH_ENV=$(cd "$REPO" && "$APP_PY" - <<'EOF'
import shlex

from artalk_streamlit_realtime import launcher

config = launcher._load_launch_config()
profiles = config["profiles"]
profile = launcher._merge(profiles["spaces"], {"env": profiles["lab"].get("env", {})})
gpus = launcher._discover_gpus()
# Slurm shows the job only its own GPU, and CUDA_VISIBLE_DEVICES already
# pins it, so the GPU is used for the preset alone.
preset = launcher._match_preset(config.get("presets", []), gpus[0]) if gpus else None
print(f"PRESET={shlex.quote(preset['name'] if preset else 'none')}")
for key, value in sorted(launcher._compute_env(profile, preset, None).items()):
    print(f"export {key}={shlex.quote(value)}")
EOF
)
eval "$LAUNCH_ENV"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
  rm -rf "$SECRETS_DIR"
}
trap cleanup EXIT
# scancel sends SIGTERM, which would otherwise end bash without the EXIT trap.
trap 'exit 143' TERM INT
requeue() {
  echo "walltime nearly over; requeueing job $SLURM_JOB_ID"
  scontrol requeue "$SLURM_JOB_ID"
}
[ -n "${SLURM_JOB_ID:-}" ] && trap requeue USR1

echo "host $(hostname) job ${SLURM_JOB_ID:-none} preset $PRESET data $DATA_DIR"
nvidia-smi -L
echo "WebRTC: $TURN_STATUS"
echo "Avatar registration: $REGISTRATION_STATUS"
echo "Login: $LOGIN_STATUS"
echo "OpenAI key: $KEY_STATUS"
# shellcheck disable=SC2206
ST_REMOTE_EXTRA=(${ST_REMOTE_ARGS:-})
APP_LOG=$DATA_DIR/app-${SLURM_JOB_ID:-$$}-${SLURM_RESTART_COUNT:-0}.log
(cd "$DATA_DIR" && exec "${ST_REMOTE[@]}" --host 0.0.0.0 --port "$APP_PORT" \
  --provider "${ST_REMOTE_PROVIDER:-ngrok}" --no-browser --no-tui \
  --streamlit-arg="--secrets.files=$SECRETS_FILE" \
  "${ST_REMOTE_EXTRA[@]}" "$REPO/streamlit_app.py") > "$APP_LOG" 2>&1 &
APP_PID=$!
PIDS+=("$APP_PID")
URL=""
for _ in $(seq 1 300); do
  URL=$(grep -A1 "Remote HTTPS URL:" "$APP_LOG" 2>/dev/null | sed -n '2s/ //gp' || true)
  [ -n "$URL" ] && curl -fsS --max-time 5 -o /dev/null "http://127.0.0.1:$APP_PORT/_stcore/health" && break
  kill -0 "$APP_PID" 2>/dev/null || break
  sleep 1
done
# The tunnel prints its URL before it registers it, and an ngrok static domain
# that another session already serves is refused only then, which takes the
# app down with it.
sleep 5
if [ -z "$URL" ] || ! kill -0 "$APP_PID" 2>/dev/null; then
  echo "the app or its tunnel did not come up:" >&2
  tail -40 "$APP_LOG" >&2
  exit 1
fi
echo "ready after $((SECONDS - START)) s; app log $APP_LOG"
echo "Open in the browser: $URL"
URL_HOST=${URL#https://}
URL_HOST=${URL_HOST%%/*}
if [ "$ARTALK_LOGIN" = 1 ] && [ "$URL_HOST" != "$ARTALK_PUBLIC_HOST" ]; then
  echo "WARNING: the tunnel serves $URL_HOST, not ARTALK_PUBLIC_HOST=$ARTALK_PUBLIC_HOST; sign-in will fail"
elif [ "$ARTALK_LOGIN" != 1 ] && [ -f "$OAUTH_ENV" ] && [ -z "${ARTALK_PUBLIC_HOST:-}" ]; then
  echo "To turn login on, set ARTALK_PUBLIC_HOST=$URL_HOST and register https://$URL_HOST/oauth2callback in the OAuth app"
fi
echo "Node-local app: http://$(hostname -f):$APP_PORT"
# The USR1 trap interrupts wait while the app keeps running, so wait again
# until the app itself exits or Slurm ends the job.
set +e
while :; do
  wait "$APP_PID"
  status=$?
  kill -0 "$APP_PID" 2>/dev/null || break
done
exit "$status"
