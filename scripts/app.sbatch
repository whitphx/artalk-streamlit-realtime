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
ST_REMOTE_BIN=${ST_REMOTE_BIN:-$(dirname "$(command -v "$APP_PY")")/st-remote}
[ -x "$ST_REMOTE_BIN" ] || { echo "$ST_REMOTE_BIN is not an executable; set ST_REMOTE_BIN" >&2; exit 1; }

DATA_DIR=${APP_DATA_DIR:-$REPO/.cache/app}
mkdir -p "$DATA_DIR/.streamlit"
# The app runs from the data dir, because its diagnostics dumps land in the
# working directory. Only the Streamlit config follows it there. The repo's
# secrets.toml stays behind, since a key in it would be billed for every
# anonymous visitor.
ln -sfn "$REPO/.streamlit/config.toml" "$DATA_DIR/.streamlit/config.toml"
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

# The Space's login rides on HF OAuth, which a lab host does not have.
export ARTALK_LOGIN=${ARTALK_LOGIN:-0}
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
# shellcheck disable=SC2206
ST_REMOTE_EXTRA=(${ST_REMOTE_ARGS:-})
APP_LOG=$DATA_DIR/app-${SLURM_JOB_ID:-$$}-${SLURM_RESTART_COUNT:-0}.log
(cd "$DATA_DIR" && exec "$ST_REMOTE_BIN" --host 0.0.0.0 --port "$APP_PORT" \
  --provider "${ST_REMOTE_PROVIDER:-ngrok}" --no-browser --no-tui \
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
