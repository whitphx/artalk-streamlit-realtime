#!/bin/bash
#SBATCH -J s2s-demo
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --output=s2s-demo-%j.log
#
# Serves an open speech-to-speech backend on one GPU and opens audio_demo.py
# on it through an HTTPS tunnel. Submit from the repo root:
#
#   sbatch scripts/s2s_demo.sbatch                           # cascaded stack
#   DEMO_BACKEND=personaplex sbatch scripts/s2s_demo.sbatch  # PersonaPlex
#
# The public URL appears in s2s-demo-<jobid>.log. See the README section
# "Talking to an open speech-to-speech stack" for the variables.
set -euo pipefail

START=$SECONDS
REPO=${ARTALK_REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
if [ ! -f "$REPO/audio_demo.py" ]; then
  echo "audio_demo.py not found in $REPO; submit from the repo root or set ARTALK_REPO" >&2
  exit 1
fi
LOCAL_ENV=${S2S_DEMO_ENV:-$REPO/s2s-demo.local.env}
# shellcheck disable=SC1090
[ -f "$LOCAL_ENV" ] && . "$LOCAL_ENV"

DEMO_BACKEND=${DEMO_BACKEND:-s2s}
S2S_LANG=${S2S_LANG:-en}
case $DEMO_BACKEND in
  s2s)
    : "${S2S_DIR:?set S2S_DIR to a huggingface/speech-to-speech checkout}"
    : "${VLLM_PY:?set VLLM_PY to the python of an environment with vLLM}"
    S2S_BIN=${S2S_BIN:-$S2S_DIR/.venv/bin/speech-to-speech}
    REQUIRED=("$VLLM_PY" "$S2S_BIN")
    ;;
  personaplex)
    : "${PERSONAPLEX_PYTHON:?set PERSONAPLEX_PYTHON to the python of an environment with NVIDIA PersonaPlex installed}"
    REQUIRED=("$PERSONAPLEX_PYTHON")
    if [ "$S2S_LANG" != en ]; then
      echo "PersonaPlex speaks English only; unset S2S_LANG" >&2
      exit 1
    fi
    ;;
  *) echo "DEMO_BACKEND must be s2s or personaplex, not $DEMO_BACKEND" >&2; exit 1 ;;
esac
if [ -z "${ARTALK_STREAMLIT_PYTHON:-}" ] && [ -x "$REPO/.pixi/envs/default/bin/python" ]; then
  APP_PY=$REPO/.pixi/envs/default/bin/python
else
  APP_PY=${ARTALK_STREAMLIT_PYTHON:-python}
fi
ST_REMOTE_BIN=${ST_REMOTE_BIN:-$(dirname "$(command -v "$APP_PY")")/st-remote}
for exe in "${REQUIRED[@]}" "$ST_REMOTE_BIN"; do
  [ -x "$exe" ] || { echo "$exe is not an executable; check the variables above" >&2; exit 1; }
done
LLM=${S2S_LLM_MODEL:-Qwen/Qwen3.5-4B}
LOG_DIR=${S2S_DEMO_LOG_DIR:-$REPO/.cache/s2s-demo/${SLURM_JOB_ID:-$$}}
mkdir -p "$LOG_DIR"

# Ports derived from the job id, so that two jobs sharing a node do not collide.
BASE_PORT=$((20000 + ${SLURM_JOB_ID:-$$} % 20000))
VLLM_PORT=$BASE_PORT
S2S_PORT=$((BASE_PORT + 1))
APP_PORT=$((BASE_PORT + 2))
PERSONAPLEX_PORT=$((BASE_PORT + 3))

export PYTHONNOUSERSITE=1 WANDB_MODE=disabled OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}

PIDS=()
cleanup() {
  for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT
# scancel sends SIGTERM, which would otherwise end bash without the EXIT trap.
trap 'exit 143' TERM INT

wait_for_log() {  # log pattern pid timeout_s
  for _ in $(seq 1 "$4"); do
    grep -q "$2" "$1" 2>/dev/null && return 0
    kill -0 "$3" 2>/dev/null || return 1
    sleep 1
  done
  return 1
}

start_s2s() {
  # speech-to-speech's chat-completions backend sends no temperature, so
  # without a generation-config override vLLM samples at 1.0 whenever the model
  # snapshot ships no generation_config.json (Qwen3.5-4B does not).
  # VLLM_USE_FLASHINFER_SAMPLER=0: FlashInfer's sampler JIT-builds with ninja
  # and nvcc on first use, which the PyTorch sampler does not need.
  PATH=$(dirname "$VLLM_PY"):$PATH VLLM_USE_FLASHINFER_SAMPLER=0 \
  "$VLLM_PY" -m vllm.entrypoints.openai.api_server --model "$LLM" \
    --host 127.0.0.1 --port "$VLLM_PORT" \
    --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.35}" \
    --kv-cache-memory-bytes "${VLLM_KV_CACHE_BYTES:-1073741824}" \
    --max-model-len 8192 --max-num-seqs 4 \
    --override-generation-config '{"temperature": 0.0}' \
    --language-model-only > "$LOG_DIR/vllm.log" 2>&1 &
  VLLM_PID=$!
  PIDS+=("$VLLM_PID")
  if ! wait_for_log "$LOG_DIR/vllm.log" "Application startup complete" "$VLLM_PID" 1200; then
    echo "vLLM did not come up:" >&2
    tail -40 "$LOG_DIR/vllm.log" >&2
    exit 1
  fi
  echo "vLLM up after $((SECONDS - START)) s"
  # The first request after startup is about a second slower than later ones.
  curl -sS --max-time 60 -o /dev/null "http://127.0.0.1:$VLLM_PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$LLM\",\"messages\":[{\"role\":\"user\",\"content\":\"Hello.\"}],\"max_tokens\":16,\"chat_template_kwargs\":{\"enable_thinking\":false}}" || true

  S2S_ARGS=(
    --host 127.0.0.1 --port "$S2S_PORT"
    --llm_backend chat-completions --model_name "$LLM"
    --responses_api_base_url "http://127.0.0.1:$VLLM_PORT/v1"
    --responses_api_api_key none --responses_api_stream
    # The default of 3 holds TTS back until three sentences or the end of the
    # reply, and most conversational replies are shorter than that.
    --stream_batch_sentences 1
  )
  case $S2S_LANG in
    en) ;;
    ja)
      S2S_ARGS+=(--stt qwen3-asr --qwen3_asr_language ja --qwen3_tts_language japanese --qwen3_tts_speaker Ono_Anna)
      export ARTALK_AUDIO_DEMO_INSTRUCTIONS=${ARTALK_AUDIO_DEMO_INSTRUCTIONS:-"あなたはARTalkアバターを通して話しています。日本語で、簡潔に会話するように答えてください。"}
      ;;
    *) echo "S2S_LANG must be en or ja, not $S2S_LANG" >&2; exit 1 ;;
  esac
  [ -n "${S2S_HOLD_MS:-}" ] && S2S_ARGS+=(--speculative_reopen_ms "$S2S_HOLD_MS")
  # shellcheck disable=SC2206
  S2S_ARGS+=(${S2S_EXTRA_ARGS:-})
  (cd "$S2S_DIR" && exec "$S2S_BIN" serve "${S2S_ARGS[@]}") > "$LOG_DIR/s2s.log" 2>&1 &
  S2S_PID=$!
  PIDS+=("$S2S_PID")
  if ! wait_for_log "$LOG_DIR/s2s.log" "Uvicorn running" "$S2S_PID" 1200; then
    echo "speech-to-speech did not come up:" >&2
    tail -40 "$LOG_DIR/s2s.log" >&2
    exit 1
  fi
  echo "speech-to-speech up after $((SECONDS - START)) s: ws://127.0.0.1:$S2S_PORT/v1"

  export ARTALK_AUDIO_DEMO_ENDPOINT="Local speech-to-speech"
  export ARTALK_S2S_URL="ws://127.0.0.1:$S2S_PORT/v1"
  SERVERS="vLLM :$VLLM_PORT, speech-to-speech :$S2S_PORT"
}

start_personaplex() {
  # shellcheck disable=SC2206
  PERSONAPLEX_ARGS=(${PERSONAPLEX_EXTRA_ARGS:-})
  "$PERSONAPLEX_PYTHON" -m moshi.server --host 127.0.0.1 --port "$PERSONAPLEX_PORT" \
    "${PERSONAPLEX_ARGS[@]}" > "$LOG_DIR/personaplex.log" 2>&1 &
  PERSONAPLEX_PID=$!
  PIDS+=("$PERSONAPLEX_PID")
  if ! wait_for_log "$LOG_DIR/personaplex.log" "Access the Web UI" "$PERSONAPLEX_PID" 1200; then
    echo "the PersonaPlex server did not come up:" >&2
    tail -40 "$LOG_DIR/personaplex.log" >&2
    exit 1
  fi
  echo "PersonaPlex up after $((SECONDS - START)) s: ws://127.0.0.1:$PERSONAPLEX_PORT"

  export ARTALK_AUDIO_DEMO_BACKEND=PersonaPlex
  export ARTALK_PERSONAPLEX_URL="ws://127.0.0.1:$PERSONAPLEX_PORT"
  SERVERS="PersonaPlex :$PERSONAPLEX_PORT"
}

echo "host $(hostname) job ${SLURM_JOB_ID:-none} backend $DEMO_BACKEND lang $S2S_LANG logs $LOG_DIR"
nvidia-smi -L
case $DEMO_BACKEND in
  s2s) start_s2s ;;
  personaplex) start_personaplex ;;
esac

# A compute node sits behind the lab's NAT, so media to a browser elsewhere
# needs a TURN relay whenever its credentials are available.
if [ -n "${CLOUDFLARE_TURN_KEY_ID:-}" ]; then
  export ARTALK_ICE_PROVIDER=${ARTALK_ICE_PROVIDER:-cloudflare}
fi
# shellcheck disable=SC2206
ST_REMOTE_EXTRA=(${ST_REMOTE_ARGS:-})
(cd "$REPO" && exec "$ST_REMOTE_BIN" --host 0.0.0.0 --port "$APP_PORT" \
  --provider "${ST_REMOTE_PROVIDER:-ngrok}" --no-browser --no-tui \
  "${ST_REMOTE_EXTRA[@]}" audio_demo.py) > "$LOG_DIR/app.log" 2>&1 &
APP_PID=$!
PIDS+=("$APP_PID")
URL=""
for _ in $(seq 1 180); do
  URL=$(grep -A1 "Remote HTTPS URL:" "$LOG_DIR/app.log" 2>/dev/null | sed -n '2s/ //gp' || true)
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
  tail -40 "$LOG_DIR/app.log" >&2
  exit 1
fi
echo "ready after $((SECONDS - START)) s"
echo "Open in the browser: $URL"
echo "Node-local app: http://$(hostname -f):$APP_PORT ($SERVERS)"
wait "$APP_PID"
