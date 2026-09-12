# ARTalk Streamlit Realtime

Standalone Streamlit realtime demo for the packaged ARTalk + GAGAvatar stack.

This project owns only the Streamlit/WebRTC application layer. ARTalk and
GAGAvatar are consumed as Python packages: pinned git revisions through the
lockfile by default, or editable checkouts for development.

## Quick start (fresh CUDA host)

```bash
scripts/bootstrap.sh          # pixi env from pixi.lock + pinned vendor checkouts
pixi run artalk-demo assets   # model weights (FLAME_with_eye.pt stays manual)
pixi run artalk-demo doctor   # validates install, CUDA arch coverage, assets
pixi run artalk-demo up       # launch; picks an idle GPU and its preset
```

`artalk-demo up` reads `launch.toml`: profiles (`lab` = ngrok tunnel, `server`
= local bind for SSH forwarding, `offline` = localhost + `HF_HUB_OFFLINE`) and
the measured per-GPU-tier render presets, applied as environment defaults so
explicit flags and caller env vars always win. App arguments go after `--`:

```bash
pixi run artalk-demo up --profile server -- --motion-model artalk1s
```

`scripts/bootstrap.sh --with-track` additionally provisions the avatar
registration environment (`scripts/provision_track_env.sh`, a separate env
because its dependency set conflicts with the runtime stack).

For the server path there is a `Dockerfile` + `compose.yaml` (assets and the
Hugging Face cache are volumes; the image needs the NVIDIA Container Toolkit).

### Developing against an ARTalk/GAGAvatar checkout

The environment installs both packages from pinned revisions, so edits in a
checkout are invisible to the app until it is installed over them:

```bash
uv pip install --python .pixi/envs/default/bin/python -e /path/to/ARTalk --no-deps
```

Plain `pixi run` re-syncs the environment to the lockfile and silently reverts
that, so run with `--no-install` while the override is in place:

```bash
pixi run --no-install artalk-demo up
```

`artalk-demo doctor` reports each package's origin, and `pixi install --locked`
restores the pinned state.

## Setup (development against editable checkouts)

Install the application and local editable runtime packages into the Python
environment that already contains the heavy CUDA/PyTorch stack:

```bash
export ARTALK_STREAMLIT_PYTHON=/path/to/python
export ARTALK_PACKAGE_DIR=/path/to/packaged/ARTalk
export GAGAVATAR_PACKAGE_DIR=/path/to/packaged/GAGAvatar
scripts/install_runtime.sh
```

The install script uses `uv pip --python` to install into that environment.
`ARTALK_PACKAGE_DIR` and `GAGAVATAR_PACKAGE_DIR` must point at packaged
ARTalk/GAGAvatar checkouts.

Equivalent manual commands:

```bash
uv pip install --python "$ARTALK_STREAMLIT_PYTHON" -e "$ARTALK_PACKAGE_DIR" --no-deps
uv pip install --python "$ARTALK_STREAMLIT_PYTHON" -e "$GAGAVATAR_PACKAGE_DIR" --no-deps
uv pip install --python "$ARTALK_STREAMLIT_PYTHON" av numpy openai streamlit streamlit-webrtc streamlit-remote
uv pip install --python "$ARTALK_STREAMLIT_PYTHON" -e . --no-deps
```

`--no-deps` is intentional for the heavy runtime packages;
install their PyTorch, PyTorch3D, CUDA, and Gaussian rasterizer dependencies
through the upstream environment instructions.

## Assets

The default asset root is configured in `pyproject.toml`:

```toml
[tool.artalk.assets]
root = "assets"
```

For local development, either create an ignored `assets` symlink that points at
a complete ARTalk asset tree or populate the tree with the package downloaders:

```bash
artalk-assets download --root assets --include-optional
gagavatar-assets download --root assets/GAGAvatar
```

The downloaders fetch only assets with explicit upstream sources and verify
known sizes and SHA-256 hashes. They do not download FLAME assets; provide
`assets/FLAME_with_eye.pt` manually according to the FLAME license terms.

GAGAvatar assets are resolved from the same tree by default:
`GAGAvatar/GAGAvatar.pt`, `GAGAvatar/tracked.pt`, and `FLAME_with_eye.pt`.

Override the configured root when using different assets:

```bash
export ARTALK_ASSET_DIR=/path/to/ARTalk/assets
```

To run a different ARTalk model variant (e.g. a retrained checkpoint that
shares the default audio encoder), point at it directly; the sidebar shows
which checkpoint is loaded:

```bash
export ARTALK_CHECKPOINT=/path/to/ARTalk_english.pt
# or: scripts/run_app.sh -- --artalk-checkpoint /path/to/ARTalk_english.pt
```

(`--artalk-audio-encoder` selects the encoder architecture when a variant
uses a different one.) `scripts/check_streaming_parity.py` (ARTalk repo) and
`scripts/benchmark_pipeline.py` accept the same overrides for validating a
new checkpoint before wiring it into a session.

[Fallingwater](https://github.com/xg-chu/Fallingwater), the streaming
successor to ARTalk, can drive the avatar instead. It needs its checkout (for
the `core` package) and a generator checkpoint, and pulls the MOSS audio
tokenizer (~7 GB) from Hugging Face on first use:

```bash
export ARTALK_MOTION_MODEL=fallingwater
export FALLINGWATER_DIR=/path/to/Fallingwater
export FALLINGWATER_CHECKPOINT=/path/to/iter_75000.pt
```

Generation is still chunk-wise at 4 s, so the latency floor matches ARTalk's.

For non-standard layouts, pass `--gagavatar-asset-dir`,
`--gagavatar-model-path`, `--gagavatar-tracked-path`, or
`--gagavatar-flame-model-path` as Streamlit app arguments.

## Run

For the prepared local runtime environment in this checkout:

```bash
source scripts/activate_runtime.sh
streamlit run streamlit_app.py
```

The activation helper prepends `.mamba-env/bin` to `PATH`, sets
`PYTHONNOUSERSITE=1` so global user-site packages do not override the runtime
stack, and keeps Hugging Face and package-manager caches under this checkout.

```bash
ARTALK_STREAMLIT_PYTHON=/path/to/python scripts/run_app.sh
```

Source edits do not trigger a rerun: Streamlit rescans every loaded module on
its event loop whenever `sys.modules` changes, and with this app's dependency
set that blocks the WebRTC tracks for hundreds of milliseconds to several
seconds, long enough for the browser to conceal audio and freeze video. Set
`ARTALK_STREAMLIT_SOURCE_WATCHER=1` to restore the watcher when editing.

The launcher uses `st-remote` from the selected Python environment and defaults
to an ngrok HTTPS tunnel so the browser gets the secure context that microphone
access requires. Set `ST_REMOTE_PROVIDER` or pass `--provider` to use another
tunnel provider, or pass `--no-remote` to serve locally only. Set
`ST_REMOTE_BIN` to override the executable, and set `STREAMLIT_SERVER_ADDRESS`
or `STREAMLIT_SERVER_PORT` to change the local bind address.

Pass `st-remote` options before `--`, and `streamlit_app.py` options after
`--`:

```bash
ARTALK_STREAMLIT_PYTHON=/path/to/python \
  scripts/run_app.sh --no-remote -- \
    --device cuda \
    --render-res 512 \
    --render-batch-size 8
```

The realtime output buffer can be tuned when rendering is close to, but not
always faster than, realtime. Increasing prebuffer adds latency but gives the
renderer more slack; reducing segment size publishes rendered audio/video back
to WebRTC sooner.

```bash
ARTALK_STREAMLIT_PYTHON=/path/to/python \
  scripts/run_app.sh --no-remote -- \
    --device cuda \
    --render-res 512 \
    --render-batch-size 8 \
    --output-prebuffer-seconds 2.0 \
    --output-segment-seconds 0.5
```

### When rendering falls behind

Video is slaved to a media clock that only advances on real audio, so playback
never runs ahead of what was rendered. What differs between policies is how
playback resumes after the output buffer starves mid-response:

```bash
pixi run artalk-demo up -- --output-underrun-policy rebuffer
```

`continuous` (default) resumes on the first complete audio frame, staying as
close to live as possible; when the renderer is marginal the buffer then
oscillates around empty, so speech carries repeated short silences and the
avatar repeats its last frame through each one. `rebuffer` instead waits for
the buffer to refill to `--output-rebuffer-seconds` (default: the prebuffer),
turning many stalls into few and keeping more of the rendered motion, at the
cost of trailing live by up to `--max-added-latency-seconds`; past that ceiling
it behaves like `continuous` so a durably slow renderer cannot push latency up
without bound.

The `audio_rebuffer_events`, `video_frames_dropped_for_sync` and
`added_latency_seconds` counters in the diagnostics panel show which is
happening.

## Profiling

Capture PyTorch Profiler traces of the pipeline worker to inspect per-op
CPU/GPU timing and synchronization points:

```bash
ARTALK_STREAMLIT_PYTHON=/path/to/python \
  scripts/run_app.sh --no-remote -- \
    --device cuda \
    --profile-trace-dir profiles
```

Profiling is chunk-scoped: only worker iterations that cross ARTalk's 4-second
model chunk boundary (the ones that run inference and rendering) are profiled.
By default the first chunk is skipped as warm-up and the next two are captured
(`--profile-skip-chunks`, `--profile-max-chunks`). Note that in Interactive
mode chunks fill only while the assistant is speaking, so short test
conversations may never reach the second chunk — hold a longer conversation or
pass `--profile-skip-chunks 0`. Capture counters reset when the pipeline
restarts (session stop or settings change). Each capture writes one
Chrome trace under a per-run subdirectory of `--profile-trace-dir`; open the
JSON files in <https://ui.perfetto.dev> or `chrome://tracing`. Pipeline stages
are labeled `artalk.*` in the trace. Trace export happens on the worker thread
and can stall the pipeline for a moment, so keep profiling off in normal runs.

When profiling is enabled, the app shows a **Torch profiler** panel above the
diagnostics column with a per-chunk operator summary (`key_averages`, sorted by
self CUDA time) and a download button for each Chrome trace — useful when the
app runs on a remote GPU host.

The renderer can call `torch.cuda.synchronize()` at stage boundaries so the
per-stage diagnostics timings are attributable. This is **off by default**
because the syncs serialize GPU work (measured 2x slower mesh chunk renders);
without them, per-stage timings only measure kernel launch, and queued GPU
work is attributed to whichever stage forces the next sync (typically the
GPU-to-CPU copy). Enable with `--renderer-stage-sync` when reading per-stage
timings or profiler traces.

Browser microphone access requires a secure context. Use `http://localhost:8501`
directly or forward a remote GPU host:

```bash
ssh -L 8501:localhost:8501 <gpu-host>
```

## Registering avatars from face images

The sidebar's **Register avatar** expander accepts an uploaded face image and
builds a new GAGAvatar avatar from it on demand. Tracking (face detection,
FLAME fitting, matting) runs in the separate
[GAGAvatar_track](https://github.com/xg-chu/GAGAvatar_track) environment as a
subprocess, because its dependency stack conflicts with the realtime runtime
environment. Point the app at that environment:

```bash
export GAGAVATAR_TRACK_PYTHON=/path/to/gagavatar-track-env/bin/python
export GAGAVATAR_TRACK_DIR=/path/to/GAGAvatar_track
```

(or pass `--gagavatar-track-python` / `--gagavatar-track-dir` as app
arguments). Registered avatars are stored one directory per avatar under
`--user-avatar-dir` (default `user_avatars/`, git-ignored) and appear as
`gagavatar:<name>` under **Appearance**, persisting across restarts. Names
are slugified, and a name that is already taken gets an automatic `-2` /
`-3` suffix instead of overwriting the existing avatar. The
built-in `tracked.pt` asset is never modified. Tracking runs on CPU by
default (`--gagavatar-track-device`), taking well under a minute per image
with no GPU memory contention; switch to `cuda` for faster tracking on GPUs
supported by the tracker environment's torch build.

Interactive mode picks its conversation backend in the sidebar: the OpenAI
Realtime API (or any server speaking that protocol, chosen under **Endpoint**),
OpenAI's GPT-Live, or a local PersonaPlex server. Both OpenAI backends read the
API key from a local Streamlit secrets file:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

`.streamlit/secrets.toml` is ignored by git.

GPT-Live is a separate protocol rather than another Realtime model. It is full
duplex, so like PersonaPlex it has no turn boundary to truncate at barge-in,
and it splits the voice frontend from a backend that reasons and calls tools.
That backend is off by default; the sidebar can hand it to a Responses model
with web search, billed on top of the session's per-second charge.

## Notes

- [Realtime performance notes](docs/realtime-performance-notes.md)
