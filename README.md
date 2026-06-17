# ARTalk Streamlit Realtime

Standalone Streamlit realtime demo for the packaged ARTalk + GAGAvatar stack.

This project owns only the Streamlit/WebRTC application layer. It expects ARTalk
and GAGAvatar to be installed as Python packages, typically from editable
checkouts while the packaging branches are under development.

## Setup

Install the application and local editable runtime packages into the Python
environment that already contains the heavy CUDA/PyTorch stack:

```bash
export ARTALK_STREAMLIT_PYTHON=/home/mil/tsuchiya/.local/share/mamba/envs/artalk-web/bin/python
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

Override these paths when using different assets:

```bash
export ARTALK_ASSET_DIR=/path/to/ARTalk/assets
export GAGAVATAR_MODEL_PATH=/path/to/GAGAvatar/assets/GAGAvatar.pt
export GAGAVATAR_TRACKED_PATH=/path/to/GAGAvatar/assets/tracked.pt
export GAGAVATAR_FLAME_MODEL_PATH=/path/to/ARTalk/assets/FLAME_with_eye.pt
```

By default, assets are resolved from the editable ARTalk package checkout when
available, using the `assets/` directory next to the installed `app` package.
GAGAvatar assets are resolved from the same ARTalk asset tree:
`GAGAvatar/GAGAvatar.pt`, `GAGAvatar/tracked.pt`, and `FLAME_with_eye.pt`.
Pass `--asset-dir` to point at a complete ARTalk asset tree. For non-standard
layouts, pass `--gagavatar-model-path`, `--gagavatar-tracked-path`, or
`--gagavatar-flame-model-path` as Streamlit app arguments.

## Run

```bash
ARTALK_STREAMLIT_PYTHON=/home/mil/tsuchiya/.local/share/mamba/envs/artalk-web/bin/python scripts/run_app.sh
```

The launcher uses `st-remote` from the selected Python environment. Set
`ST_REMOTE_BIN` to override the executable, and set `STREAMLIT_SERVER_ADDRESS`
or `STREAMLIT_SERVER_PORT` to change the local bind address.

Pass `st-remote` options before `--`, and `streamlit_app.py` options after
`--`:

```bash
ARTALK_STREAMLIT_PYTHON=/home/mil/tsuchiya/.local/share/mamba/envs/artalk-web/bin/python \
  scripts/run_app.sh --no-remote -- \
    --device cuda \
    --asset-dir /path/to/ARTalk/assets \
    --render-res 512
```

Browser microphone access requires a secure context. Use `http://localhost:8501`
directly or forward a remote GPU host:

```bash
ssh -L 8501:localhost:8501 <gpu-host>
```

Interactive mode uses the OpenAI Realtime API. Set `OPENAI_API_KEY` in the
environment or enter it in the sidebar.
