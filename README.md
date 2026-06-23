# ARTalk Streamlit Realtime

Standalone Streamlit realtime demo for the packaged ARTalk + GAGAvatar stack.

This project owns only the Streamlit/WebRTC application layer. It expects ARTalk
and GAGAvatar to be installed as Python packages, typically from editable
checkouts while the packaging branches are under development.

## Setup

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

For non-standard layouts, pass `--gagavatar-asset-dir`,
`--gagavatar-model-path`, `--gagavatar-tracked-path`, or
`--gagavatar-flame-model-path` as Streamlit app arguments.

## Run

```bash
ARTALK_STREAMLIT_PYTHON=/path/to/python scripts/run_app.sh
```

The launcher uses `st-remote` from the selected Python environment. Set
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

Browser microphone access requires a secure context. Use `http://localhost:8501`
directly or forward a remote GPU host:

```bash
ssh -L 8501:localhost:8501 <gpu-host>
```

Interactive mode uses the OpenAI Realtime API. Store the API key in a local
Streamlit secrets file:

```toml
# .streamlit/secrets.toml
OPENAI_API_KEY = "sk-..."
```

`.streamlit/secrets.toml` is ignored by git.

## Notes

- [Realtime performance notes](docs/realtime-performance-notes.md)
