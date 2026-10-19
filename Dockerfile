# Deployment image for the server path (compose.yaml) and for a Hugging Face
# Docker Space (docs/hf-spaces-plan.md). Needs the NVIDIA Container Toolkit at
# runtime; the CUDA runtime itself ships inside the pixi environment, so no GPU
# is needed to build. FLAME is license-gated and never enters the image: the
# server path mounts it, the Space fetches it at start from a private repo.
FROM ghcr.io/prefix-dev/pixi:0.78.0

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000, which the base image already has
# (Ubuntu's default user), so only its directories are added. Ownership is
# given by number throughout: that user is named ubuntu, and the image has no
# user named after the home directory.
RUN mkdir -p /home/user/app && chown -R 1000:1000 /home/user
USER 1000
ENV HOME=/home/user
WORKDIR /home/user/app

# The environment and the weights are built from the lock and the manifests
# alone, so they sit in layers that application edits do not invalidate. The
# editable install of this package only records paths, so empty placeholders
# stand in for its modules until the final COPY.
COPY --chown=1000:1000 pixi.toml pixi.lock pyproject.toml ./
COPY --chown=1000:1000 wheels/*.whl ./wheels/
COPY --chown=1000:1000 scripts/bootstrap.sh ./scripts/
RUN touch streamlit_app.py && mkdir artalk_streamlit_realtime \
 && touch artalk_streamlit_realtime/__init__.py
# The gagavatar package's submodule is declared with an ssh URL, which the
# image can neither run nor authenticate.
RUN git config --global url.https://github.com/.insteadOf git@github.com: \
 && ./scripts/bootstrap.sh --without-fallingwater

# A Space has no volume to mount weights from, so it bakes the public ones in
# (set the BAKE_ASSETS=1 variable on the Space), including the audio encoder
# the 1 s model loads from the Hub cache; the compose path mounts them.
ARG BAKE_ASSETS=0
RUN if [ "$BAKE_ASSETS" = 1 ]; then \
      pixi run python -m artalk.assets download --root assets --include-optional && \
      pixi run python -m gagavatar.assets download --root assets/GAGAvatar && \
      HF_HOME=/home/user/app/.cache/huggingface pixi run python -c \
        "from transformers import Wav2Vec2Model; Wav2Vec2Model.from_pretrained('facebook/wav2vec2-xls-r-300m')"; \
    fi

# Avatar registration runs GAGAvatar_track in an environment of its own. It
# tracks on CPU, which leaves the GPU to the live session and needs neither a
# CUDA toolkit to build nor CUDA wheels in the image. Its Hub downloads go to
# the HF_HOME the launcher sets, where the tracker looks for them at run time.
COPY --from=ghcr.io/astral-sh/uv:0.12.5 /uv /usr/local/bin/uv
COPY --chown=1000:1000 envs/track-requirements.txt ./envs/
COPY --chown=1000:1000 wheels/track-cpu ./wheels/track-cpu
COPY --chown=1000:1000 scripts/provision_track_env.sh scripts/fetch_track_resources.sh ./scripts/
RUN UV_NO_CACHE=1 scripts/provision_track_env.sh --cpu \
 && HF_HOME=/home/user/app/.cache/huggingface scripts/fetch_track_resources.sh

COPY --chown=1000:1000 . .

EXPOSE 7860 8501
CMD ["scripts/spaces_start.sh"]
