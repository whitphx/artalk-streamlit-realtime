# Deployment image for the server path (compose.yaml) and for a Hugging Face
# Docker Space (docs/hf-spaces-plan.md). Needs the NVIDIA Container Toolkit at
# runtime; the CUDA runtime itself ships inside the pixi environment, so no GPU
# is needed to build. FLAME is license-gated and never enters the image: the
# server path mounts it, the Space fetches it at start from a private repo.
FROM ghcr.io/prefix-dev/pixi:0.78.0

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
WORKDIR /home/user/app
COPY --chown=user . .
RUN ./scripts/bootstrap.sh

# A Space has no volume to mount weights from, so it bakes the public ones in
# (set the BAKE_ASSETS=1 variable on the Space); the compose path mounts them.
ARG BAKE_ASSETS=0
RUN if [ "$BAKE_ASSETS" = 1 ]; then \
      pixi run python -m artalk.assets download --root assets --include-optional && \
      pixi run python -m gagavatar.assets download --root assets/GAGAvatar; \
    fi

EXPOSE 7860 8501
CMD ["scripts/spaces_start.sh"]
