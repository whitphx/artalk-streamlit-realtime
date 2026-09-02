# Server deployment image. Needs the NVIDIA Container Toolkit at runtime; the
# CUDA runtime itself ships inside the pixi environment. No GPU is needed to
# build. Assets are volume-mounted (FLAME is license-gated and cannot be
# redistributed inside the image) — see compose.yaml.
FROM ghcr.io/prefix-dev/pixi:0.78.0

RUN apt-get update && apt-get install -y --no-install-recommends git curl ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . .
RUN ./scripts/bootstrap.sh

EXPOSE 8501
ENTRYPOINT ["pixi", "run", "artalk-demo"]
CMD ["up", "--profile", "server"]
