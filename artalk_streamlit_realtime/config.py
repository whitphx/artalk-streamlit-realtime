"""Shared app constants and CLI parsing."""

from __future__ import annotations

import argparse
import os

OPENAI_REALTIME_SAMPLE_RATE = 24000

DEFAULT_APPEARANCE = "mesh"
DEFAULT_STYLE = "default"
DEFAULT_REALTIME_MODEL = "gpt-realtime"
DEFAULT_REALTIME_VOICE = "alloy"
REALTIME_VOICES = [
    "alloy",
    "ash",
    "ballad",
    "coral",
    "echo",
    "sage",
    "shimmer",
    "verse",
]
DEFAULT_REALTIME_INSTRUCTIONS = (
    "You are speaking through an ARTalk avatar. Keep responses concise "
    "and conversational."
)

ARTALK_SAMPLE_RATE = 16000
ARTALK_FPS = 25
DEFAULT_OUTPUT_AUDIO_PREBUFFER_SECONDS = 1.00
DEFAULT_OUTPUT_SEGMENT_SECONDS = 1.00

SILENCE_PUMP_CHUNK_SECONDS = 0.25
SILENCE_PUMP_CHUNK_SAMPLES = int(ARTALK_SAMPLE_RATE * SILENCE_PUMP_CHUNK_SECONDS)
SILENCE_PUMP_IDLE_SECONDS = 1.00
SILENCE_PUMP_MAX_AUDIO_BUFFER_SECONDS = 3.00
SILENCE_PUMP_MAX_AUDIO_BUFFER_SAMPLES = int(
    ARTALK_SAMPLE_RATE * SILENCE_PUMP_MAX_AUDIO_BUFFER_SECONDS
)
SILENCE_PUMP_MAX_VIDEO_FRAMES = int(ARTALK_FPS * SILENCE_PUMP_MAX_AUDIO_BUFFER_SECONDS)


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default=os.environ.get("ARTALK_DEVICE", "cuda"),
        type=str,
    )
    parser.add_argument(
        "--render-res",
        default=int(os.environ.get("ARTALK_RENDER_RES", "512")),
        type=int,
    )
    parser.add_argument(
        "--render-batch-size",
        default=int(os.environ.get("ARTALK_RENDER_BATCH_SIZE", "8")),
        type=int,
    )
    parser.add_argument(
        "--output-prebuffer-seconds",
        default=float(
            os.environ.get(
                "ARTALK_OUTPUT_PREBUFFER_SECONDS",
                str(DEFAULT_OUTPUT_AUDIO_PREBUFFER_SECONDS),
            )
        ),
        type=float,
    )
    parser.add_argument(
        "--output-segment-seconds",
        default=float(
            os.environ.get(
                "ARTALK_OUTPUT_SEGMENT_SECONDS",
                str(DEFAULT_OUTPUT_SEGMENT_SECONDS),
            )
        ),
        type=float,
    )
    parser.add_argument(
        "--renderer-stage-sync",
        action=argparse.BooleanOptionalAction,
        default=_env_flag("ARTALK_RENDERER_STAGE_SYNC", False),
        help=(
            "Synchronize CUDA at renderer stage boundaries so per-stage "
            "diagnostics timings are attributable. Off by default: the syncs "
            "serialize GPU work (measured 2x slower mesh chunk renders) and "
            "production runs should not pay for timing attribution."
        ),
    )
    parser.add_argument(
        "--profile-trace-dir",
        default=os.environ.get("ARTALK_PROFILE_TRACE_DIR"),
        type=str,
        help=(
            "Enable PyTorch Profiler capture and write one Chrome trace per "
            "profiled ARTalk motion chunk under this directory."
        ),
    )
    parser.add_argument(
        "--profile-skip-chunks",
        default=int(os.environ.get("ARTALK_PROFILE_SKIP_CHUNKS", "1")),
        type=int,
        help="Motion chunks to skip as warm-up before capturing traces.",
    )
    parser.add_argument(
        "--profile-max-chunks",
        default=int(os.environ.get("ARTALK_PROFILE_MAX_CHUNKS", "2")),
        type=int,
        help="Maximum number of motion-chunk traces to capture per pipeline.",
    )
    parser.add_argument(
        "--asset-dir",
        default=os.environ.get("ARTALK_ASSET_DIR"),
        type=str,
    )
    parser.add_argument(
        "--gagavatar-asset-dir",
        default=os.environ.get("GAGAVATAR_ASSET_DIR"),
        type=str,
    )
    parser.add_argument(
        "--gagavatar-model-path",
        default=os.environ.get("GAGAVATAR_MODEL_PATH"),
        type=str,
    )
    parser.add_argument(
        "--gagavatar-tracked-path",
        default=os.environ.get("GAGAVATAR_TRACKED_PATH"),
        type=str,
    )
    parser.add_argument(
        "--gagavatar-flame-model-path",
        default=os.environ.get("GAGAVATAR_FLAME_MODEL_PATH"),
        type=str,
    )
    args, _ = parser.parse_known_args()
    return args
