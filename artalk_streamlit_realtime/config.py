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

SILENCE_PUMP_CHUNK_SECONDS = 0.25
SILENCE_PUMP_CHUNK_SAMPLES = int(ARTALK_SAMPLE_RATE * SILENCE_PUMP_CHUNK_SECONDS)
SILENCE_PUMP_IDLE_SECONDS = 1.00
SILENCE_PUMP_MAX_AUDIO_BUFFER_SECONDS = 3.00
SILENCE_PUMP_MAX_AUDIO_BUFFER_SAMPLES = int(
    ARTALK_SAMPLE_RATE * SILENCE_PUMP_MAX_AUDIO_BUFFER_SECONDS
)
SILENCE_PUMP_MAX_VIDEO_FRAMES = int(ARTALK_FPS * SILENCE_PUMP_MAX_AUDIO_BUFFER_SECONDS)


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

