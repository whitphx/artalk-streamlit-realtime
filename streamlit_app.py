#!/usr/bin/env python

"""Standalone Streamlit + streamlit-webrtc realtime demo for ARTalk.

This app is intentionally outside the ARTalk repository. It expects ARTalk and
GAGAvatar to be installed as Python packages and locates model assets through
environment variables.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import av
import numpy as np
import streamlit as st
import torch
from streamlit.errors import StreamlitSecretNotFoundError
from artalk.assets import ARTalkAssets
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_sink_track,
    create_audio_source_track,
    create_video_source_track,
    webrtc_streamer,
)
from streamlit_webrtc.shutdown import SessionShutdownObserver

from artalk.flame_model import RenderMesh
from artalk.realtime_pipeline import ARTalkPipeline
from artalk.runtime import ARTalkRuntime, ARTalkRuntimeConfig, available_styles
from gagavatar.assets import AssetConfigError as GAGAvatarAssetConfigError
from gagavatar.assets import GAGAvatarAssets

logger = logging.getLogger(__name__)

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


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default
    return str(value) if value is not None else default


def resolve_gagavatar_assets(args, artalk_assets: ARTalkAssets) -> GAGAvatarAssets:
    has_gagavatar_override = any(
        [
            args.gagavatar_asset_dir,
            args.gagavatar_model_path,
            args.gagavatar_tracked_path,
            args.gagavatar_flame_model_path,
        ]
    )
    if has_gagavatar_override:
        return GAGAvatarAssets.resolve(
            root=args.gagavatar_asset_dir,
            model_path=args.gagavatar_model_path,
            tracked_path=args.gagavatar_tracked_path,
            flame_model_path=args.gagavatar_flame_model_path,
        )
    if args.asset_dir:
        return GAGAvatarAssets.from_artalk_assets(artalk_assets)
    try:
        return GAGAvatarAssets.from_pyproject()
    except GAGAvatarAssetConfigError:
        return GAGAvatarAssets.from_artalk_assets(artalk_assets)


@st.cache_resource
def load_artalk_runtime(device: str, render_res: int, asset_root: str):
    assets = ARTalkAssets.from_root(asset_root)
    runtime = ARTalkRuntime(
        ARTalkRuntimeConfig(
            assets=assets,
            device=device,
            flame_scale=1.0,
        )
    )
    mesh = RenderMesh(
        image_size=render_res,
        faces=runtime.flame_model.get_faces(),
        scale=1.0,
    )
    return runtime, mesh


@st.cache_data
def list_gagavatar_ids(tracked_path: str | None):
    if not tracked_path:
        return []
    try:
        tracked = torch.load(tracked_path, map_location="cpu", weights_only=False)
    except FileNotFoundError:
        return []
    avatar_ids = ["avatar"] if "avatar" in tracked and len(tracked) == 1 else sorted(tracked)
    return avatar_ids


@st.cache_data
def list_style_ids(asset_dir: str):
    return available_styles(asset_dir)


@st.cache_data
def load_style_motion(asset_dir: str, style_id: str):
    if style_id == DEFAULT_STYLE:
        return None
    style_motion = torch.load(
        Path(asset_dir) / "style_motion" / f"{style_id}.pt",
        map_location="cpu",
        weights_only=True,
    )
    if tuple(style_motion.shape) != (50, 106):
        raise ValueError(f"Invalid style motion shape: {tuple(style_motion.shape)}")
    return style_motion


class StreamingGAGAvatarAdapter:
    """Adapter from packaged ``gagavatar.runtime`` to ARTalk's streaming renderer API."""

    def __init__(self, runtime):
        self.runtime = runtime

    def set_avatar_id(self, avatar_id: str):
        self.runtime.set_avatar_id(avatar_id)

    def build_forward_batch(self, motion_code: torch.Tensor, _flame_model=None):
        return self.runtime.build_forward_batch(motion_code)

    def forward_expression(self, batch: dict):
        return self.runtime.render_rgb_batch(batch)


class PipelineSilencePump:
    """App-level idle audio filler for ARTalk's chunked streaming model.

    OpenAI Realtime and browser loopback both deliver audio only while the
    upstream source is producing sound. ARTalk, however, emits motion only after
    enough samples have accumulated for its model chunk. If the source goes
    silent, the final partial chunk can sit below that threshold forever, which
    stops new frames. The app owns this glue policy because it knows the upstream
    transport semantics; ARTalk only exposes ``push_silence`` as a sample input.
    """

    def __init__(self, pipeline: ARTalkPipeline) -> None:
        self._pipeline = pipeline
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._input_started = False
        self._last_real_input_s = 0.0
        self._last_pump_s = 0.0
        self._thread: threading.Thread | None = None

        metrics = self._pipeline.metrics
        metrics.set("silence_pump_chunk_seconds", SILENCE_PUMP_CHUNK_SECONDS)
        metrics.set("silence_pump_chunk_samples", SILENCE_PUMP_CHUNK_SAMPLES)
        metrics.set("silence_pump_idle_seconds", SILENCE_PUMP_IDLE_SECONDS)
        metrics.set(
            "silence_pump_max_audio_buffer_samples",
            SILENCE_PUMP_MAX_AUDIO_BUFFER_SAMPLES,
        )
        metrics.set("silence_pump_max_video_frames", SILENCE_PUMP_MAX_VIDEO_FRAMES)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ARTalkSilencePump",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    def mark_input(self) -> None:
        with self._lock:
            self._input_started = True
            self._last_real_input_s = time.perf_counter()
        self._pipeline.metrics.inc("silence_pump_real_input_marks")

    def _run(self) -> None:
        while not self._stop_event.wait(0.1):
            if self._pipeline.is_stopped:
                return
            with self._lock:
                input_started = self._input_started
                last_real_input_s = self._last_real_input_s
                last_pump_s = self._last_pump_s
            if not input_started:
                continue

            now = time.perf_counter()
            idle_s = now - last_real_input_s if last_real_input_s else 0.0
            metrics = self._pipeline.metrics
            metrics.set("silence_pump_input_idle_s", idle_s)
            if idle_s < SILENCE_PUMP_IDLE_SECONDS:
                metrics.inc("silence_pump_recent_input_skips")
                continue
            if last_pump_s and now - last_pump_s < SILENCE_PUMP_CHUNK_SECONDS:
                metrics.inc("silence_pump_pacing_skips")
                continue

            snapshot = self._pipeline.output_buffer_snapshot()
            audio_buffered = snapshot["audio_out_buffer_samples"]
            video_depth = snapshot["video_queue_depth"]
            audio_in_depth = snapshot["audio_in_queue_depth"]
            worker_busy = bool(snapshot.get("worker_busy"))
            metrics.set("silence_pump_audio_buffer_samples", audio_buffered)
            metrics.set("silence_pump_video_queue_depth", video_depth)
            metrics.set("silence_pump_audio_in_queue_depth", audio_in_depth)
            metrics.set("silence_pump_worker_busy", 1 if worker_busy else 0)
            if worker_busy:
                metrics.inc("silence_pump_worker_busy_skips")
                continue
            if audio_in_depth > 0:
                metrics.inc("silence_pump_input_queue_skips")
                continue
            if (
                audio_buffered >= SILENCE_PUMP_MAX_AUDIO_BUFFER_SAMPLES
                or video_depth >= SILENCE_PUMP_MAX_VIDEO_FRAMES
            ):
                metrics.inc("silence_pump_backpressure_skips")
                continue

            with self._lock:
                self._last_pump_s = now
            metrics.inc("silence_pump_chunks")
            metrics.inc("silence_pump_samples", SILENCE_PUMP_CHUNK_SAMPLES)
            self._pipeline.push_silence(SILENCE_PUMP_CHUNK_SECONDS)


@st.cache_resource
def load_gagavatar(device: str, model_path: str, tracked_path: str | None, flame_model_path: str | None):
    from gagavatar.runtime import GAGAvatarRuntime, GAGAvatarRuntimeConfig

    runtime = GAGAvatarRuntime(
        GAGAvatarRuntimeConfig(
            model_path=model_path,
            tracked_path=tracked_path,
            flame_model_path=flame_model_path,
            device=device,
        )
    )
    return StreamingGAGAvatarAdapter(runtime), runtime.flame_model


class OpenAIRealtimeBridge:
    """Bridge browser mic audio to OpenAI and feed response audio to ARTalk."""

    def __init__(
        self,
        *,
        api_key: str,
        pipeline: ARTalkPipeline,
        on_audio_output: Callable[[], None] | None,
        model: str,
        voice: str,
        instructions: str,
    ) -> None:
        self._api_key = api_key
        self._pipeline = pipeline
        self._on_audio_output = on_audio_output
        self._model = model
        self._voice = voice
        self._instructions = instructions

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._input_queue: Optional["asyncio.Queue[bytes]"] = None
        self._stop_event: Optional[asyncio.Event] = None
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._resampler = av.AudioResampler(
            format="s16", layout="mono", rate=OPENAI_REALTIME_SAMPLE_RATE
        )

        self._state_lock = threading.Lock()
        self._connected = False
        self._error: Optional[str] = None
        self._user_transcript = ""
        self._assistant_transcript = ""

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._ready_event.clear()
        with self._state_lock:
            self._error = None
        self._thread = threading.Thread(
            target=self._run,
            name="OpenAIRealtimeBridge",
            daemon=True,
        )
        self._thread.start()
        self._ready_event.wait(timeout=3.0)

    def wait_until_connected(self, timeout: float) -> bool:
        stop_at = time.monotonic() + timeout
        while time.monotonic() < stop_at:
            with self._state_lock:
                if self._connected:
                    return True
                if self._error:
                    return False
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        loop, stop_event = self._loop, self._stop_event
        if loop is not None and stop_event is not None and not loop.is_closed():
            loop.call_soon_threadsafe(stop_event.set)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=3.0)
        self._thread = None
        with self._state_lock:
            self._connected = False

    def push_input(self, frame: av.AudioFrame) -> None:
        loop, q = self._loop, self._input_queue
        if loop is None or q is None or loop.is_closed():
            return
        for resampled in self._resampler.resample(frame):
            arr = resampled.to_ndarray()
            pcm = arr.astype(np.int16, copy=False).tobytes()
            if not pcm:
                continue
            try:
                loop.call_soon_threadsafe(self._queue_input, pcm)
            except RuntimeError:
                return

    def _queue_input(self, pcm: bytes) -> None:
        if self._input_queue is None:
            return
        try:
            self._input_queue.put_nowait(pcm)
        except asyncio.QueueFull:
            pass

    def snapshot(self) -> dict:
        with self._state_lock:
            return {
                "connected": self._connected,
                "error": self._error,
                "user": self._user_transcript,
                "assistant": self._assistant_transcript,
            }

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._input_queue = asyncio.Queue(maxsize=256)
            self._stop_event = asyncio.Event()
            self._ready_event.set()
            loop.run_until_complete(self._session())
        except Exception as exc:
            logger.exception("OpenAI Realtime bridge crashed")
            with self._state_lock:
                self._error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._state_lock:
                self._connected = False
            try:
                loop.close()
            finally:
                self._loop = None

    async def _session(self) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise RuntimeError(
                "Install the OpenAI SDK to use Interactive mode: pip install openai"
            ) from exc

        if self._stop_event is None or self._input_queue is None:
            raise RuntimeError("Realtime bridge loop is not initialized")

        client = AsyncOpenAI(api_key=self._api_key)
        async with client.realtime.connect(model=self._model) as conn:
            await conn.session.update(
                session={
                    "type": "realtime",
                    "model": self._model,
                    "instructions": self._instructions,
                    "audio": {
                        "input": {"turn_detection": {"type": "server_vad"}},
                        "output": {"voice": self._voice},
                    },
                }
            )
            with self._state_lock:
                self._connected = True

            tasks = [
                asyncio.create_task(self._send_loop(conn), name="openai-send"),
                asyncio.create_task(self._recv_loop(conn), name="openai-recv"),
                asyncio.create_task(self._stop_event.wait(), name="openai-stop"),
            ]
            try:
                await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

    async def _send_loop(self, conn) -> None:
        if self._input_queue is None:
            return
        while True:
            pcm = await self._input_queue.get()
            await conn.input_audio_buffer.append(
                audio=base64.b64encode(pcm).decode("ascii")
            )

    async def _recv_loop(self, conn) -> None:
        async for event in conn:
            etype = getattr(event, "type", "")
            if etype == "response.output_audio.delta":
                self._push_response_audio(base64.b64decode(event.delta))
            elif etype == "response.output_audio_transcript.delta":
                with self._state_lock:
                    self._assistant_transcript += getattr(event, "delta", "") or ""
            elif etype == "response.done":
                with self._state_lock:
                    if (
                        self._assistant_transcript
                        and not self._assistant_transcript.endswith("\n")
                    ):
                        self._assistant_transcript += "\n"
            elif etype == "conversation.item.input_audio_transcription.delta":
                with self._state_lock:
                    self._user_transcript += getattr(event, "delta", "") or ""
            elif etype == "conversation.item.input_audio_transcription.completed":
                with self._state_lock:
                    if self._user_transcript and not self._user_transcript.endswith("\n"):
                        self._user_transcript += "\n"
            elif etype == "error":
                err = getattr(event, "error", None)
                msg = getattr(err, "message", None) or repr(err)
                logger.warning("OpenAI Realtime API error: %s", msg)
                with self._state_lock:
                    self._error = msg

    def _push_response_audio(self, pcm: bytes) -> None:
        if len(pcm) < 2:
            return
        if len(pcm) % 2:
            pcm = pcm[:-1]
        samples = np.frombuffer(pcm, dtype=np.int16)
        if samples.size == 0:
            return
        frame = av.AudioFrame.from_ndarray(
            samples[np.newaxis, :], format="s16", layout="mono"
        )
        frame.sample_rate = OPENAI_REALTIME_SAMPLE_RATE
        if self._on_audio_output is not None:
            self._on_audio_output()
        self._pipeline.push_audio_frame(frame)


parser = argparse.ArgumentParser()
parser.add_argument("--device", default=os.environ.get("ARTALK_DEVICE", "cuda"), type=str)
parser.add_argument("--render-res", default=int(os.environ.get("ARTALK_RENDER_RES", "512")), type=int)
parser.add_argument(
    "--render-batch-size",
    default=int(os.environ.get("ARTALK_RENDER_BATCH_SIZE", "8")),
    type=int,
)
parser.add_argument("--asset-dir", default=os.environ.get("ARTALK_ASSET_DIR"), type=str)
parser.add_argument("--gagavatar-asset-dir", default=os.environ.get("GAGAVATAR_ASSET_DIR"), type=str)
parser.add_argument("--gagavatar-model-path", default=os.environ.get("GAGAVATAR_MODEL_PATH"), type=str)
parser.add_argument("--gagavatar-tracked-path", default=os.environ.get("GAGAVATAR_TRACKED_PATH"), type=str)
parser.add_argument(
    "--gagavatar-flame-model-path",
    default=os.environ.get("GAGAVATAR_FLAME_MODEL_PATH"),
    type=str,
)
args, _ = parser.parse_known_args()

artalk_assets = ARTalkAssets.resolve(root=args.asset_dir)
gagavatar_assets = resolve_gagavatar_assets(args, artalk_assets)
asset_dir = artalk_assets.root
tracked_path = gagavatar_assets.tracked_path
model_path = gagavatar_assets.model_path
flame_model_path = gagavatar_assets.flame_model_path

st.set_page_config(page_title="ARTalk Realtime", page_icon=":speech_balloon:")
st.title("ARTalk Realtime")
st.caption(
    "Speak into the microphone — the avatar starts moving "
    "~4 seconds later (model chunk floor)."
)

try:
    artalk_runtime, mesh_renderer = load_artalk_runtime(
        args.device,
        args.render_res,
        str(asset_dir),
    )
except Exception as exc:
    st.error(f"Failed to initialize ARTalk runtime: {exc}")
    st.stop()

model = artalk_runtime.model
flame_model = artalk_runtime.flame_model

with st.sidebar:
    gagavatar_ids = list_gagavatar_ids(str(tracked_path) if tracked_path else None)
    style_ids = list_style_ids(str(asset_dir))
    if not gagavatar_ids:
        with st.expander("GAGAvatar assets", expanded=True):
            st.write(f"Asset dir: `{asset_dir}`")
            st.write(f"Tracked: `{tracked_path}`")
            st.write(f"Tracked exists: `{bool(tracked_path and tracked_path.exists())}`")
            st.write(f"Model: `{model_path}`")
            st.write(f"Model exists: `{bool(model_path and model_path.exists())}`")
            st.write(f"FLAME: `{flame_model_path}`")
            st.write(f"FLAME exists: `{bool(flame_model_path and flame_model_path.exists())}`")
        st.warning(
            "No GAGAvatar avatars were found. Set ARTALK_ASSET_DIR or pass "
            "--asset-dir to an asset tree containing GAGAvatar/tracked.pt."
        )
    appearance_options = [DEFAULT_APPEARANCE] + [f"gagavatar:{avatar_id}" for avatar_id in gagavatar_ids]
    appearance = st.selectbox("Appearance", appearance_options, index=0)
    default_style_index = (
        style_ids.index("natural_0") + 1 if "natural_0" in style_ids else 0
    )
    style_id = st.selectbox(
        "Style",
        [DEFAULT_STYLE] + style_ids,
        index=default_style_index,
    )
    mode = st.radio("Mode", ["Loopback", "Interactive"], horizontal=True)
    api_key = ""
    realtime_model = DEFAULT_REALTIME_MODEL
    realtime_voice = DEFAULT_REALTIME_VOICE
    realtime_instructions = DEFAULT_REALTIME_INSTRUCTIONS
    if mode == "Interactive":
        st.header("OpenAI Realtime")
        api_key = get_secret("OPENAI_API_KEY")
        if api_key:
            st.success("Secret loaded.")
        else:
            st.warning("Secret is not configured.")
        realtime_model = st.text_input("Model", value=DEFAULT_REALTIME_MODEL)
        realtime_voice = st.selectbox(
            "Voice",
            REALTIME_VOICES,
            index=REALTIME_VOICES.index(DEFAULT_REALTIME_VOICE),
        )
        realtime_instructions = st.text_area(
            "Instructions",
            value=DEFAULT_REALTIME_INSTRUCTIONS,
            height=120,
        )


PIPELINE_KEY = "artalk_pipeline"
PIPELINE_CONFIG_KEY = "artalk_pipeline_config"
SILENCE_PUMP_KEY = "artalk_silence_pump"
SILENCE_PUMP_CONFIG_KEY = "artalk_silence_pump_config"
BRIDGE_KEY = "openai_realtime_bridge"
BRIDGE_CONFIG_KEY = "openai_realtime_bridge_config"
BRIDGE_SHUTDOWN_OBSERVER_KEY = "openai_realtime_bridge_shutdown_observer"


def metric_value(counters: dict, key: str, default: float = 0.0) -> float:
    value = counters.get(key, default)
    return float(value) if value is not None else default


def elapsed_rate(counters: dict, count_key: str, start_key: str | None = None) -> float:
    now = metric_value(counters, "now_s")
    start = metric_value(counters, start_key, 0.0) if start_key else 0.0
    if not start:
        uptime = metric_value(counters, "uptime_s")
        start = now - uptime if now and uptime else 0.0
    elapsed = max(now - start, 1e-6) if now and start else 0.0
    if elapsed <= 0:
        return 0.0
    return metric_value(counters, count_key) / elapsed


def duration_row(label: str, durations: dict, key: str) -> dict:
    stat = durations.get(key, {})
    return {
        "stage": label,
        "count": int(stat.get("count", 0)),
        "last_ms": round(float(stat.get("last_ms", 0.0)), 1),
        "avg_ms": round(float(stat.get("avg_ms", 0.0)), 1),
        "max_ms": round(float(stat.get("max_ms", 0.0)), 1),
    }


def render_pipeline_diagnostics(pipeline: ARTalkPipeline) -> None:
    snapshot = pipeline.metrics.snapshot()
    counters = snapshot["counters"]
    durations = snapshot["durations"]

    st.subheader("Pipeline diagnostics")
    chunk_floor_s = metric_value(counters, "streamer_chunk_floor_s")
    first_motion_latency_s = counters.get("first_motion_latency_s")
    latency_text = (
        f"{float(first_motion_latency_s):.2f}s"
        if first_motion_latency_s is not None
        else "waiting"
    )
    rendered_fps = metric_value(counters, "last_render_chunk_fps")
    cumulative_rendered_fps = elapsed_rate(counters, "rendered_frames", "first_motion_s")
    video_callback_fps = elapsed_rate(counters, "video_callbacks", "first_video_callback_s")
    video_real_fps = elapsed_rate(counters, "video_frames_served", "first_video_callback_s")
    video_placeholder_fps = elapsed_rate(
        counters,
        "video_placeholder_frames",
        "first_video_callback_s",
    )
    audio_callback_fps = elapsed_rate(counters, "audio_callbacks", "first_audio_callback_s")

    cols = st.columns(4)
    cols[0].metric("Model floor", f"{chunk_floor_s:.2f}s")
    cols[1].metric("First motion", latency_text)
    cols[2].metric("Render chunk FPS", f"{rendered_fps:.1f}")
    cols[3].metric("Video callback FPS", f"{video_callback_fps:.1f}")

    stage_rows = [
        duration_row("Resample input", durations, "resample"),
        duration_row("ARTalk streamer feed", durations, "artalk_streamer_feed"),
        duration_row("Savgol smoother", durations, "smoother_feed"),
        duration_row("Renderer warm-up", durations, "renderer_warmup"),
        duration_row("Warm-up prepare", durations, "warmup_avatar_prepare_frame"),
        duration_row("Warm-up forward", durations, "warmup_avatar_forward_model"),
        duration_row("Warm-up GPU copy", durations, "warmup_avatar_gpu_to_cpu_copy"),
        duration_row("Warm-up RGB convert", durations, "warmup_rgb_tensor_to_numpy"),
        duration_row("Avatar prepare frame", durations, "avatar_prepare_frame"),
        duration_row("Avatar forward model", durations, "avatar_forward_model"),
        duration_row("Avatar GPU to CPU copy", durations, "avatar_gpu_to_cpu_copy"),
        duration_row("Avatar render frame", durations, "avatar_render_frame"),
        duration_row("RGB tensor to ndarray", durations, "rgb_tensor_to_numpy"),
        duration_row("Avatar prepare batch", durations, "avatar_prepare_batch"),
        duration_row("Avatar forward batch", durations, "avatar_forward_batch"),
        duration_row("Avatar GPU to CPU batch", durations, "avatar_gpu_to_cpu_batch"),
        duration_row("Avatar render batch", durations, "avatar_render_batch"),
        duration_row("RGB batch to ndarray", durations, "rgb_batch_to_numpy"),
        duration_row("Render chunk total", durations, "render_chunk_total"),
    ]
    st.dataframe(
        stage_rows,
        hide_index=True,
        width="stretch",
        key="pipeline_stage_metrics",
    )

    queue_cols = st.columns(4)
    queue_cols[0].metric("Audio in queue", int(metric_value(counters, "audio_in_queue_depth")))
    queue_cols[1].metric("Video queue", int(metric_value(counters, "video_queue_depth")))
    queue_cols[2].metric(
        "Audio out buffer",
        f"{metric_value(counters, 'audio_out_buffer_samples') / 16000.0:.2f}s",
    )
    queue_cols[3].metric(
        "Streamer buffer",
        f"{metric_value(counters, 'streamer_buffer_samples') / 16000.0:.2f}s",
    )

    output_rows = [
        {
            "name": "audio frames pushed",
            "value": int(metric_value(counters, "audio_frames_pushed")),
        },
        {
            "name": "audio samples fed",
            "value": int(metric_value(counters, "audio_samples_fed_to_streamer")),
        },
        {
            "name": "silence samples queued",
            "value": int(metric_value(counters, "silence_pump_samples")),
        },
        {
            "name": "silence pump chunks",
            "value": int(metric_value(counters, "silence_pump_chunks")),
        },
        {
            "name": "silence pump skips",
            "value": int(metric_value(counters, "silence_pump_backpressure_skips")),
        },
        {
            "name": "silence pump input-queue skips",
            "value": int(metric_value(counters, "silence_pump_input_queue_skips")),
        },
        {
            "name": "silence pump worker-busy skips",
            "value": int(metric_value(counters, "silence_pump_worker_busy_skips")),
        },
        {
            "name": "silence pump recent-input skips",
            "value": int(metric_value(counters, "silence_pump_recent_input_skips")),
        },
        {
            "name": "silence pump pacing skips",
            "value": int(metric_value(counters, "silence_pump_pacing_skips")),
        },
        {
            "name": "silence pump input idle seconds",
            "value": round(metric_value(counters, "silence_pump_input_idle_s"), 3),
        },
        {
            "name": "silence pump idle threshold",
            "value": round(metric_value(counters, "silence_pump_idle_seconds"), 3),
        },
        {
            "name": "silence pump audio buffer",
            "value": int(metric_value(counters, "silence_pump_audio_buffer_samples")),
        },
        {
            "name": "silence pump video queue",
            "value": int(metric_value(counters, "silence_pump_video_queue_depth")),
        },
        {
            "name": "silence pump worker busy",
            "value": int(metric_value(counters, "silence_pump_worker_busy")),
        },
        {
            "name": "motion chunks",
            "value": int(metric_value(counters, "motion_chunks_produced")),
        },
        {
            "name": "motion frames",
            "value": int(metric_value(counters, "motion_frames_produced")),
        },
        {
            "name": "smoothed frames",
            "value": int(metric_value(counters, "smoothed_frames_produced")),
        },
        {
            "name": "render batch size",
            "value": int(metric_value(counters, "render_batch_size")),
        },
        {
            "name": "render batches",
            "value": int(metric_value(counters, "render_batches")),
        },
        {
            "name": "render batch frames",
            "value": int(metric_value(counters, "render_batch_frames")),
        },
        {
            "name": "last render chunk frames",
            "value": int(metric_value(counters, "last_render_chunk_frames")),
        },
        {
            "name": "last render chunk seconds",
            "value": round(metric_value(counters, "last_render_chunk_s"), 3),
        },
        {
            "name": "last render media seconds",
            "value": round(metric_value(counters, "last_render_chunk_media_s"), 3),
        },
        {
            "name": "render realtime ratio",
            "value": round(metric_value(counters, "last_render_realtime_ratio"), 3),
        },
        {
            "name": "last audio samples emitted",
            "value": int(metric_value(counters, "last_audio_samples_emitted")),
        },
        {
            "name": "audio playback started",
            "value": int(metric_value(counters, "audio_playback_started")),
        },
        {
            "name": "audio prebuffer samples",
            "value": int(metric_value(counters, "output_audio_prebuffer_samples")),
        },
        {
            "name": "output segment min frames",
            "value": int(metric_value(counters, "output_segment_min_frames")),
        },
        {
            "name": "synced audio samples served",
            "value": int(metric_value(counters, "synced_audio_samples_served")),
        },
        {
            "name": "synced audio frame index",
            "value": int(metric_value(counters, "synced_audio_frame_index")),
        },
        {
            "name": "cumulative rendered FPS",
            "value": round(cumulative_rendered_fps, 1),
        },
        {
            "name": "rendered frames",
            "value": int(metric_value(counters, "rendered_frames")),
        },
        {
            "name": "output segments",
            "value": int(metric_value(counters, "output_segments_published")),
        },
        {
            "name": "output segment frames",
            "value": int(metric_value(counters, "output_segment_frames")),
        },
        {
            "name": "output segment audio samples",
            "value": int(metric_value(counters, "output_segment_audio_samples")),
        },
        {
            "name": "last output segment frames",
            "value": int(metric_value(counters, "last_output_segment_frames")),
        },
        {
            "name": "last output segment audio samples",
            "value": int(metric_value(counters, "last_output_segment_audio_samples")),
        },
        {
            "name": "video callbacks",
            "value": int(metric_value(counters, "video_callbacks")),
        },
        {
            "name": "video callback FPS",
            "value": round(video_callback_fps, 1),
        },
        {
            "name": "video frames served",
            "value": int(metric_value(counters, "video_frames_served")),
        },
        {
            "name": "video real-frame FPS",
            "value": round(video_real_fps, 1),
        },
        {
            "name": "video placeholders",
            "value": int(metric_value(counters, "video_placeholder_frames")),
        },
        {
            "name": "video placeholder FPS",
            "value": round(video_placeholder_fps, 1),
        },
        {
            "name": "video frames dropped",
            "value": int(metric_value(counters, "video_frames_dropped")),
        },
        {
            "name": "video frames dropped for sync",
            "value": int(metric_value(counters, "video_frames_dropped_for_sync")),
        },
        {
            "name": "video target frame index",
            "value": int(metric_value(counters, "last_video_target_frame_index")),
        },
        {
            "name": "video frame index served",
            "value": int(metric_value(counters, "last_video_frame_index_served")),
        },
        {
            "name": "video frame index enqueued",
            "value": int(metric_value(counters, "last_video_frame_index_enqueued")),
        },
        {
            "name": "audio callbacks",
            "value": int(metric_value(counters, "audio_callbacks")),
        },
        {
            "name": "audio callback FPS",
            "value": round(audio_callback_fps, 1),
        },
        {
            "name": "audio frames served",
            "value": int(metric_value(counters, "audio_frames_served")),
        },
        {
            "name": "audio pre-playback silence frames",
            "value": int(metric_value(counters, "audio_preplayback_silence_frames")),
        },
        {
            "name": "audio short-buffer frames",
            "value": int(metric_value(counters, "audio_short_buffer_frames")),
        },
        {
            "name": "audio playback underrun frames",
            "value": int(metric_value(counters, "audio_playback_underrun_frames")),
        },
        {
            "name": "input audio time",
            "value": round(metric_value(counters, "last_input_audio_time_s"), 3),
        },
        {
            "name": "video source time",
            "value": round(metric_value(counters, "last_video_source_time_s"), 3),
        },
        {
            "name": "audio source time",
            "value": round(metric_value(counters, "last_audio_source_time_s"), 3),
        },
    ]
    st.dataframe(
        output_rows,
        hide_index=True,
        width="stretch",
        key="pipeline_output_counters",
    )


def stop_silence_pump() -> None:
    pump = st.session_state.pop(SILENCE_PUMP_KEY, None)
    if pump is not None:
        pump.stop()
    st.session_state.pop(SILENCE_PUMP_CONFIG_KEY, None)


def get_silence_pump(pipeline: ARTalkPipeline) -> PipelineSilencePump:
    config = id(pipeline)
    pump = st.session_state.get(SILENCE_PUMP_KEY)
    if pump is not None and st.session_state.get(SILENCE_PUMP_CONFIG_KEY) != config:
        pump.stop()
        pump = None
    if pump is None:
        pump = PipelineSilencePump(pipeline)
        pump.start()
        st.session_state[SILENCE_PUMP_KEY] = pump
        st.session_state[SILENCE_PUMP_CONFIG_KEY] = config
    return pump


def stop_bridge() -> None:
    bridge = st.session_state.pop(BRIDGE_KEY, None)
    if bridge is not None:
        bridge.stop()
    observer = st.session_state.pop(BRIDGE_SHUTDOWN_OBSERVER_KEY, None)
    if isinstance(observer, SessionShutdownObserver):
        observer.stop()
    st.session_state.pop(BRIDGE_CONFIG_KEY, None)


def split_appearance(value: str) -> tuple[str, str | None]:
    if value == DEFAULT_APPEARANCE:
        return "mesh", None
    source, avatar_id = value.split(":", 1)
    if source != "gagavatar" or not avatar_id:
        raise ValueError(f"Unknown appearance: {value}")
    return source, avatar_id


def get_pipeline() -> ARTalkPipeline:
    renderer_mode, avatar_id = split_appearance(appearance)
    style_motion = load_style_motion(str(asset_dir), style_id)
    render_res = args.render_res if renderer_mode == "mesh" else 512
    gagavatar = None
    gagavatar_flame = None
    if renderer_mode == "gagavatar":
        if model_path is None:
            raise RuntimeError("GAGAVATAR_MODEL_PATH is required for GAGAvatar appearance.")
        gagavatar, gagavatar_flame = load_gagavatar(
            args.device,
            str(model_path),
            str(tracked_path) if tracked_path else None,
            str(flame_model_path) if flame_model_path else None,
        )
    config = (
        args.device,
        mode,
        render_res,
        args.render_batch_size,
        appearance,
        style_id,
        str(asset_dir),
        str(model_path) if model_path else None,
        str(tracked_path) if tracked_path else None,
    )
    pipeline = st.session_state.get(PIPELINE_KEY)
    if pipeline is not None and st.session_state.get(PIPELINE_CONFIG_KEY) != config:
        stop_silence_pump()
        pipeline.stop()
        pipeline = None
    if pipeline is None:
        pipeline = ARTalkPipeline(
            model=model,
            flame_model=flame_model,
            mesh_renderer=mesh_renderer,
            device=args.device,
            style_motion=style_motion,
            render_res=render_res,
            render_batch_size=args.render_batch_size,
            renderer_mode=renderer_mode,
            gagavatar=gagavatar,
            gagavatar_flame=gagavatar_flame,
            shape_id=avatar_id,
        )
        st.session_state[PIPELINE_KEY] = pipeline
        st.session_state[PIPELINE_CONFIG_KEY] = config
    return pipeline


def stop_pipeline() -> None:
    stop_silence_pump()
    pipeline = st.session_state.pop(PIPELINE_KEY, None)
    if pipeline is not None:
        pipeline.stop()
    st.session_state.pop(PIPELINE_CONFIG_KEY, None)


if mode == "Loopback":
    stop_bridge()

if mode == "Interactive" and not api_key:
    stop_bridge()
    stop_pipeline()
    st.info("Configure the secret to use Interactive mode.")
    st.stop()


try:
    pipeline = get_pipeline()
except Exception as exc:
    st.error(f"Failed to initialize ARTalk avatar pipeline: {exc}")
    st.stop()

silence_pump = get_silence_pump(pipeline)


def get_bridge() -> OpenAIRealtimeBridge:
    config = (
        api_key,
        realtime_model,
        realtime_voice,
        realtime_instructions,
        id(pipeline),
    )
    bridge = st.session_state.get(BRIDGE_KEY)
    if bridge is not None and st.session_state.get(BRIDGE_CONFIG_KEY) != config:
        stop_bridge()
        bridge = None
    if bridge is None:
        bridge = OpenAIRealtimeBridge(
            api_key=api_key,
            pipeline=pipeline,
            on_audio_output=silence_pump.mark_input,
            model=realtime_model,
            voice=realtime_voice,
            instructions=realtime_instructions,
        )
        st.session_state[BRIDGE_KEY] = bridge
        st.session_state[BRIDGE_CONFIG_KEY] = config
        st.session_state[BRIDGE_SHUTDOWN_OBSERVER_KEY] = SessionShutdownObserver(
            bridge.stop
        )
    return bridge


bridge = get_bridge() if mode == "Interactive" else None
if bridge is not None and not bridge.is_running:
    with st.spinner("Connecting to OpenAI Realtime..."):
        bridge.start()
        bridge.wait_until_connected(timeout=8.0)
    snap = bridge.snapshot()
    if snap["error"]:
        st.error(f"OpenAI Realtime API error: {snap['error']}")
    elif not snap["connected"]:
        st.warning("OpenAI Realtime is still connecting. Wait a moment before START.")


def on_loopback_audio_frame(frame: av.AudioFrame) -> None:
    silence_pump.mark_input()
    pipeline.push_audio_frame(frame)


def on_interactive_audio_frame(frame: av.AudioFrame) -> None:
    if bridge is not None:
        bridge.push_input(frame)


def on_audio_ended() -> None:
    stop_bridge()
    stop_pipeline()


video_source_track = create_video_source_track(
    pipeline.video_source_callback,
    key=f"artalk_video_source_{mode.lower()}",
    fps=25,
)
audio_source_track = create_audio_source_track(
    pipeline.audio_source_callback,
    key=f"artalk_audio_source_{mode.lower()}",
    sample_rate=16000,
    ptime=0.020,
)
audio_sink_track = create_audio_sink_track(
    on_loopback_audio_frame if mode == "Loopback" else on_interactive_audio_frame,
    key=f"artalk_audio_sink_{mode.lower()}",
    on_ended=on_audio_ended,
)

streamer_key = f"artalk_{mode.lower()}"


def on_change() -> None:
    ctx = st.session_state.get(streamer_key)
    if ctx is None:
        return
    if mode == "Interactive" and ctx.state.playing and bridge is not None:
        bridge.start()
    if not ctx.state.playing and not ctx.state.signalling:
        if bridge is not None:
            bridge.stop()
        stop_pipeline()
        video_source_track.stop()
        audio_source_track.stop()

if bridge is not None:

    @st.fragment(run_every="500ms")
    def render_interactive_status() -> None:
        snap = bridge.snapshot()
        if snap["error"]:
            st.error(f"OpenAI Realtime API error: {snap['error']}")
        with st.container(height=260, border=True):
            if snap["assistant"]:
                st.markdown(snap["assistant"])
            else:
                st.caption("Waiting for response...")


def render_webrtc_component() -> None:
    webrtc_streamer(
        key=streamer_key,
        mode=WebRtcMode.SENDRECV,
        source_video_track=video_source_track,
        source_audio_track=audio_source_track,
        sink_audio_track=audio_sink_track,
        media_stream_constraints={"audio": True, "video": False},
        on_change=on_change,
    )


if bridge is not None:
    video_col, preview_col = st.columns([2, 1])
    with video_col:
        render_webrtc_component()
    with preview_col:
        render_interactive_status()
else:
    render_webrtc_component()


@st.fragment(run_every="500ms")
def render_diagnostics_fragment() -> None:
    with st.expander("Pipeline diagnostics", expanded=(mode == "Loopback")):
        render_pipeline_diagnostics(pipeline)


render_diagnostics_fragment()
