#!/usr/bin/env python

"""Standalone Streamlit + streamlit-webrtc realtime demo for ARTalk.

This app is intentionally outside the ARTalk repository. It expects ARTalk and
GAGAvatar to be installed as Python packages and locates model assets through
environment variables.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import signal
import threading
import time
from pathlib import Path

import av
import streamlit as st
from artalk.assets import ARTalkAssets
from artalk.realtime_pipeline import ARTalkPipeline
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_sink_track,
    create_audio_source_track,
    create_video_source_track,
    webrtc_streamer,
)

from artalk_streamlit_realtime.assets import resolve_gagavatar_assets
from artalk_streamlit_realtime.avatar_registry import (
    AvatarRegistrationError,
    UserAvatarRegistry,
    slugify_avatar_id,
)
from artalk_streamlit_realtime.config import (
    ARTALK_FPS,
    ARTALK_SAMPLE_RATE,
    CUSTOM_ENDPOINT_LABEL,
    DEFAULT_APPEARANCE,
    DEFAULT_LIVE_DELEGATION_MODEL,
    DEFAULT_LIVE_INSTRUCTIONS,
    DEFAULT_LIVE_MODEL,
    DEFAULT_LIVE_VOICE,
    DEFAULT_PERSONAPLEX_TEXT_PROMPT,
    DEFAULT_PERSONAPLEX_URL,
    DEFAULT_PERSONAPLEX_VOICE,
    DEFAULT_REALTIME_INSTRUCTIONS,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_VOICE,
    DEFAULT_REALTIME_WEBSOCKET_BASE_URL,
    DEFAULT_STYLE,
    LIVE_VOICES,
    PERSONAPLEX_VOICES,
    REALTIME_ENDPOINT_PRESETS,
    REALTIME_VOICES,
    parse_args,
)
from artalk_streamlit_realtime.diagnostics import (
    render_event_log_panel,
    render_gc_panel,
    render_pipeline_diagnostics,
    render_profiler_panel,
    save_diagnostics_snapshot,
)
from artalk_streamlit_realtime.event_log import PipelineEventWatcher, SessionEventLog
from artalk_streamlit_realtime.ice import STUN_ONLY, resolve_rtc_configuration
from artalk_streamlit_realtime.sessions import current_session_id, slots
from artalk_streamlit_realtime.artalk1s import ARTalk1sStreamer
from artalk_streamlit_realtime.fallingwater import FallingwaterStreamer
from artalk_streamlit_realtime.framemodel import FrameModelStreamer
from artalk_streamlit_realtime.gc_probe import freeze_loaded_objects, gc_pause_probe
from artalk_streamlit_realtime.hang_watchdog import script_hang_watchdog
from artalk_streamlit_realtime.loop_watchdog import loop_stall_watchdog
from artalk_streamlit_realtime.openai_bridge import OpenAIRealtimeBridge
from artalk_streamlit_realtime.openai_live_bridge import OpenAILiveBridge
from artalk_streamlit_realtime.personaplex_bridge import PersonaPlexBridge
from artalk_streamlit_realtime.runtime import (
    list_gagavatar_ids,
    list_style_ids,
    load_artalk_runtime,
    load_artalk1s_streamer_model,
    load_fallingwater_streamer_model,
    load_frame_streamer_model,
    load_gagavatar,
    load_style_motion,
)
from artalk_streamlit_realtime.silence import PipelineSilencePump
from artalk_streamlit_realtime.streamlit_patches import (
    disable_streamlit_source_watcher,
)

PIPELINE_KEY = "artalk_pipeline"
PIPELINE_CONFIG_KEY = "artalk_pipeline_config"
# A rerun can start while the previous script run is still inside pipeline
# construction (nothing in there is an st.* call the runner could stop at),
# and concurrent constructions race in process-global CUDA state: graph
# captures against renders, and torch's non-thread-safe lazy linalg init.
_PIPELINE_BUILD_LOCK = threading.Lock()
SILENCE_PUMP_KEY = "artalk_silence_pump"
SILENCE_PUMP_CONFIG_KEY = "artalk_silence_pump_config"
BACKEND_OPENAI = "OpenAI Realtime"
BACKEND_OPENAI_LIVE = "OpenAI Live"
BACKEND_PERSONAPLEX = "PersonaPlex"
BRIDGE_KEY = "openai_realtime_bridge"
BRIDGE_CONFIG_KEY = "openai_realtime_bridge_config"
EVENT_LOG_KEY = "artalk_event_log"
EVENT_WATCHER_KEY = "artalk_event_watcher"
EVENT_WATCHER_CONFIG_KEY = "artalk_event_watcher_config"


def get_secret(name: str, default: str = "") -> str:
    """A secret from Streamlit's secrets file, else from the environment,
    which is how a Space delivers its secrets."""
    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        value = None
    if value is None:
        value = os.environ.get(name)
    return str(value) if value else default


def api_key_input(secret_name: str) -> str:
    """The configured secret, or a key typed into this browser session."""
    api_key = get_secret(secret_name)
    if api_key:
        st.success(f"`{secret_name}` loaded.")
        return api_key
    return st.text_input(
        secret_name,
        type="password",
        key=f"api_key_{secret_name}",
        help="Kept for this browser session only. Usage is billed to this key.",
    ).strip()


def register_avatar_with_progress(
    registry: UserAvatarRegistry,
    avatar_id: str,
    image_bytes: bytes,
    suffix: str,
    reference_s: float | None,
) -> dict:
    """Run the blocking tracking subprocess in a worker thread while the
    script thread animates a progress bar against the measured duration of
    the previous run (or elapsed time only, on the first run)."""
    outcome: dict = {}

    def run() -> None:
        try:
            outcome["entry"] = registry.register(avatar_id, image_bytes, suffix)
        except Exception as exc:
            outcome["error"] = exc

    worker = threading.Thread(target=run, name="AvatarTracking", daemon=True)
    started = time.monotonic()
    worker.start()
    bar = st.progress(0.0)
    while worker.is_alive():
        script_hang_watchdog.beat()
        elapsed = time.monotonic() - started
        if reference_s is not None:
            bar.progress(
                min(elapsed / reference_s, 0.99),
                text=(
                    f"Tracking face... {elapsed:.0f} s "
                    f"(took {reference_s:.0f} s last time)"
                ),
            )
        else:
            bar.progress(
                min(elapsed / 60.0, 0.99),
                text=f"Tracking face... {elapsed:.0f} s (first run, may take a minute)",
            )
        time.sleep(0.25)
    worker.join()
    if "error" in outcome:
        bar.empty()
        raise outcome["error"]
    bar.progress(
        1.0,
        text=f"Tracking face... done in {time.monotonic() - started:.0f} s",
    )
    return outcome["entry"]


def split_appearance(value: str) -> tuple[str, str | None]:
    if value == DEFAULT_APPEARANCE:
        return "mesh", None
    source, avatar_id = value.split(":", 1)
    if source != "gagavatar" or not avatar_id:
        raise ValueError(f"Unknown appearance: {value}")
    return source, avatar_id


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


def get_event_log() -> SessionEventLog:
    log = st.session_state.get(EVENT_LOG_KEY)
    if log is None:
        log = SessionEventLog()
        st.session_state[EVENT_LOG_KEY] = log
    return log


def stop_event_watcher() -> None:
    watcher = st.session_state.pop(EVENT_WATCHER_KEY, None)
    if watcher is not None:
        watcher.stop()
    st.session_state.pop(EVENT_WATCHER_CONFIG_KEY, None)


def get_event_watcher(
    pipeline: ARTalkPipeline, event_log: SessionEventLog
) -> PipelineEventWatcher:
    config = id(pipeline)
    watcher = st.session_state.get(EVENT_WATCHER_KEY)
    if watcher is not None and st.session_state.get(EVENT_WATCHER_CONFIG_KEY) != config:
        watcher.stop()
        watcher = None
    if watcher is None:
        watcher = PipelineEventWatcher(pipeline, event_log)
        watcher.start()
        st.session_state[EVENT_WATCHER_KEY] = watcher
        st.session_state[EVENT_WATCHER_CONFIG_KEY] = config
    return watcher


def stop_bridge() -> None:
    bridge = st.session_state.pop(BRIDGE_KEY, None)
    if bridge is not None:
        bridge.stop()
    st.session_state.pop(BRIDGE_CONFIG_KEY, None)


def stop_pipeline() -> None:
    stop_silence_pump()
    stop_event_watcher()
    pipeline = st.session_state.pop(PIPELINE_KEY, None)
    if pipeline is not None:
        pipeline.stop()
    st.session_state.pop(PIPELINE_CONFIG_KEY, None)
    slots.release(current_session_id())


def main() -> None:
    gc_pause_probe.install()
    # `kill -USR1 <pid>` dumps every thread's Python stack to stderr (the
    # launcher terminal) — the first thing to reach for when the app hangs.
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    disable_streamlit_source_watcher()
    try:
        from streamlit_webrtc.eventloop import get_global_event_loop

        loop_stall_watchdog.install(get_global_event_loop())
    except Exception:
        logging.getLogger(__name__).warning(
            "[loop-watchdog] not armed", exc_info=True
        )
    args = parse_args()
    slots.limit = args.max_sessions
    artalk_assets = ARTalkAssets.resolve(root=args.asset_dir)
    gagavatar_assets = resolve_gagavatar_assets(args, artalk_assets)
    asset_dir = artalk_assets.root
    tracked_path = gagavatar_assets.tracked_path
    model_path = gagavatar_assets.model_path
    flame_model_path = gagavatar_assets.flame_model_path

    st.set_page_config(
        page_title="ARTalk Realtime",
        page_icon=":speech_balloon:",
        layout="wide",
    )
    st.title("ARTalk Realtime")
    chunk_caption = st.empty()

    try:
        artalk_runtime, mesh_renderer = load_artalk_runtime(
            args.device,
            args.render_res,
            str(asset_dir),
            audio_encoder=args.artalk_audio_encoder,
            checkpoint_path=args.artalk_checkpoint,
        )
    except Exception as exc:
        st.error(f"Failed to initialize ARTalk runtime: {exc}")
        st.stop()

    model = artalk_runtime.model
    flame_model = artalk_runtime.flame_model

    user_registry = UserAvatarRegistry(
        args.user_avatar_dir,
        track_python=args.gagavatar_track_python,
        track_dir=args.gagavatar_track_dir,
        track_device=args.gagavatar_track_device,
    )

    with st.sidebar:
        motion_labels = {"artalk": "ARTalk (4 s chunks)", "artalk1s": "ARTalk 1 s"}
        if args.motion_model in motion_labels and args.artalk1s_train_code_dir and args.artalk1s_checkpoint:
            motion_model = st.radio(
                "Motion model",
                list(motion_labels),
                index=list(motion_labels).index(args.motion_model),
                format_func=motion_labels.__getitem__,
                horizontal=True,
                help=(
                    "The 1 s model starts speaking about three seconds sooner "
                    "and keeps the head still. Switching rebuilds the pipeline."
                ),
            )
        else:
            motion_model = args.motion_model
        if motion_model == "fallingwater":
            st.caption(
                "Fallingwater model: "
                f"`{Path(args.fallingwater_checkpoint or '?').name}`"
            )
        elif motion_model == "artalk1s":
            st.caption(
                "ARTalk 1s model: "
                f"`{Path(args.artalk1s_checkpoint or '?').name}`"
            )
        elif motion_model == "frame":
            st.caption(
                "Frame model: "
                f"`{Path(args.frame_checkpoint or '?').name}`"
            )
        else:
            st.caption(f"ARTalk model: `{Path(artalk_runtime.checkpoint_path).name}`")
        gagavatar_ids = list_gagavatar_ids(str(tracked_path) if tracked_path else None)
        user_avatar_ids = user_registry.list_ids()
        style_ids = list_style_ids(str(asset_dir))
        if not gagavatar_ids and not user_avatar_ids:
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

        appearance_options = [
            DEFAULT_APPEARANCE,
            *[f"gagavatar:{avatar_id}" for avatar_id in gagavatar_ids],
            *[f"gagavatar:{avatar_id}" for avatar_id in user_avatar_ids],
        ]
        appearance = st.selectbox("Appearance", appearance_options, index=0)
        default_style_index = (
            style_ids.index("natural_0") + 1 if "natural_0" in style_ids else 0
        )
        style_id = st.selectbox(
            "Style",
            [DEFAULT_STYLE, *style_ids],
            index=default_style_index,
        )
        modes = ["Loopback", "Interactive"]
        mode = st.radio(
            "Mode", modes, index=modes.index(args.default_mode.capitalize()), horizontal=True
        )
        mic_processing = st.toggle(
            "Mic echo cancellation & noise suppression",
            value=True,
            key="mic_processing",
            help=(
                "Browser-side echoCancellation / noiseSuppression / "
                "autoGainControl on the microphone. Echo cancellation is "
                "what stops the avatar's speaker output from looping back "
                "into the mic (and self-interrupting it). Takes effect on "
                "the next START."
            ),
        )

        with st.expander("Playback when rendering falls behind"):
            st.radio(
                "Underrun policy",
                ["continuous", "rebuffer"],
                index=["continuous", "rebuffer"].index(
                    args.output_underrun_policy
                ),
                key="underrun_policy",
                horizontal=True,
                help=(
                    "continuous resumes on the first complete audio frame, "
                    "staying closest to live; when the renderer is marginal "
                    "the buffer then oscillates around empty and both tracks "
                    "stutter. rebuffer waits for the buffer to refill, "
                    "trading latency for continuity. Applies immediately, so "
                    "the two can be compared within one conversation."
                ),
            )
            st.slider(
                "Refill target (s)",
                0.0,
                3.0,
                value=float(
                    args.output_rebuffer_seconds
                    if args.output_rebuffer_seconds is not None
                    else args.output_prebuffer_seconds
                ),
                step=0.25,
                key="rebuffer_seconds",
            )
            st.slider(
                "Max added latency (s)",
                0.0,
                10.0,
                value=float(args.max_added_latency_seconds),
                step=0.5,
                key="max_added_latency",
            )

        api_key = ""
        realtime_model = DEFAULT_REALTIME_MODEL
        realtime_voice = DEFAULT_REALTIME_VOICE
        realtime_instructions = DEFAULT_REALTIME_INSTRUCTIONS
        realtime_ws_base_url = DEFAULT_REALTIME_WEBSOCKET_BASE_URL
        backend = BACKEND_OPENAI
        live_model = DEFAULT_LIVE_MODEL
        live_voice = DEFAULT_LIVE_VOICE
        live_instructions = DEFAULT_LIVE_INSTRUCTIONS
        live_delegation_model = ""
        personaplex_url = DEFAULT_PERSONAPLEX_URL
        personaplex_voice = DEFAULT_PERSONAPLEX_VOICE
        personaplex_text_prompt = DEFAULT_PERSONAPLEX_TEXT_PROMPT
        if mode == "Interactive":
            st.header("Conversation backend")
            backend = st.radio(
                "Backend",
                [BACKEND_OPENAI, BACKEND_OPENAI_LIVE, BACKEND_PERSONAPLEX],
                horizontal=True,
                help=(
                    "Live and PersonaPlex are full duplex: they stream "
                    "continuously and have no turn boundaries, so there is no "
                    "barge-in truncation."
                ),
            )
        if mode == "Interactive" and backend == BACKEND_OPENAI_LIVE:
            api_key = api_key_input("OPENAI_API_KEY")
            # Keyed so the Realtime branch's identically labelled widgets
            # cannot be mistaken for these when the backend is switched.
            live_model = st.text_input(
                "Model", value=DEFAULT_LIVE_MODEL, key="live_model"
            ).strip()
            live_voice = st.selectbox(
                "Voice",
                LIVE_VOICES,
                index=LIVE_VOICES.index(DEFAULT_LIVE_VOICE),
                key="live_voice",
            )
            live_instructions = st.text_area(
                "Instructions",
                value=DEFAULT_LIVE_INSTRUCTIONS,
                height=120,
                key="live_instructions",
            )
            delegate = st.toggle(
                "Delegate tasks to a Responses backend",
                value=False,
                key="live_delegate",
                help=(
                    "Splitting the voice frontend from a backend that reasons "
                    "and calls tools is what separates Live from Realtime. The "
                    "backend model and its tool calls are billed on top of the "
                    "session's per-second charge; without it the avatar "
                    "answers from the Live model alone."
                ),
            )
            if delegate:
                live_delegation_model = st.text_input(
                    "Backend model",
                    value=DEFAULT_LIVE_DELEGATION_MODEL,
                    key="live_delegation_model",
                ).strip()
                st.caption("Web search is the only tool offered to the backend.")
        if mode == "Interactive" and backend == BACKEND_PERSONAPLEX:
            personaplex_url = st.text_input(
                "moshi server URL", value=DEFAULT_PERSONAPLEX_URL
            ).strip()
            personaplex_voice = st.selectbox(
                "Voice prompt",
                PERSONAPLEX_VOICES,
                index=PERSONAPLEX_VOICES.index(DEFAULT_PERSONAPLEX_VOICE),
            )
            personaplex_text_prompt = st.text_area(
                "Persona", value=DEFAULT_PERSONAPLEX_TEXT_PROMPT, height=120
            )
        if mode == "Interactive" and backend == BACKEND_OPENAI:
            preset_names = list(REALTIME_ENDPOINT_PRESETS) + [CUSTOM_ENDPOINT_LABEL]
            # An endpoint supplied through the environment is by definition not
            # one of the presets, so start on Custom and carry it in.
            endpoint_name = st.selectbox(
                "Endpoint",
                preset_names,
                index=preset_names.index(CUSTOM_ENDPOINT_LABEL)
                if DEFAULT_REALTIME_WEBSOCKET_BASE_URL
                else 0,
                help="Anything speaking the OpenAI Realtime protocol.",
            )
            if endpoint_name == CUSTOM_ENDPOINT_LABEL:
                realtime_ws_base_url = st.text_input(
                    "WebSocket base URL",
                    value=DEFAULT_REALTIME_WEBSOCKET_BASE_URL,
                    help="The SDK appends `/realtime`.",
                ).strip()
                secret_name = "OPENAI_API_KEY"
                endpoint_model = DEFAULT_REALTIME_MODEL
                endpoint_voices = tuple(REALTIME_VOICES)
            else:
                endpoint = REALTIME_ENDPOINT_PRESETS[endpoint_name]
                realtime_ws_base_url = endpoint.base_url
                secret_name = endpoint.secret
                endpoint_model = endpoint.model
                endpoint_voices = endpoint.voices
                if realtime_ws_base_url:
                    st.caption(f"`{realtime_ws_base_url}`")

            if secret_name:
                api_key = api_key_input(secret_name)
            else:
                st.info("This endpoint takes no credential.")
            # Keyed per endpoint so switching presets re-seeds these rather
            # than carrying the previous provider's names over.
            realtime_model = st.text_input(
                "Model",
                value=endpoint_model,
                key=f"realtime_model_{endpoint_name}",
                help="Sent in the connection URL, so a wrong name fails the handshake.",
            ).strip()
            if endpoint_voices:
                realtime_voice = st.selectbox(
                    "Voice",
                    endpoint_voices,
                    key=f"realtime_voice_{endpoint_name}",
                )
            else:
                realtime_voice = DEFAULT_REALTIME_VOICE
                st.caption("Voice is configured on the server.")
            realtime_instructions = st.text_area(
                "Instructions",
                value=DEFAULT_REALTIME_INSTRUCTIONS,
                height=120,
            )

        if user_registry.can_register:
            with st.expander("Register avatar"):
                registered_flash = st.session_state.pop("avatar_registered_flash", None)
                if registered_flash is not None:
                    elapsed_s = registered_flash.get("elapsed_s")
                    took = f" in {elapsed_s:.0f} s" if elapsed_s else ""
                    st.success(
                        f"Registered{took} — select "
                        f"`gagavatar:{registered_flash['avatar_id']}` under Appearance."
                    )
                    st.image(
                        registered_flash["vis_image"],
                        clamp=True,
                        caption=f"Tracked fit: {registered_flash['avatar_id']}",
                    )
                upload = st.file_uploader(
                    "Face image",
                    type=["jpg", "jpeg", "png"],
                    key="avatar_upload",
                )
                if upload is not None:
                    avatar_name = st.text_input(
                        "Avatar name",
                        value=Path(upload.name).stem,
                        key="avatar_name",
                    )
                    ctx = st.session_state.get(f"artalk_{mode.lower()}")
                    if (
                        ctx is not None
                        and ctx.state.playing
                        and user_registry.track_device.startswith("cuda")
                    ):
                        st.warning(
                            "A session is playing. Tracking shares the GPU and "
                            "may cause a brief hiccup or fail on low memory."
                        )
                    if st.button("Register", key="avatar_register"):
                        avatar_id = slugify_avatar_id(avatar_name)
                        if not avatar_id:
                            st.error("Avatar name is empty after sanitizing.")
                        else:
                            avatar_id = user_registry.allocate_id(
                                avatar_id, reserved=gagavatar_ids
                            )
                            suffix = Path(upload.name).suffix.lower() or ".png"
                            try:
                                entry = register_avatar_with_progress(
                                    user_registry,
                                    avatar_id,
                                    upload.getvalue(),
                                    suffix,
                                    user_registry.last_track_seconds(),
                                )
                            except AvatarRegistrationError as exc:
                                st.error(str(exc))
                            else:
                                # The Appearance selectbox above rendered before
                                # this handler ran; rerun so it includes the new
                                # avatar, carrying the confirmation across as a
                                # flash.
                                st.session_state["avatar_registered_flash"] = {
                                    "avatar_id": avatar_id,
                                    "vis_image": entry["vis_image"]
                                    .numpy()
                                    .transpose(1, 2, 0),
                                    "elapsed_s": user_registry.last_track_seconds(),
                                }
                                st.rerun()
            if user_avatar_ids:
                delete_id = st.selectbox(
                    "Delete registered avatar",
                    ["(select)", *user_avatar_ids],
                    key="avatar_delete_select",
                )
                if delete_id != "(select)" and st.button(
                    f"Delete {delete_id}", key="avatar_delete"
                ):
                    user_registry.delete(delete_id)
                    st.rerun()

    chunk_caption.caption(
        "Speak into the microphone — the avatar starts moving "
        f"~{1 if motion_model == 'artalk1s' else 4} seconds later (model chunk floor)."
    )

    def get_pipeline() -> ARTalkPipeline:
        renderer_mode, avatar_id = split_appearance(appearance)
        style_motion = load_style_motion(str(asset_dir), style_id)
        render_res = args.render_res if renderer_mode == "mesh" else 512
        gagavatar = None
        gagavatar_flame = None
        if renderer_mode == "gagavatar":
            if model_path is None:
                raise RuntimeError(
                    "GAGAVATAR_MODEL_PATH is required for GAGAvatar appearance."
                )
            gagavatar, gagavatar_flame = load_gagavatar(
                args.device,
                str(model_path),
                str(tracked_path) if tracked_path else None,
                str(flame_model_path) if flame_model_path else None,
                args.user_avatar_dir,
                autocast_dtype="float16" if args.renderer_fp16 else None,
                compile_mode="cuda-graph" if args.renderer_compile else None,
            )
        streamer = None
        if motion_model == "fallingwater":
            if not args.fallingwater_dir or not args.fallingwater_checkpoint:
                raise RuntimeError(
                    "--fallingwater-dir and --fallingwater-checkpoint are "
                    "required for --motion-model fallingwater."
                )
            streamer = FallingwaterStreamer(
                load_fallingwater_streamer_model(
                    args.fallingwater_dir,
                    args.fallingwater_checkpoint,
                    args.device,
                )
            )
        elif motion_model == "artalk1s":
            if not args.artalk1s_train_code_dir or not args.artalk1s_checkpoint:
                raise RuntimeError(
                    "--artalk1s-train-code-dir and --artalk1s-checkpoint are "
                    "required for --motion-model artalk1s."
                )
            streamer = ARTalk1sStreamer(
                load_artalk1s_streamer_model(
                    args.artalk1s_train_code_dir,
                    args.artalk1s_checkpoint,
                    args.device,
                ),
                style_motion=style_motion,
            )
        elif motion_model == "frame":
            if not args.frame_package_dir or not args.frame_checkpoint:
                raise RuntimeError(
                    "--frame-package-dir and --frame-checkpoint are required "
                    "for --motion-model frame."
                )
            streamer = FrameModelStreamer(
                load_frame_streamer_model(
                    args.frame_package_dir,
                    args.frame_checkpoint,
                    args.device,
                ),
                # The photoreal renderer consumes the model's native layout
                # (live eye pose); the mesh path still speaks the released
                # 106-dim layout.
                native_layout=renderer_mode == "gagavatar",
            )
        config = (
            args.device,
            args.artalk_audio_encoder,
            args.artalk_checkpoint,
            motion_model,
            args.fallingwater_checkpoint,
            args.artalk1s_checkpoint,
            args.frame_checkpoint,
            mode,
            render_res,
            args.render_batch_size,
            args.output_prebuffer_seconds,
            args.output_segment_seconds,
            args.output_underrun_policy,
            args.output_rebuffer_seconds,
            args.max_added_latency_seconds,
            args.renderer_stage_sync,
            args.render_uint8_gpu,
            args.renderer_fp16,
            args.renderer_compile,
            args.profile_trace_dir,
            args.profile_skip_chunks,
            args.profile_max_chunks,
            appearance,
            style_id,
            str(asset_dir),
            str(model_path) if model_path else None,
            str(tracked_path) if tracked_path else None,
        )
        with _PIPELINE_BUILD_LOCK:
            # A run that waited here while another run built the
            # pipeline must adopt it, not build a twin.
            pipeline = st.session_state.get(PIPELINE_KEY)
            # A stopped pipeline is one the slot reaper tore down while this
            # session was disconnected; it came back and needs a fresh one.
            if pipeline is not None and (
                st.session_state.get(PIPELINE_CONFIG_KEY) != config or pipeline.is_stopped
            ):
                stop_silence_pump()
                pipeline.stop()
                pipeline = None
            if pipeline is None:
                pipeline = ARTalkPipeline(
                    model=model,
                    streamer=streamer,
                    flame_model=flame_model,
                    mesh_renderer=mesh_renderer,
                    device=args.device,
                    style_motion=style_motion,
                    render_res=render_res,
                    render_batch_size=args.render_batch_size,
                    output_audio_prebuffer_seconds=args.output_prebuffer_seconds,
                    output_segment_seconds=args.output_segment_seconds,
                    output_underrun_policy=args.output_underrun_policy,
                    output_rebuffer_seconds=args.output_rebuffer_seconds,
                    max_added_latency_seconds=args.max_added_latency_seconds,
                    renderer_stage_sync=args.renderer_stage_sync,
                    renderer_output_uint8=args.render_uint8_gpu,
                    warm_key_extra=(
                        f"fp16={args.renderer_fp16},compile={args.renderer_compile},"
                        f"motion={motion_model}"
                    ),
                    profile_trace_dir=args.profile_trace_dir,
                    profile_skip_chunks=args.profile_skip_chunks,
                    profile_max_chunks=args.profile_max_chunks,
                    renderer_mode=renderer_mode,
                    gagavatar=gagavatar,
                    gagavatar_flame=gagavatar_flame,
                    shape_id=avatar_id,
                )
                st.session_state[PIPELINE_KEY] = pipeline
                st.session_state[PIPELINE_CONFIG_KEY] = config
                freeze_loaded_objects(force=True)
        return pipeline

    if mode == "Loopback":
        stop_bridge()

    # A Realtime endpoint may take no credential; Live is OpenAI's alone.
    needs_secret = not api_key and (
        (backend == BACKEND_OPENAI and not realtime_ws_base_url)
        or backend == BACKEND_OPENAI_LIVE
    )
    if mode == "Interactive" and needs_secret:
        stop_bridge()
        stop_pipeline()
        st.info("Enter an API key in the sidebar to use Interactive mode.")
        st.stop()

    session_id = current_session_id()
    if not slots.acquire(session_id):
        st.warning("Another visitor is using the GPU right now. Try again in a moment.")
        st.button("Retry")
        st.stop()

    try:
        pipeline = get_pipeline()
    except Exception as exc:
        st.error(f"Failed to initialize ARTalk avatar pipeline: {exc}")
        # Initialization spans several optional model paths whose imports and
        # checkpoints fail in ways the message alone does not locate.
        st.exception(exc)
        st.stop()
    slots.attach(session_id, "pipeline", pipeline)

    freeze_loaded_objects()

    # PersonaPlex streams continuously, including its own silence, so the pump
    # would feed the pipeline a second time and fight the model's audio.
    if mode == "Interactive" and backend == BACKEND_PERSONAPLEX:
        stop_silence_pump()
        silence_pump = None
    else:
        silence_pump = get_silence_pump(pipeline)
    slots.attach(session_id, "pump", silence_pump)
    pipeline.set_output_underrun_policy(
        st.session_state.underrun_policy,
        rebuffer_seconds=st.session_state.rebuffer_seconds,
        max_added_latency_seconds=st.session_state.max_added_latency,
    )
    event_log = get_event_log()
    slots.attach(session_id, "watcher", get_event_watcher(pipeline, event_log))

    def get_bridge() -> OpenAIRealtimeBridge | OpenAILiveBridge | PersonaPlexBridge:
        config = (
            backend,
            api_key,
            realtime_model,
            realtime_voice,
            realtime_instructions,
            realtime_ws_base_url,
            live_model,
            live_voice,
            live_instructions,
            live_delegation_model,
            personaplex_url,
            personaplex_voice,
            personaplex_text_prompt,
            id(pipeline),
        )
        bridge = st.session_state.get(BRIDGE_KEY)
        if bridge is not None and st.session_state.get(BRIDGE_CONFIG_KEY) != config:
            stop_bridge()
            bridge = None
        if bridge is None:
            if backend == BACKEND_PERSONAPLEX:
                bridge = PersonaPlexBridge(
                    url=personaplex_url,
                    pipeline=pipeline,
                    on_audio_output=None,
                    text_prompt=personaplex_text_prompt,
                    voice_prompt=personaplex_voice,
                )
            elif backend == BACKEND_OPENAI_LIVE:
                bridge = OpenAILiveBridge(
                    api_key=api_key,
                    pipeline=pipeline,
                    on_audio_output=silence_pump.mark_input if silence_pump else None,
                    model=live_model,
                    voice=live_voice,
                    instructions=live_instructions,
                    delegation_model=live_delegation_model,
                    event_log=event_log,
                )
            else:
                bridge = OpenAIRealtimeBridge(
                    api_key=api_key,
                    pipeline=pipeline,
                    on_audio_output=silence_pump.mark_input if silence_pump else None,
                    model=realtime_model,
                    voice=realtime_voice,
                    instructions=realtime_instructions,
                    websocket_base_url=realtime_ws_base_url,
                    event_log=event_log,
                )
            st.session_state[BRIDGE_KEY] = bridge
            st.session_state[BRIDGE_CONFIG_KEY] = config
        return bridge

    bridge = get_bridge() if mode == "Interactive" else None
    if bridge is not None and not bridge.is_running:
        # PersonaPlex withholds its handshake until the system prompts have
        # been stepped through the model, which is far slower than a hosted API.
        connect_timeout = 60.0 if backend == BACKEND_PERSONAPLEX else 8.0
        with st.spinner(f"Connecting to {backend}..."):
            bridge.start()
            bridge.wait_until_connected(timeout=connect_timeout)
        snap = bridge.snapshot()
        if snap["error"]:
            st.error(f"{backend} error: {snap['error']}")
        elif not snap["connected"]:
            st.warning(f"{backend} is still connecting. Wait a moment before START.")

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
        fps=ARTALK_FPS,
    )
    audio_source_track = create_audio_source_track(
        pipeline.audio_source_callback,
        key=f"artalk_audio_source_{mode.lower()}",
        sample_rate=ARTALK_SAMPLE_RATE,
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

    if bridge is not None:

        @st.fragment(run_every="500ms")
        def render_interactive_status() -> None:
            snap = bridge.snapshot()
            if snap["error"]:
                st.error(f"{backend} error: {snap['error']}")
            # Full duplex has no turn boundary to reconcile against, so how far
            # the listener trails the model is the only handle on barge-in.
            # It tracks the output prebuffer, which is the knob to tune.
            if "trailing_s" in snap:
                lag_col, spoken_col = st.columns(2)
                lag_col.metric("Listener trails model", f"{snap['trailing_s']:.2f} s")
                spoken_col.metric("Model audio", f"{snap['model_audio_s']:.0f} s")
            with st.container(height=260, border=True):
                if snap["assistant"]:
                    st.markdown(snap["assistant"])
                else:
                    st.caption("Waiting for response...")

    try:
        rtc_configuration = resolve_rtc_configuration(args.ice_provider)
    except Exception as exc:
        st.warning(
            f"ICE provider {args.ice_provider!r} failed ({exc}); using STUN "
            "only, which cannot connect through a proxy or a strict NAT."
        )
        rtc_configuration = STUN_ONLY

    def render_webrtc_component() -> None:
        webrtc_streamer(
            key=streamer_key,
            rtc_configuration=rtc_configuration,
            mode=WebRtcMode.SENDRECV,
            source_video_track=video_source_track,
            source_audio_track=audio_source_track,
            sink_audio_track=audio_sink_track,
            # Force both states explicitly so the toggle is a clean A/B —
            # browsers apply their own defaults for a bare {"audio": True}.
            media_stream_constraints={
                "audio": {
                    "echoCancellation": mic_processing,
                    "noiseSuppression": mic_processing,
                    "autoGainControl": mic_processing,
                },
                "video": False,
            },
            on_change=on_change,
        )

    avatar_col, diagnostics_col = st.columns([1, 1.35], gap="large")

    with avatar_col:
        st.subheader("Avatar")
        render_webrtc_component()
        if bridge is not None:
            render_interactive_status()

    @st.fragment(run_every="500ms")
    def render_diagnostics_fragment() -> None:
        render_pipeline_diagnostics(pipeline)

    @st.fragment(run_every="2s")
    def render_profiler_fragment() -> None:
        render_profiler_panel(pipeline, trace_root=args.profile_trace_dir)

    @st.fragment(run_every="1s")
    def render_gc_fragment() -> None:
        render_gc_panel(gc_pause_probe)

    @st.fragment(run_every="500ms")
    def render_event_log_fragment() -> None:
        render_event_log_panel(event_log)

    with diagnostics_col:
        # Deliberately outside the auto-rerunning fragments: a button inside
        # a run_every fragment races with its reruns and clicks get lost.
        if st.button(
            "Save diagnostics snapshot",
            key="save_diagnostics_snapshot",
            help=(
                "Click the moment you observe a jitter or lag; dumps all "
                "counters, spikes, GC state, and the session event timeline "
                "to a timestamped JSON file on the server, plus the "
                "last-served video frame as a PNG beside it."
            ),
        ):
            snapshot_path = save_diagnostics_snapshot(pipeline, event_log)
            st.caption(f"Saved `{snapshot_path}`")
        with st.expander("Session events", expanded=True):
            if st.button("Clear events", key="clear_event_log"):
                event_log.clear()
            render_event_log_fragment()
        if args.profile_trace_dir:
            with st.expander("Torch profiler", expanded=True):
                render_profiler_fragment()
        with st.expander("GC pauses", expanded=True):
            render_gc_fragment()
        with st.container(height=720, width="stretch", border=True, autoscroll=False):
            render_diagnostics_fragment()


script_hang_watchdog.start()
script_hang_watchdog.mark_run_start()
try:
    main()
finally:
    script_hang_watchdog.mark_run_end()
