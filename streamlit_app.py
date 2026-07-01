#!/usr/bin/env python

"""Standalone Streamlit + streamlit-webrtc realtime demo for ARTalk.

This app is intentionally outside the ARTalk repository. It expects ARTalk and
GAGAvatar to be installed as Python packages and locates model assets through
environment variables.
"""

from __future__ import annotations

from typing import Any

import av
import streamlit as st
from artalk.assets import ARTalkAssets
from artalk.realtime_pipeline import ARTalkPipeline
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit.runtime.scriptrunner import get_script_run_ctx
from streamlit_webrtc import (
    WebRtcMode,
    create_audio_sink_track,
    create_audio_source_track,
    create_video_source_track,
    webrtc_streamer,
)

from artalk_streamlit_realtime.assets import resolve_gagavatar_assets
from artalk_streamlit_realtime.config import (
    ARTALK_FPS,
    ARTALK_SAMPLE_RATE,
    DEFAULT_APPEARANCE,
    DEFAULT_REALTIME_INSTRUCTIONS,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_VOICE,
    DEFAULT_STYLE,
    REALTIME_VOICES,
    parse_args,
)
from artalk_streamlit_realtime.diagnostics import render_pipeline_diagnostics
from artalk_streamlit_realtime.openai_bridge import OpenAIRealtimeBridge
from artalk_streamlit_realtime.runtime import (
    list_gagavatar_ids,
    list_style_ids,
    load_artalk_runtime,
    load_gagavatar,
    load_style_motion,
)
from artalk_streamlit_realtime.silence import PipelineSilencePump

PIPELINE_KEY = "artalk_pipeline"
PIPELINE_CONFIG_KEY = "artalk_pipeline_config"
SILENCE_PUMP_KEY = "artalk_silence_pump"
SILENCE_PUMP_CONFIG_KEY = "artalk_silence_pump_config"
BRIDGE_KEY = "openai_realtime_bridge"
BRIDGE_CONFIG_KEY = "openai_realtime_bridge_config"
MEDIA_SESSION_TOKEN_KEY = "artalk_media_session_token"


def get_current_session_state() -> Any:
    """Capture the real session state for callbacks outside the script thread."""
    ctx = get_script_run_ctx()
    if ctx is not None:
        session_state = getattr(ctx, "session_state", None)
        if session_state is not None:
            return session_state
    return st.session_state


def session_state_pop(session_state: Any, key: str) -> Any:
    if key in session_state:
        value = session_state[key]
        del session_state[key]
        return value
    return None


def clear_stale_webrtc_answer(ctx: Any) -> None:
    """Clear idle SDP state left behind by streamlit-webrtc 0.74.0."""
    get_worker = getattr(ctx, "_get_worker", None)
    worker = get_worker() if callable(get_worker) else None
    if worker is not None or ctx.state.playing or ctx.state.signalling:
        return
    if not getattr(ctx, "_sdp_answer_json", None) and not getattr(
        ctx, "_is_sdp_answer_sent", False
    ):
        return
    # streamlit-webrtc resets this state when an idle context still has a
    # worker, but a stopped session can leave only stale SDP-answer fields.
    # Without a rerun, the next START may reuse that stale answer and stall.
    ctx._sdp_answer_json = None
    ctx._is_sdp_answer_sent = False
    ctx._component_value_snapshot = None
    st.rerun()


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default
    return str(value) if value is not None else default


def split_appearance(value: str) -> tuple[str, str | None]:
    if value == DEFAULT_APPEARANCE:
        return "mesh", None
    source, avatar_id = value.split(":", 1)
    if source != "gagavatar" or not avatar_id:
        raise ValueError(f"Unknown appearance: {value}")
    return source, avatar_id


def stop_silence_pump(session_state: Any | None = None) -> None:
    session_state = session_state if session_state is not None else st.session_state
    pump = session_state_pop(session_state, SILENCE_PUMP_KEY)
    if pump is not None:
        pump.stop()
    session_state_pop(session_state, SILENCE_PUMP_CONFIG_KEY)


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


def stop_bridge(session_state: Any | None = None) -> None:
    session_state = session_state if session_state is not None else st.session_state
    bridge = session_state_pop(session_state, BRIDGE_KEY)
    if bridge is not None:
        bridge.stop()
    session_state_pop(session_state, BRIDGE_CONFIG_KEY)


def stop_pipeline(session_state: Any | None = None) -> None:
    session_state = session_state if session_state is not None else st.session_state
    stop_silence_pump(session_state)
    pipeline = session_state_pop(session_state, PIPELINE_KEY)
    if pipeline is not None:
        pipeline.stop()
    session_state_pop(session_state, PIPELINE_CONFIG_KEY)


def main() -> None:
    args = parse_args()
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

        appearance_options = [
            DEFAULT_APPEARANCE,
            *[f"gagavatar:{avatar_id}" for avatar_id in gagavatar_ids],
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
        mode = st.radio("Mode", ["Loopback", "Interactive"], index=1, horizontal=True)

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
            )
        config = (
            args.device,
            mode,
            render_res,
            args.render_batch_size,
            args.output_prebuffer_seconds,
            args.output_segment_seconds,
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
                output_audio_prebuffer_seconds=args.output_prebuffer_seconds,
                output_segment_seconds=args.output_segment_seconds,
                renderer_mode=renderer_mode,
                gagavatar=gagavatar,
                gagavatar_flame=gagavatar_flame,
                shape_id=avatar_id,
            )
            st.session_state[PIPELINE_KEY] = pipeline
            st.session_state[PIPELINE_CONFIG_KEY] = config
        return pipeline

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

    session_state = get_current_session_state()
    media_session_token = object()
    session_state[MEDIA_SESSION_TOKEN_KEY] = media_session_token

    def on_media_ended() -> None:
        current_token = (
            session_state[MEDIA_SESSION_TOKEN_KEY]
            if MEDIA_SESSION_TOKEN_KEY in session_state
            else None
        )
        if current_token is not media_session_token:
            return
        stop_bridge(session_state)
        stop_pipeline(session_state)

    video_source_track = create_video_source_track(
        pipeline.video_source_callback,
        key=f"artalk_video_source_{mode.lower()}",
        fps=ARTALK_FPS,
        on_ended=on_media_ended,
    )
    audio_source_track = create_audio_source_track(
        pipeline.audio_source_callback,
        key=f"artalk_audio_source_{mode.lower()}",
        sample_rate=ARTALK_SAMPLE_RATE,
        ptime=0.020,
        on_ended=on_media_ended,
    )
    audio_sink_track = create_audio_sink_track(
        on_loopback_audio_frame if mode == "Loopback" else on_interactive_audio_frame,
        key=f"artalk_audio_sink_{mode.lower()}",
        on_ended=on_media_ended,
    )

    streamer_key = f"artalk_{mode.lower()}"

    def on_change() -> None:
        ctx = st.session_state.get(streamer_key)
        if ctx is None:
            return
        if mode == "Interactive" and ctx.state.playing and bridge is not None:
            bridge.start()
        if not ctx.state.playing and not ctx.state.signalling:
            stop_bridge()
            stop_pipeline()

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
        ctx = webrtc_streamer(
            key=streamer_key,
            mode=WebRtcMode.SENDRECV,
            source_video_track=video_source_track,
            source_audio_track=audio_source_track,
            sink_audio_track=audio_sink_track,
            media_stream_constraints={"audio": True, "video": False},
            on_change=on_change,
        )
        clear_stale_webrtc_answer(ctx)

    avatar_col, diagnostics_col = st.columns([1, 1.35], gap="large")

    with avatar_col:
        st.subheader("Avatar")
        render_webrtc_component()
        if bridge is not None:
            render_interactive_status()

    @st.fragment(run_every="500ms")
    def render_diagnostics_fragment() -> None:
        render_pipeline_diagnostics(pipeline)

    with diagnostics_col:
        with st.container(height=720, width="stretch", border=True, autoscroll=False):
            render_diagnostics_fragment()


main()
