#!/usr/bin/env python

"""Standalone Streamlit + streamlit-webrtc realtime demo for ARTalk.

This app is intentionally outside the ARTalk repository. It expects ARTalk and
GAGAvatar to be installed as Python packages and locates model assets through
environment variables.
"""

from __future__ import annotations

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
from artalk_streamlit_realtime.diagnostics import (
    render_gc_panel,
    render_pipeline_diagnostics,
    render_profiler_panel,
)
from artalk_streamlit_realtime.gc_probe import gc_pause_probe
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
    st.session_state.pop(BRIDGE_CONFIG_KEY, None)


def stop_pipeline() -> None:
    stop_silence_pump()
    pipeline = st.session_state.pop(PIPELINE_KEY, None)
    if pipeline is not None:
        pipeline.stop()
    st.session_state.pop(PIPELINE_CONFIG_KEY, None)


def main() -> None:
    gc_pause_probe.install()
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
            args.renderer_stage_sync,
            args.profile_trace_dir,
            args.profile_skip_chunks,
            args.profile_max_chunks,
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
                renderer_stage_sync=args.renderer_stage_sync,
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

    with diagnostics_col:
        if args.profile_trace_dir:
            with st.expander("Torch profiler", expanded=True):
                render_profiler_fragment()
        with st.expander("GC pauses", expanded=True):
            render_gc_fragment()
        with st.container(height=720, width="stretch", border=True, autoscroll=False):
            render_diagnostics_fragment()


main()
