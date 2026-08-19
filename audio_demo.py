#!/usr/bin/env python

"""Audio-only conversation demo: talk to a backend with no avatar in the way.

The avatar contributes a full second of output prebuffer, which dominates
perceived latency and makes backends hard to compare by ear. This app runs the
same bridges against a passthrough audio pipeline, so what you hear is the
model plus the network and nothing else. It needs no GPU and no ARTalk assets.

  streamlit run audio_demo.py
"""

from __future__ import annotations

import av
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from streamlit_webrtc import WebRtcMode, create_audio_sink_track, create_audio_source_track, webrtc_streamer

from artalk_streamlit_realtime.audio_only_pipeline import (
    DEFAULT_AUDIO_ONLY_PREBUFFER_SECONDS,
    AudioOnlyPipeline,
)
from artalk_streamlit_realtime.config import (
    CUSTOM_ENDPOINT_LABEL,
    DEFAULT_PERSONAPLEX_TEXT_PROMPT,
    DEFAULT_PERSONAPLEX_URL,
    DEFAULT_PERSONAPLEX_VOICE,
    DEFAULT_REALTIME_INSTRUCTIONS,
    DEFAULT_REALTIME_MODEL,
    DEFAULT_REALTIME_VOICE,
    DEFAULT_REALTIME_WEBSOCKET_BASE_URL,
    OPENAI_REALTIME_SAMPLE_RATE,
    PERSONAPLEX_SAMPLE_RATE,
    PERSONAPLEX_VOICES,
    REALTIME_ENDPOINT_PRESETS,
    REALTIME_VOICES,
)
from artalk_streamlit_realtime.openai_bridge import OpenAIRealtimeBridge
from artalk_streamlit_realtime.personaplex_bridge import PersonaPlexBridge

BACKEND_OPENAI = "OpenAI Realtime"
BACKEND_PERSONAPLEX = "PersonaPlex"
PIPELINE_KEY = "audio_demo_pipeline"
BRIDGE_KEY = "audio_demo_bridge"
CONFIG_KEY = "audio_demo_config"


def get_secret(name: str, default: str = "") -> str:
    try:
        value = st.secrets.get(name, default)
    except StreamlitSecretNotFoundError:
        return default
    return str(value) if value is not None else default


def stop_bridge() -> None:
    bridge = st.session_state.pop(BRIDGE_KEY, None)
    if bridge is not None:
        bridge.stop()
    st.session_state.pop(CONFIG_KEY, None)


def main() -> None:
    st.set_page_config(page_title="Voice backend demo", page_icon=":studio_microphone:")
    st.title("Voice backend demo")
    st.caption("Audio only, so latency is the model and the network, not the avatar.")

    with st.sidebar:
        backend = st.radio("Backend", [BACKEND_OPENAI, BACKEND_PERSONAPLEX])
        prebuffer = st.slider(
            "Output prebuffer (s)",
            0.0, 1.0, DEFAULT_AUDIO_ONLY_PREBUFFER_SECONDS, 0.05,
            help="Jitter margin. Raise it if audio breaks up, lower it for less lag.",
        )

        api_key = ""
        model = DEFAULT_REALTIME_MODEL
        voice = DEFAULT_REALTIME_VOICE
        instructions = DEFAULT_REALTIME_INSTRUCTIONS
        ws_base_url = DEFAULT_REALTIME_WEBSOCKET_BASE_URL
        pp_url = DEFAULT_PERSONAPLEX_URL
        pp_voice = DEFAULT_PERSONAPLEX_VOICE
        pp_prompt = DEFAULT_PERSONAPLEX_TEXT_PROMPT

        if backend == BACKEND_OPENAI:
            names = list(REALTIME_ENDPOINT_PRESETS) + [CUSTOM_ENDPOINT_LABEL]
            chosen = st.selectbox(
                "Endpoint",
                names,
                index=names.index(CUSTOM_ENDPOINT_LABEL)
                if DEFAULT_REALTIME_WEBSOCKET_BASE_URL
                else 0,
            )
            if chosen == CUSTOM_ENDPOINT_LABEL:
                ws_base_url = st.text_input(
                    "WebSocket base URL",
                    value=DEFAULT_REALTIME_WEBSOCKET_BASE_URL,
                    help="The SDK appends `/realtime`.",
                ).strip()
                secret_name = "OPENAI_API_KEY"
            else:
                endpoint = REALTIME_ENDPOINT_PRESETS[chosen]
                ws_base_url = endpoint.base_url
                secret_name = endpoint.secret
                if ws_base_url:
                    st.caption(f"`{ws_base_url}`")
            api_key = get_secret(secret_name) if secret_name else ""
            if api_key:
                st.success(f"`{secret_name}` loaded.")
            elif secret_name:
                st.warning(f"`{secret_name}` is not configured.")
            model = st.text_input("Model", value=DEFAULT_REALTIME_MODEL)
            voice = st.selectbox(
                "Voice", REALTIME_VOICES, index=REALTIME_VOICES.index(DEFAULT_REALTIME_VOICE)
            )
            instructions = st.text_area(
                "Instructions", value=DEFAULT_REALTIME_INSTRUCTIONS, height=120
            )
        else:
            pp_url = st.text_input("moshi server URL", value=DEFAULT_PERSONAPLEX_URL).strip()
            pp_voice = st.selectbox(
                "Voice prompt",
                PERSONAPLEX_VOICES,
                index=PERSONAPLEX_VOICES.index(DEFAULT_PERSONAPLEX_VOICE),
            )
            pp_prompt = st.text_area(
                "Persona", value=DEFAULT_PERSONAPLEX_TEXT_PROMPT, height=120
            )

    # Each backend speaks at its own rate; keeping it avoids a resample.
    sample_rate = (
        PERSONAPLEX_SAMPLE_RATE if backend == BACKEND_PERSONAPLEX
        else OPENAI_REALTIME_SAMPLE_RATE
    )
    config = (
        backend, prebuffer, sample_rate, api_key, model, voice, instructions,
        ws_base_url, pp_url, pp_voice, pp_prompt,
    )
    if st.session_state.get(CONFIG_KEY) != config:
        stop_bridge()
        st.session_state[PIPELINE_KEY] = AudioOnlyPipeline(
            sample_rate=sample_rate, prebuffer_seconds=prebuffer
        )
    pipeline = st.session_state[PIPELINE_KEY]

    bridge = st.session_state.get(BRIDGE_KEY)
    if bridge is None:
        if backend == BACKEND_PERSONAPLEX:
            bridge = PersonaPlexBridge(
                url=pp_url,
                pipeline=pipeline,
                on_audio_output=None,
                text_prompt=pp_prompt,
                voice_prompt=pp_voice,
            )
        else:
            bridge = OpenAIRealtimeBridge(
                api_key=api_key,
                pipeline=pipeline,
                on_audio_output=None,
                model=model,
                voice=voice,
                instructions=instructions,
                websocket_base_url=ws_base_url,
            )
        st.session_state[BRIDGE_KEY] = bridge
        st.session_state[CONFIG_KEY] = config

    if not bridge.is_running:
        # PersonaPlex withholds its handshake until the system prompts have
        # been stepped through the model, which a hosted API does not do.
        timeout = 60.0 if backend == BACKEND_PERSONAPLEX else 8.0
        with st.spinner(f"Connecting to {backend}..."):
            bridge.start()
            bridge.wait_until_connected(timeout=timeout)

    snap = bridge.snapshot()
    if snap["error"]:
        st.error(f"{backend}: {snap['error']}")
    elif not snap["connected"]:
        st.warning(f"{backend} is still connecting. Wait before pressing START.")

    def on_mic_frame(frame: av.AudioFrame) -> None:
        bridge.push_input(frame)

    audio_source_track = create_audio_source_track(
        pipeline.audio_source_callback,
        key=f"audio_demo_source_{backend}_{sample_rate}",
        sample_rate=sample_rate,
        ptime=0.020,
    )
    audio_sink_track = create_audio_sink_track(
        on_mic_frame,
        key=f"audio_demo_sink_{backend}",
        on_ended=stop_bridge,
    )
    webrtc_streamer(
        key=f"audio_demo_{backend}",
        mode=WebRtcMode.SENDRECV,
        source_audio_track=audio_source_track,
        sink_audio_track=audio_sink_track,
        media_stream_constraints={"audio": True, "video": False},
    )

    @st.fragment(run_every="500ms")
    def render_status() -> None:
        snap = bridge.snapshot()
        counters = pipeline.metrics_snapshot()["counters"]
        cols = st.columns(3)
        cols[0].metric("Buffered", f"{counters['audio_out_buffer_seconds']:.2f} s")
        cols[1].metric("Underruns", counters["audio_playback_underrun_frames"])
        if "trailing_s" in snap:
            cols[2].metric("Trails model", f"{snap['trailing_s']:.2f} s")
        if snap.get("user"):
            st.caption(f"You: {snap['user']}")
        with st.container(height=240, border=True):
            st.markdown(snap["assistant"] or "_Waiting for a response..._")

    render_status()


if __name__ == "__main__":
    main()
