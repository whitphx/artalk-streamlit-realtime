#!/usr/bin/env python

"""Standalone Streamlit + streamlit-webrtc realtime demo for ARTalk.

This app is intentionally outside the ARTalk repository. It expects ARTalk and
GAGAvatar to be installed as Python packages and locates model assets through
environment variables.
"""

from __future__ import annotations

import faulthandler
import inspect
import logging
import os
import re
import signal
import threading
import time
from pathlib import Path

import av
import streamlit as st
import torch
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

from artalk_streamlit_realtime.ai_badge import BadgedARTalkPipeline
from artalk_streamlit_realtime.assets import resolve_gagavatar_assets
from artalk_streamlit_realtime.avatar_gallery import (
    avatar_gallery,
    builtin_thumbnails,
    user_thumbnail,
)
from artalk_streamlit_realtime.avatar_registry import (
    AvatarRegistrationError,
    UserAvatarRegistry,
    owner_dir,
    prune_session_owners,
    session_owner_dir,
    slugify_avatar_id,
)
from artalk_streamlit_realtime.client_render import (
    CHANNEL_LABEL,
    CHANNEL_OPTIONS,
    FRONTEND_SCRIPT,
    NETWORKS,
    ORT_CDN_URL,
    QUANT_STEP,
    ClientRenderAdapter,
    ClientRenderAssets,
    ClientRenderPipeline,
    serve_dirs,
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
    VOICE_DESCRIPTIONS,
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
from artalk_streamlit_realtime.sessions import (
    ENDED_NOT_STARTED,
    ENDED_TIME_LIMIT,
    current_session_id,
    live_session_ids,
    slots,
)
from artalk_streamlit_realtime.artalk1s import ARTalk1sStreamer
from artalk_streamlit_realtime.fallingwater import FallingwaterStreamer
from artalk_streamlit_realtime.framemodel import FrameModelStreamer
from artalk_streamlit_realtime.gc_probe import freeze_loaded_objects, gc_pause_probe
from artalk_streamlit_realtime.hang_watchdog import script_hang_watchdog
from artalk_streamlit_realtime.login import parse_allow_list, require_login
from artalk_streamlit_realtime.loop_watchdog import loop_stall_watchdog
from artalk_streamlit_realtime.openai_bridge import OpenAIRealtimeBridge
from artalk_streamlit_realtime.openai_live_bridge import OpenAILiveBridge
from artalk_streamlit_realtime.personaplex_bridge import PersonaPlexBridge
from artalk_streamlit_realtime.runtime import (
    StreamingGAGAvatarAdapter,
    list_gagavatar_ids,
    list_style_ids,
    load_artalk1s_streamer_model,
    load_artalk_runtime,
    load_fallingwater_streamer_model,
    load_frame_streamer_model,
    load_gagavatar,
    load_style_motion,
    preimport_path_rebinders,
)
from artalk_streamlit_realtime.session_records import (
    MediaSettings,
    SessionRecorder,
    appearance_kind,
    load_consent_notice,
    load_voice_consent_notice,
    open_record_store,
    render_feedback_form,
    render_feedback_slot,
)
from artalk_streamlit_realtime.silence import PipelineSilencePump
from artalk_streamlit_realtime.streamlit_patches import (
    disable_streamlit_source_watcher,
)

PIPELINE_KEY = "artalk_pipeline"
PIPELINE_CONFIG_KEY = "artalk_pipeline_config"
# A rerun can start while the previous script run is still loading models
# (nothing in there is an st.* call the runner could stop at), and two runs
# doing that at once race in process-global state: CUDA graph captures
# against renders, torch's non-thread-safe lazy linalg init, and `sys.path`
# between torch.hub and the packages that rebind it (see
# runtime.preimport_path_rebinders). One lock covers the loads and the
# construction they feed.
_MODEL_BUILD_LOCK = threading.Lock()
SILENCE_PUMP_KEY = "artalk_silence_pump"
SILENCE_PUMP_CONFIG_KEY = "artalk_silence_pump_config"
BACKEND_OPENAI = "OpenAI Realtime"
BACKEND_OPENAI_LIVE = "OpenAI Live"
BACKEND_PERSONAPLEX = "PersonaPlex"
BACKEND_LABELS = {
    "openai-realtime": BACKEND_OPENAI,
    "openai-live": BACKEND_OPENAI_LIVE,
    "personaplex": BACKEND_PERSONAPLEX,
}
BRIDGE_KEY = "openai_realtime_bridge"
BRIDGE_CONFIG_KEY = "openai_realtime_bridge_config"
EVENT_LOG_KEY = "artalk_event_log"
EVENT_WATCHER_KEY = "artalk_event_watcher"
EVENT_WATCHER_CONFIG_KEY = "artalk_event_watcher_config"
SESSION_ENDED_KEY = "artalk_session_ended"
QUEUED_KEY = "artalk_queued"
SESSION_RECORDER_KEY = "artalk_session_recorder"
APPEARANCE_KEY = "artalk_appearance"
AVATAR_GALLERY_KEY = "avatar_gallery"
VOICE_HELP = (
    "Descriptions are the provider's own. OpenAI publishes none for its "
    "original voices, so those show only their names."
)
SESSION_END_WARNING_SECONDS = 10
PAGE_TITLE = "ARTalk Realtime"

# Browsers start an AudioContext suspended and show the notification prompt
# only in response to a click, so the visitor opts in here while waiting; the
# admission alert then reuses the context the click unlocked.
TURN_ALERT_OPT_IN = """
<div>
  <button id="artalk-turn-alert" type="button"
    style="padding: 0.4rem 0.8rem; border-radius: 0.5rem; border: 1px solid #ccc;
           background: transparent; color: inherit; cursor: pointer; font: inherit;">
    Alert me with a sound when it is my turn
  </button>
  <span id="artalk-turn-alert-status" aria-live="polite" style="margin-left: 0.5rem;"></span>
</div>
<script>
(() => {
  const button = document.getElementById("artalk-turn-alert");
  const status = document.getElementById("artalk-turn-alert-status");
  if (!button || button.dataset.bound) return;
  button.dataset.bound = "1";
  button.addEventListener("click", async () => {
    window.artalkTurnAudio = window.artalkTurnAudio || new AudioContext();
    await window.artalkTurnAudio.resume();
    if ("Notification" in window && Notification.permission === "default") {
      await Notification.requestPermission();
    }
    const notify = "Notification" in window && Notification.permission === "granted";
    status.textContent = notify
      ? "You will hear a sound and get a notification."
      : "You will hear a sound.";
  });
})();
</script>
"""

TURN_ALERT_SCRIPT = f"""
<script>
(() => {{
  document.title = "Your turn: press START · {PAGE_TITLE}";
  if (window.artalkTurnAlerted) return;
  window.artalkTurnAlerted = true;
  // Without the opt-in click, a context still starts if the visitor has
  // clicked anywhere on the page before; otherwise it stays suspended and
  // the alert is silent.
  const audio = window.artalkTurnAudio || (window.artalkTurnAudio = new AudioContext());
  audio.resume().then(() => {{
    for (const [offset, freq] of [[0, 660], [0.18, 880]]) {{
      const osc = audio.createOscillator();
      const gain = audio.createGain();
      const t = audio.currentTime + offset;
      osc.frequency.value = freq;
      gain.gain.setValueAtTime(0.0001, t);
      gain.gain.exponentialRampToValueAtTime(0.2, t + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0001, t + 0.15);
      osc.connect(gain).connect(audio.destination);
      osc.start(t);
      osc.stop(t + 0.16);
    }}
  }});
  if ("Notification" in window && Notification.permission === "granted") {{
    new Notification("{PAGE_TITLE}", {{ body: "It is your turn. Press START to begin." }});
  }}
}})();
</script>
"""


def queue_title_script(position: int) -> str:
    # The tick changes the HTML every second: Streamlit reruns a script only
    # when its element changes, and a full rerun resets the title to the
    # page config's.
    return (
        f"<script>window.artalkQueueTick = {time.time():.0f};"
        "window.artalkTurnAlerted = false;"
        f'document.title = "#{position} in queue · {PAGE_TITLE}";</script>'
    )


def format_wait(seconds: float) -> str:
    minutes, seconds = divmod(round(seconds), 60)
    return f"{minutes} min {seconds} s" if minutes else f"{seconds} s"


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


def voice_label(voice: str) -> str:
    description = VOICE_DESCRIPTIONS.get(voice)
    return f"{voice} ({description})" if description else voice


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


def get_silence_pump(
    pipeline: ARTalkPipeline, idle_motion: bool, continuous_input: bool
) -> PipelineSilencePump:
    config = (id(pipeline), idle_motion, continuous_input)
    pump = st.session_state.get(SILENCE_PUMP_KEY)
    if pump is not None and st.session_state.get(SILENCE_PUMP_CONFIG_KEY) != config:
        pump.stop()
        pump = None
    if pump is None:
        pump = PipelineSilencePump(
            pipeline, idle_motion=idle_motion, continuous_input=continuous_input
        )
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


def get_session_recorder(args, login_id: str | None) -> SessionRecorder | None:
    recorder = st.session_state.get(SESSION_RECORDER_KEY)
    if recorder is None:
        notice = load_consent_notice(args.consent_notice, args.consent_notice_text)
        if notice is None:
            return None
        recorder = SessionRecorder(
            open_record_store(args.session_data_dir, args.session_data_repo),
            notice,
            login_id=login_id,
            log_sessions=args.session_logging,
            feedback=args.feedback,
            voice_notice=load_voice_consent_notice(
                args.voice_consent_notice, args.voice_consent_notice_text
            ),
        )
        st.session_state[SESSION_RECORDER_KEY] = recorder
    return recorder


def stop_pipeline() -> None:
    recorder = st.session_state.get(SESSION_RECORDER_KEY)
    if recorder is not None:
        recorder.end_media("stopped")
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
    preimport_path_rebinders()
    try:
        from streamlit_webrtc.eventloop import get_global_event_loop

        loop_stall_watchdog.install(get_global_event_loop())
    except Exception:
        logging.getLogger(__name__).warning(
            "[loop-watchdog] not armed", exc_info=True
        )
    args = parse_args()
    if args.torch_threads:
        torch.set_num_threads(args.torch_threads)
    slots.limit = args.max_sessions
    slots.start_grace = args.start_grace
    artalk_assets = ARTalkAssets.resolve(root=args.asset_dir)
    gagavatar_assets = resolve_gagavatar_assets(args, artalk_assets)
    asset_dir = artalk_assets.root
    tracked_path = gagavatar_assets.tracked_path
    model_path = gagavatar_assets.model_path
    flame_model_path = gagavatar_assets.flame_model_path

    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=":speech_balloon:",
        layout="wide",
    )
    title_col, view_col = st.columns([3, 1], vertical_alignment="bottom")
    title_col.title(PAGE_TITLE)
    pro = view_col.toggle(
        "Professional mode",
        key="pro",
        # In the URL, so a reload or a bookmark keeps the view.
        bind="query-params",
        help="Every setting, plus the pipeline diagnostics, in an Advanced tab.",
    )
    login_id = (
        require_login(
            parse_allow_list(args.login_email_domains),
            parse_allow_list(args.login_orgs),
            parse_allow_list(args.login_users),
        )
        if args.login
        else None
    )
    recorder = None
    if (
        args.feedback
        or args.session_logging
        or args.voice_consent_notice
        or args.voice_consent_notice_text
    ):
        recorder = get_session_recorder(args, login_id)
    chunk_caption = st.empty()

    client_assets = None
    if args.client_render:
        if not args.client_render_dir or model_path is None:
            st.error(
                "--client-render needs --client-render-dir and the GAGAvatar "
                "assets (its vertices come from the GAGAvatar runtime)."
            )
            st.stop()
        # Probed by parameter rather than by version: the data-channel build
        # is an unreleased checkout, so no version number marks it.
        if "data_channels" not in inspect.signature(webrtc_streamer).parameters:
            st.error(
                "--client-render needs a streamlit-webrtc build with data "
                "channels, and the installed one has none. See the README's "
                "section on rendering in the browser."
            )
            st.stop()
        client_assets = ClientRenderAssets(args.client_render_dir)
        client_dirs = serve_dirs(client_assets)

    try:
        with _MODEL_BUILD_LOCK:
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

    # The preload and the pipeline build must pass identical arguments, or
    # the preload misses the resource cache and the build loads again.
    def load_photoreal_renderer():
        return load_gagavatar(
            args.device,
            str(model_path),
            str(tracked_path) if tracked_path else None,
            str(flame_model_path) if flame_model_path else None,
            autocast_dtype="float16" if args.renderer_fp16 else None,
            compile_mode="cuda-graph" if args.renderer_compile else None,
        )

    if args.preload_gagavatar and model_path is not None:
        try:
            with _MODEL_BUILD_LOCK:
                load_photoreal_renderer()
        except Exception as exc:
            # Keep mesh mode usable; only a photoreal pick should fail, as it
            # does without the preload.
            st.warning(f"Failed to preload the photoreal renderer: {exc}")

    avatar_root = args.user_avatar_dir
    if args.isolate_user_avatars:
        live = live_session_ids()
        if live is not None:
            prune_session_owners(live)
        if login_id is not None:
            avatar_root = owner_dir(args.user_avatar_dir, "user", login_id)
        else:
            avatar_root = session_owner_dir(args.user_avatar_dir, current_session_id())
    user_registry = UserAvatarRegistry(
        avatar_root,
        track_python=args.gagavatar_track_python,
        track_dir=args.gagavatar_track_dir,
        track_device=args.gagavatar_track_device,
    )

    gagavatar_ids = list_gagavatar_ids(str(tracked_path) if tracked_path else None)
    user_avatar_ids = user_registry.list_ids()
    style_ids = list_style_ids(str(asset_dir))

    avatar_col, panel_col = st.columns([1, 1.35] if pro else [1, 1], gap="large")
    if pro:
        picker_box, advanced_box = panel_col.tabs(["Avatar and voice", "Advanced"])
    else:
        picker_box, advanced_box = panel_col.container(), None
    with avatar_col:
        st.subheader("Avatar")
        # Filled once the mode is known, which the panel beside decides.
        ai_caption = st.empty()
        # Every notice goes inside this container: an element that appears or
        # vanishes ahead of the WebRTC component between runs makes the
        # browser remount it, which drops the media connection.
        status_box = st.container()

    with picker_box:
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

        offered = {
            avatar_id.strip() for avatar_id in args.avatars.split(",") if avatar_id.strip()
        }
        builtin_ids = sorted(
            (
                avatar_id
                for avatar_id in gagavatar_ids
                if not offered or avatar_id in offered
            ),
            # Numbered ids in numeric order: 2.jpg before 10.jpg.
            key=lambda name: [
                (0, int(part)) if part.isdigit() else (1, part)
                for part in re.split(r"(\d+)", name)
            ],
        )
        if client_assets is not None:
            # The browser can only draw avatars it has a bundle for.
            bundled = set(client_assets.avatar_ids())
            builtin_ids = [avatar_id for avatar_id in builtin_ids if avatar_id in bundled]
            if not builtin_ids:
                st.error(
                    f"No avatar in {client_assets.root / 'avatars'} matches a "
                    "GAGAvatar avatar offered here."
                )
                st.stop()
        builtin_previews = builtin_thumbnails(str(tracked_path) if tracked_path else None)

        def user_preview(avatar_id: str) -> str:
            path = user_registry.root / avatar_id / "tracked.pt"
            return user_thumbnail(str(path), path.stat().st_mtime)

        # Built-in ids can be file names that name the person pictured, so
        # visitors see a number instead. Visitors named their own uploads.
        gallery_items = [
            *(
                [{"value": DEFAULT_APPEARANCE, "label": "Mesh", "src": None, "alt": ""}]
                if client_assets is None
                else []
            ),
            *[
                {
                    "value": f"gagavatar:{avatar_id}",
                    "label": f"Avatar {number}",
                    "src": builtin_previews.get(avatar_id),
                    "alt": f"Avatar {number} preview",
                }
                for number, avatar_id in enumerate(builtin_ids, start=1)
            ],
            *(
                [
                    {
                        "value": f"gagavatar:{avatar_id}",
                        "label": avatar_id,
                        "src": user_preview(avatar_id),
                        "alt": f"{avatar_id} preview",
                    }
                    for avatar_id in user_avatar_ids
                ]
                if client_assets is None
                else []
            ),
        ]
        appearance_options = [item["value"] for item in gallery_items]
        # Also catches an avatar deleted since the last run.
        if st.session_state.get(APPEARANCE_KEY) not in appearance_options:
            st.session_state[APPEARANCE_KEY] = (
                args.default_appearance
                if args.default_appearance in appearance_options
                else appearance_options[0]
            )
        appearance = st.session_state[APPEARANCE_KEY]

        def on_avatar_pick() -> None:
            st.session_state[APPEARANCE_KEY] = st.session_state[AVATAR_GALLERY_KEY].picked

        avatar_gallery(
            "Avatar",
            gallery_items,
            appearance,
            key=AVATAR_GALLERY_KEY,
            on_pick=on_avatar_pick,
        )
        modes = ["Loopback", "Interactive"]
        mode = st.radio(
            "Mode",
            modes,
            index=modes.index(args.default_mode.capitalize()),
            horizontal=True,
            help=(
                "Loopback moves the avatar to your own voice and needs no API "
                "key. Interactive lets you talk with an AI through the avatar."
            ),
        )
    ai_caption.caption(
        "The avatar's face, voice and words are generated by AI."
        if mode == "Interactive"
        else "The avatar's face is generated by AI; the voice is your own."
    )

    # Easy mode leaves these widgets out, so each value is read back from
    # Session State with the widget's own default. `persist_state` keeps
    # what was set in professional mode while easy mode hides it, so
    # switching views does not change the session or rebuild the pipeline.
    motion_labels = {"artalk": "ARTalk (4 s chunks)", "artalk1s": "ARTalk 1 s"}
    can_switch_motion = bool(
        args.motion_model in motion_labels
        and args.artalk1s_train_code_dir
        and args.artalk1s_checkpoint
    )
    style_options = [DEFAULT_STYLE, *style_ids]
    default_style = "natural_0" if "natural_0" in style_ids else DEFAULT_STYLE
    default_rebuffer_seconds = float(
        args.output_rebuffer_seconds
        if args.output_rebuffer_seconds is not None
        else args.output_prebuffer_seconds
    )
    if advanced_box is not None:
        with advanced_box:
            st.subheader("Avatar motion")
            if can_switch_motion:
                st.radio(
                    "Motion model",
                    list(motion_labels),
                    index=list(motion_labels).index(args.motion_model),
                    format_func=motion_labels.__getitem__,
                    horizontal=True,
                    key="motion_model",
                    persist_state="page",
                    help=(
                        "The 1 s model starts speaking about three seconds sooner. "
                        "Switching rebuilds the pipeline."
                    ),
                )
    motion_model = (
        st.session_state.get("motion_model", args.motion_model)
        if can_switch_motion
        else args.motion_model
    )
    if advanced_box is not None:
        with advanced_box:
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
            if client_assets is not None:
                st.selectbox(
                    "Browser renderer",
                    list(NETWORKS),
                    format_func=NETWORKS.__getitem__,
                    key="client_network",
                    persist_state="page",
                    help="Switches without restarting the session.",
                )
            st.selectbox(
                "Style",
                style_options,
                index=style_options.index(default_style),
                key="style",
                persist_state="page",
            )
            st.toggle(
                "Mic echo cancellation & noise suppression",
                value=True,
                key="mic_processing",
                persist_state="page",
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
                    persist_state="page",
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
                    value=default_rebuffer_seconds,
                    step=0.25,
                    key="rebuffer_seconds",
                    persist_state="page",
                )
                st.slider(
                    "Max added latency (s)",
                    0.0,
                    10.0,
                    value=float(args.max_added_latency_seconds),
                    step=0.5,
                    key="max_added_latency",
                    persist_state="page",
                )
    style_id = st.session_state.get("style", default_style)
    client_network = st.session_state.get("client_network", next(iter(NETWORKS)))
    mic_processing = st.session_state.get("mic_processing", True)
    underrun_policy = st.session_state.get("underrun_policy", args.output_underrun_policy)
    rebuffer_seconds = st.session_state.get("rebuffer_seconds", default_rebuffer_seconds)
    max_added_latency = st.session_state.get(
        "max_added_latency", float(args.max_added_latency_seconds)
    )

    api_key = ""
    realtime_model = DEFAULT_REALTIME_MODEL
    realtime_voice = DEFAULT_REALTIME_VOICE
    realtime_instructions = DEFAULT_REALTIME_INSTRUCTIONS
    realtime_ws_base_url = DEFAULT_REALTIME_WEBSOCKET_BASE_URL
    realtime_declare_audio_format = False
    live_model = DEFAULT_LIVE_MODEL
    live_voice = DEFAULT_LIVE_VOICE
    live_instructions = DEFAULT_LIVE_INSTRUCTIONS
    live_delegation_model = ""
    personaplex_url = DEFAULT_PERSONAPLEX_URL
    personaplex_voice = DEFAULT_PERSONAPLEX_VOICE
    personaplex_text_prompt = DEFAULT_PERSONAPLEX_TEXT_PROMPT
    show_backend = advanced_box is not None and mode == "Interactive"
    if show_backend:
        with advanced_box:
            st.subheader("Conversation backend")
            backend_options = list(BACKEND_LABELS.values())
            st.radio(
                "Backend",
                backend_options,
                index=backend_options.index(BACKEND_LABELS[args.default_backend]),
                horizontal=True,
                key="backend",
                persist_state="page",
                help=(
                    "Live and PersonaPlex are full duplex: they stream "
                    "continuously and have no turn boundaries, so there is no "
                    "barge-in truncation."
                ),
            )
    backend = st.session_state.get("backend", BACKEND_LABELS[args.default_backend])
    if mode == "Interactive" and backend == BACKEND_OPENAI_LIVE:
        with picker_box:
            api_key = api_key_input("OPENAI_API_KEY")
            live_voice = st.selectbox(
                "Voice",
                LIVE_VOICES,
                index=LIVE_VOICES.index(DEFAULT_LIVE_VOICE),
                format_func=voice_label,
                key="live_voice",
                help=VOICE_HELP,
            )
        if show_backend:
            with advanced_box:
                st.text_input(
                    "Model", value=DEFAULT_LIVE_MODEL, key="live_model", persist_state="page"
                )
                st.text_area(
                    "Instructions",
                    value=DEFAULT_LIVE_INSTRUCTIONS,
                    height=120,
                    key="live_instructions",
                    persist_state="page",
                )
                st.toggle(
                    "Delegate tasks to a Responses backend",
                    value=False,
                    key="live_delegate",
                    persist_state="page",
                    help=(
                        "Splitting the voice frontend from a backend that reasons "
                        "and calls tools is what separates Live from Realtime. The "
                        "backend model and its tool calls are billed on top of the "
                        "session's per-second charge; without it the avatar "
                        "answers from the Live model alone."
                    ),
                )
                if st.session_state.live_delegate:
                    st.text_input(
                        "Backend model",
                        value=DEFAULT_LIVE_DELEGATION_MODEL,
                        key="live_delegation_model",
                        persist_state="page",
                    )
                    st.caption("Web search is the only tool offered to the backend.")
        live_model = st.session_state.get("live_model", DEFAULT_LIVE_MODEL).strip()
        live_instructions = st.session_state.get(
            "live_instructions", DEFAULT_LIVE_INSTRUCTIONS
        )
        if st.session_state.get("live_delegate", False):
            live_delegation_model = st.session_state.get(
                "live_delegation_model", DEFAULT_LIVE_DELEGATION_MODEL
            ).strip()
    if mode == "Interactive" and backend == BACKEND_PERSONAPLEX:
        with picker_box:
            personaplex_voice = st.selectbox(
                "Voice",
                PERSONAPLEX_VOICES,
                index=PERSONAPLEX_VOICES.index(DEFAULT_PERSONAPLEX_VOICE),
                format_func=voice_label,
                key="personaplex_voice",
                help=VOICE_HELP,
            )
        if show_backend:
            with advanced_box:
                st.text_input(
                    "moshi server URL",
                    value=DEFAULT_PERSONAPLEX_URL,
                    key="personaplex_url",
                    persist_state="page",
                )
                st.text_area(
                    "Persona",
                    value=DEFAULT_PERSONAPLEX_TEXT_PROMPT,
                    height=120,
                    key="personaplex_text_prompt",
                    persist_state="page",
                )
        personaplex_url = st.session_state.get(
            "personaplex_url", DEFAULT_PERSONAPLEX_URL
        ).strip()
        personaplex_text_prompt = st.session_state.get(
            "personaplex_text_prompt", DEFAULT_PERSONAPLEX_TEXT_PROMPT
        )
    if mode == "Interactive" and backend == BACKEND_OPENAI:
        preset_names = list(REALTIME_ENDPOINT_PRESETS) + [CUSTOM_ENDPOINT_LABEL]
        # An endpoint supplied through the environment is by definition not
        # one of the presets, so start on Custom and carry it in.
        default_endpoint = (
            CUSTOM_ENDPOINT_LABEL if DEFAULT_REALTIME_WEBSOCKET_BASE_URL else preset_names[0]
        )
        if show_backend:
            with advanced_box:
                st.selectbox(
                    "Endpoint",
                    preset_names,
                    index=preset_names.index(default_endpoint),
                    key="realtime_endpoint",
                    persist_state="page",
                    help="Anything speaking the OpenAI Realtime protocol.",
                )
        endpoint_name = st.session_state.get("realtime_endpoint", default_endpoint)
        if endpoint_name == CUSTOM_ENDPOINT_LABEL:
            if show_backend:
                with advanced_box:
                    st.text_input(
                        "WebSocket base URL",
                        value=DEFAULT_REALTIME_WEBSOCKET_BASE_URL,
                        key="realtime_ws_base_url",
                        persist_state="page",
                        help="The SDK appends `/realtime`.",
                    )
            realtime_ws_base_url = st.session_state.get(
                "realtime_ws_base_url", DEFAULT_REALTIME_WEBSOCKET_BASE_URL
            ).strip()
            secret_name = "OPENAI_API_KEY"
            endpoint_model = DEFAULT_REALTIME_MODEL
            endpoint_voices = tuple(REALTIME_VOICES)
        else:
            endpoint = REALTIME_ENDPOINT_PRESETS[endpoint_name]
            realtime_ws_base_url = endpoint.base_url
            realtime_declare_audio_format = endpoint.declares_audio_format
            secret_name = endpoint.secret
            endpoint_model = endpoint.model
            endpoint_voices = endpoint.voices
        # Keyed per endpoint so switching presets re-seeds the model (and,
        # below, the voice) rather than carrying the previous provider's
        # names over.
        model_key = f"realtime_model_{endpoint_name}"
        if show_backend:
            with advanced_box:
                if endpoint_name != CUSTOM_ENDPOINT_LABEL and realtime_ws_base_url:
                    st.caption(f"`{realtime_ws_base_url}`")
                if not secret_name:
                    st.info("This endpoint takes no credential.")
                st.text_input(
                    "Model",
                    value=endpoint_model,
                    key=model_key,
                    persist_state="page",
                    help="Sent in the connection URL, so a wrong name fails the handshake.",
                )
                if not endpoint_voices:
                    st.caption("Voice is configured on the server.")
                st.text_area(
                    "Instructions",
                    value=DEFAULT_REALTIME_INSTRUCTIONS,
                    height=120,
                    key="realtime_instructions",
                    persist_state="page",
                )
        realtime_model = st.session_state.get(model_key, endpoint_model).strip()
        with picker_box:
            if secret_name:
                api_key = api_key_input(secret_name)
            if endpoint_voices:
                realtime_voice = st.selectbox(
                    "Voice",
                    endpoint_voices,
                    format_func=voice_label,
                    key=f"realtime_voice_{endpoint_name}",
                    help=VOICE_HELP,
                )
        realtime_instructions = st.session_state.get(
            "realtime_instructions", DEFAULT_REALTIME_INSTRUCTIONS
        )

    with picker_box:
        if user_registry.can_register:
            with st.expander("Your own avatars"):
                registered_flash = st.session_state.pop("avatar_registered_flash", None)
                if registered_flash is not None:
                    elapsed_s = registered_flash.get("elapsed_s")
                    took = f" in {elapsed_s:.0f} s" if elapsed_s else ""
                    st.success(
                        f"Registered{took} and selected "
                        f"`{registered_flash['avatar_id']}`."
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
                                # The gallery above rendered before this
                                # handler ran; rerun so it shows the new
                                # avatar, carrying the confirmation across as
                                # a flash.
                                st.session_state[APPEARANCE_KEY] = f"gagavatar:{avatar_id}"
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
            gagavatar_runtime = load_photoreal_renderer()
            if client_assets is not None:
                gagavatar = ClientRenderAdapter(gagavatar_runtime, client_assets)
            else:
                gagavatar = StreamingGAGAvatarAdapter(gagavatar_runtime, user_registry)
            gagavatar_flame = gagavatar_runtime.flame_model
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
            # Client rendering switches avatars on the live pipeline.
            appearance if client_assets is None else None,
            client_assets is not None,
            style_id,
            str(asset_dir),
            str(model_path) if model_path else None,
            str(tracked_path) if tracked_path else None,
        )
        # A run that waited for the build lock must adopt the pipeline the
        # run ahead of it built, not build a twin.
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
            if client_assets is not None:
                pipeline_cls = ClientRenderPipeline
            elif args.ai_badge:
                pipeline_cls = BadgedARTalkPipeline
            else:
                pipeline_cls = ARTalkPipeline
            pipeline = pipeline_cls(
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
                silence_mouth_gate=args.silence_mouth_gate,
                blink_injector=args.blink_injector,
                blink_scale=args.blink_scale,
                blink_lip_cancel=args.blink_lip_cancel,
                renderer_output_uint8=args.render_uint8_gpu,
                warm_key_extra=(
                    f"fp16={args.renderer_fp16},compile={args.renderer_compile},"
                    f"motion={motion_model},client={client_assets is not None}"
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
        if client_assets is not None:
            pipeline.set_avatar(avatar_id)
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
        with status_box:
            st.info("Enter an API key under Mode to use Interactive mode.")
        st.stop()

    session_id = current_session_id()

    def render_ended_notice(key: str) -> None:
        if st.session_state[SESSION_ENDED_KEY] == ENDED_NOT_STARTED:
            st.info(
                f"START was not pressed within {args.start_grace:g} s while "
                "another visitor was waiting, so the GPU went to them."
            )
            label = "Join the queue again"
        else:
            st.info(
                f"This session reached its {args.session_time_limit:g}-second "
                "limit and was stopped, which frees the GPU for other visitors."
            )
            label = "Start a new session"
        if st.button(label, key=key):
            st.session_state.pop(SESSION_ENDED_KEY, None)
            st.rerun(scope="app")

    ended = slots.pop_ended(session_id)
    if ended is not None:
        st.session_state[SESSION_ENDED_KEY] = ended
    if SESSION_ENDED_KEY in st.session_state:
        # The reaper already stopped what the slot held; this drops the
        # session's own references, and leaving the WebRTC component
        # unrendered closes the browser's connection.
        stop_bridge()
        stop_pipeline()
        st.session_state.pop(QUEUED_KEY, None)
        with status_box:
            render_ended_notice("restart_after_end")
            if recorder is not None:
                render_feedback_slot(recorder)
        st.stop()

    if not slots.acquire(session_id):
        # Reached here after a takeover too, so drop whatever this session was
        # still running.
        stop_pipeline()
        st.session_state[QUEUED_KEY] = True

        with status_box:

            @st.fragment(run_every="1s")
            def render_queue_status() -> None:
                ended = slots.pop_ended(session_id)
                if ended is not None:
                    st.session_state[SESSION_ENDED_KEY] = ended
                    st.rerun(scope="app")
                if slots.acquire(session_id):
                    st.rerun(scope="app")
                position = slots.queue_position(session_id)
                if position is None:
                    # Admitted since the acquire above.
                    st.rerun(scope="app")
                st.warning(
                    "Every GPU session is in use. "
                    f"You are number {position} in the queue."
                )
                wait = slots.estimated_wait(session_id, args.session_time_limit)
                if wait is None:
                    st.caption("Estimated wait: unknown, since sessions have no time limit.")
                else:
                    st.caption(f"Estimated wait: about {format_wait(wait)}.")
                st.html(queue_title_script(position), unsafe_allow_javascript=True)

            render_queue_status()
            st.caption(
                "Keep this tab open: the app appears here by itself when it is "
                f"your turn, and you then have {args.start_grace:g} s to press "
                "START. Leaving or reloading the page gives up your place."
            )
            # Outside the fragment, which would replace the button every second
            # and lose clicks.
            st.html(TURN_ALERT_OPT_IN, unsafe_allow_javascript=True)
            if not args.session_time_limit:
                if st.button("Take the GPU over"):
                    slots.take_over(session_id)
                    st.rerun()
                st.caption(
                    "Taking over ends the session in use, for when it is stuck "
                    "rather than in use. Without a time limit nothing else frees it."
                )
            if recorder is not None:
                # Answering while queued lets the first session be recorded.
                render_feedback_slot(recorder)
        st.stop()

    try:
        with _MODEL_BUILD_LOCK:
            pipeline = get_pipeline()
    except Exception as exc:
        with status_box:
            st.error(f"Failed to initialize ARTalk avatar pipeline: {exc}")
            # Initialization spans several optional model paths whose imports
            # and checkpoints fail in ways the message alone does not locate.
            st.exception(exc)
        st.stop()
    if not slots.attach(session_id, "pipeline", pipeline):
        # Taken over while this pipeline was being built; it is untracked, so
        # stop it here rather than leave it running against a lost slot.
        pipeline.stop()
        st.rerun()

    freeze_loaded_objects()

    # PersonaPlex streams continuously, including its own silence, so the pump
    # would feed the pipeline a second time and fight the model's audio.
    if mode == "Interactive" and backend == BACKEND_PERSONAPLEX:
        stop_silence_pump()
        silence_pump = None
    else:
        silence_pump = get_silence_pump(
            pipeline, args.idle_motion, continuous_input=mode == "Loopback"
        )
    slots.attach(session_id, "pump", silence_pump)
    pipeline.set_output_underrun_policy(
        underrun_policy,
        rebuffer_seconds=rebuffer_seconds,
        max_added_latency_seconds=max_added_latency,
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
            realtime_declare_audio_format,
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
                    declare_audio_format=realtime_declare_audio_format,
                    event_log=event_log,
                )
            st.session_state[BRIDGE_KEY] = bridge
            st.session_state[BRIDGE_CONFIG_KEY] = config
        bridge.reply_tap = recorder.record_reply if recorder is not None else None
        return bridge

    bridge = get_bridge() if mode == "Interactive" else None
    if args.session_time_limit:
        # The time limit is enforced off the script thread, which can reach
        # the bridge only through the slot.
        slots.attach(session_id, "bridge", bridge)
    if bridge is not None and not bridge.is_running:
        # PersonaPlex withholds its handshake until the system prompts have
        # been stepped through the model, which is far slower than a hosted API.
        connect_timeout = 60.0 if backend == BACKEND_PERSONAPLEX else 8.0
        with status_box:
            with st.spinner(f"Connecting to {backend}..."):
                bridge.start()
                bridge.wait_until_connected(timeout=connect_timeout)
            snap = bridge.snapshot()
            if snap["error"]:
                st.error(f"{backend} error: {snap['error']}")
            elif not snap["connected"]:
                st.warning(f"{backend} is still connecting. Wait a moment before START.")

    def on_loopback_audio_frame(frame: av.AudioFrame) -> None:
        # A pipeline stopped off the script thread (time limit, take-over)
        # keeps receiving the microphone until the page reruns, and its
        # input queue has no bound and no reader any more.
        if pipeline.is_stopped:
            return
        if recorder is not None:
            recorder.record_mic(frame)
        silence_pump.mark_input()
        pipeline.push_audio_frame(frame)

    def on_interactive_audio_frame(frame: av.AudioFrame) -> None:
        if recorder is not None:
            recorder.record_mic(frame)
        if bridge is not None:
            bridge.push_input(frame)

    def on_audio_ended() -> None:
        if recorder is not None:
            recorder.end_media("disconnected")
        stop_bridge()
        # The pipeline stays with the slot: STOP ends this track too, and the
        # visitor may press START again, while a closed tab is left to the
        # reaper.

    # With client rendering the browser draws the face; there is no video.
    video_source_track = (
        create_video_source_track(
            pipeline.video_source_callback,
            key=f"artalk_video_source_{mode.lower()}",
            fps=ARTALK_FPS,
        )
        if client_assets is None
        else None
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
            if recorder is not None:
                recorder.end_media("stopped")
            # Kept rather than rebuilt, as for a visitor admitted who has not
            # pressed START yet: the slot decides when it goes. The flush
            # keeps a resumed session from playing what was queued at STOP.
            pipeline.flush_output(discard_in_flight=True)
            slots.mark_stopped(session_id)

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
        status_box.warning(
            f"ICE provider {args.ice_provider!r} failed ({exc}); using STUN "
            "only, which cannot connect through a proxy or a strict NAT."
        )
        rtc_configuration = STUN_ONLY

    # Passed only when client rendering is on: the streamlit-webrtc release
    # this app pins has none of these parameters.
    client_render_kwargs = {}
    if client_assets is not None:
        client_render_kwargs = {
            "data_channels": {CHANNEL_LABEL: CHANNEL_OPTIONS},
            "frontend_script": FRONTEND_SCRIPT,
            "frontend_data": {
                "dirs": client_dirs,
                "avatar": pipeline.client_renderer.avatar_id,
                "generation": pipeline.client_renderer.generation,
                "network": client_network,
                "quantStep": QUANT_STEP,
                "ortUrl": None if client_assets.has_local_ort else ORT_CDN_URL,
                "graphCapture": client_assets.ort_patched,
                "aiBadge": args.ai_badge,
            },
        }

    def render_webrtc_component():
        return webrtc_streamer(
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
            **client_render_kwargs,
        )

    @st.fragment(run_every="1s")
    def render_session_status_fragment() -> None:
        ended = slots.pop_ended(session_id)
        if ended is not None:
            st.session_state[SESSION_ENDED_KEY] = ended
        if SESSION_ENDED_KEY in st.session_state:
            # Not a full rerun to the notice above: a run_every fragment that
            # reruns the app into a different layout races the page's other
            # fragments, and the browser can fail to render the result
            # ("'setIn' cannot be called on an ElementNode").
            render_ended_notice("restart_after_end_inline")
            # The page under the video was last drawn while the session
            # played, so its form slot stays empty until the next full run.
            # A start-grace end follows a STOP, whose run drew the form there.
            if (
                recorder is not None
                and st.session_state[SESSION_ENDED_KEY] == ENDED_TIME_LIMIT
            ):
                render_feedback_form(recorder)
            return
        now = time.monotonic()
        deadline = slots.deadline(session_id)
        start_by = slots.start_by(session_id)
        if deadline is not None:
            remaining = max(deadline - now, 0.0)
            if remaining <= SESSION_END_WARNING_SECONDS:
                st.warning(f"This session ends in {remaining:.0f} s.")
            else:
                st.caption(f"Time left in this session: {remaining:.0f} s")
            return
        if st.session_state.get(QUEUED_KEY):
            message = "It is your turn. Press **START** under the video to begin."
        else:
            message = "Another visitor is waiting for the GPU."
        if start_by is not None:
            st.success(
                f"{message} Your slot is held for "
                f"{max(start_by - now, 0.0):.0f} more seconds unless you "
                "press START."
            )
        elif st.session_state.get(QUEUED_KEY):
            st.success(message)
        if st.session_state.get(QUEUED_KEY):
            st.html(TURN_ALERT_SCRIPT, unsafe_allow_javascript=True)
        time_left = slots.time_left(session_id)
        if time_left is not None:
            st.caption(
                f"Time left in this session: {time_left:.0f} s, paused until "
                "you press START."
            )
        elif args.session_time_limit:
            st.caption(
                f"Sessions are limited to {args.session_time_limit:g} s of "
                "playing time."
            )

    with avatar_col:
        streamer_ctx = render_webrtc_component()
        if client_assets is not None:
            pipeline.channel_source = lambda: (streamer_ctx.data_channels or {}).get(
                CHANNEL_LABEL
            )
        ctx = st.session_state.get(streamer_key)
        if ctx is not None and (ctx.state.playing or ctx.state.signalling):
            slots.mark_started(session_id)
            st.session_state.pop(QUEUED_KEY, None)
        # Checked on every run rather than when START changes the state:
        # a new session started from the notice inherits a connection
        # that is still playing, so no state change starts its clock.
        if args.session_time_limit and ctx is not None and ctx.state.playing:
            slots.start_clock(session_id, args.session_time_limit)
        if args.session_time_limit or args.max_sessions:
            with status_box:
                render_session_status_fragment()
        if recorder is not None:
            recorder.pipeline = pipeline
            # The slot stops the recorder whenever the server stops the
            # session itself, which closes its record.
            slots.attach(session_id, "recorder", recorder)
            ctx = st.session_state.get(streamer_key)
            recorder.prepare_media(
                MediaSettings(
                    mode=mode.lower(),
                    backend=backend if mode == "Interactive" else None,
                    motion_model=motion_model,
                    appearance_kind=appearance_kind(appearance, user_avatar_ids),
                    style=style_id,
                    session_time_limit_s=args.session_time_limit,
                ),
                playing=ctx is not None and ctx.state.playing,
            )
            render_feedback_slot(recorder)
        if bridge is not None:
            render_interactive_status()

    if advanced_box is None:
        return

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

    with advanced_box:
        st.subheader("Diagnostics")
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
