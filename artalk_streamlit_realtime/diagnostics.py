"""Streamlit diagnostics for the realtime pipeline."""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from artalk.realtime_pipeline import ARTalkPipeline


@st.cache_data(show_spinner=False)
def _read_trace_bytes(path: str, size: int) -> bytes:
    # Trace files are written once and never modified; `size` busts the cache
    # in the unlikely case a path is reused.
    return Path(path).read_bytes()


MAX_PROFILER_TRACES_LISTED = 6


def render_profiler_panel(
    pipeline: ARTalkPipeline, trace_root: str | None = None
) -> None:
    status = pipeline.profiler_status()
    run_dir = status["run_dir"]
    if run_dir is None:
        st.caption(
            "Torch profiler is off. Launch with `--profile-trace-dir` to "
            "capture traces."
        )
        return

    if status["last_error"]:
        st.error(f"Profiler capture failed: {status['last_error']}")

    counters = pipeline.metrics_snapshot()["counters"]
    chunk_samples = int(metric_value(counters, "streamer_chunk_samples", 64000.0))
    buffered = int(metric_value(counters, "streamer_buffer_samples"))
    skip = status["skip_chunks"]
    st.progress(
        min(buffered / max(chunk_samples, 1), 1.0),
        text=(
            f"Next ARTalk chunk: {buffered}/{chunk_samples} audio samples "
            f"buffered — chunks so far: {status['chunks_seen']} "
            f"(first {skip} skipped as warm-up), captured "
            f"{status['chunks_captured']}/{status['max_chunks']} this session"
        ),
    )

    # Pipelines are recycled on session stop/restart, and each new pipeline
    # writes to a fresh run directory. List traces from the whole trace root
    # so captures survive restarts in the UI.
    root = Path(trace_root) if trace_root else Path(run_dir).parent
    traces = (
        sorted(
            root.glob("*/chunk-*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if root.exists()
        else []
    )

    if not traces:
        st.info(
            "Capturing starts when a 4-second ARTalk chunk fills with audio. "
            "In **Loopback** mode the microphone stream fills one chunk every "
            "~4 s. In **Interactive** mode chunks fill only while the "
            "**assistant** is speaking (trailing silence is filled slowly by "
            "the silence pump), so hold a conversation with responses longer "
            "than ~8 s or launch with `--profile-skip-chunks 0` to capture "
            "the first chunk. Stopping the session or changing settings "
            "restarts the capture counters."
        )
        st.caption(f"Traces will be written to `{run_dir}` on the server.")
        return

    if len(traces) > MAX_PROFILER_TRACES_LISTED:
        st.caption(
            f"Showing the latest {MAX_PROFILER_TRACES_LISTED} of "
            f"{len(traces)} traces under `{root}`."
        )
    for trace_path in traces[:MAX_PROFILER_TRACES_LISTED]:
        run_name = trace_path.parent.name
        is_current = Path(run_dir) == trace_path.parent
        captured_at = time.strftime(
            "%H:%M:%S", time.localtime(trace_path.stat().st_mtime)
        )
        label = f"{run_name} / {trace_path.name} — {captured_at}"
        if is_current:
            label += " (current session)"
        with st.expander(label, expanded=is_current):
            summary_path = trace_path.with_name(trace_path.stem + "-summary.txt")
            widget_key = f"{run_name}_{trace_path.stem}"
            if summary_path.exists():
                summary_table = summary_path.read_text()
                st.code(summary_table, language="text")
                st.download_button(
                    "Download operator summary (.txt)",
                    data=summary_table,
                    file_name=f"{run_name}-{summary_path.name}",
                    mime="text/plain",
                    key=f"profiler_summary_dl_{widget_key}",
                )
            else:
                st.caption("No operator summary was saved for this capture.")
            st.download_button(
                f"Download full Chrome trace ({trace_path.stat().st_size / 1e6:.1f} MB)",
                data=_read_trace_bytes(str(trace_path), trace_path.stat().st_size),
                file_name=f"{run_name}-{trace_path.name}",
                mime="application/json",
                key=f"profiler_trace_dl_{widget_key}",
            )
    st.markdown(
        "The table in each capture is the per-operator summary "
        "(`key_averages`, sorted by self CUDA time). The **full log** is the "
        "Chrome trace JSON: download it and open it in "
        "<https://ui.perfetto.dev> (drag & drop) or `chrome://tracing`. "
        "Pipeline stages appear as `artalk.*` spans; look for `cudaMemcpy` / "
        "`cudaStreamSynchronize` blocks inside them. Traces are also on the "
        f"server under `{root}`."
    )


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


def display_metric_name(key: str) -> str:
    return (
        key.replace("frame_pre_model", "frame_pre_artalk")
        .replace("pre_model", "pre_artalk")
        .replace("post_model", "post_artalk")
    )


def duration_rows_for_keys(durations: dict, keys: list[str]) -> list[dict]:
    return [duration_row(key, durations, key) for key in keys if key in durations]


def duration_rows_for_prefixes(durations: dict, prefixes: tuple[str, ...]) -> list[dict]:
    return [
        duration_row(key, durations, key)
        for key in sorted(durations)
        if key.startswith(prefixes)
    ]


def duration_stat(durations: dict, key: str) -> dict:
    stat = durations.get(key, {})
    return {
        "count": int(stat.get("count", 0)),
        "last_ms": float(stat.get("last_ms", 0.0)),
        "avg_ms": float(stat.get("avg_ms", 0.0)),
        "max_ms": float(stat.get("max_ms", 0.0)),
    }


def duration_text(stat: dict, field: str = "last_ms") -> str:
    if int(stat.get("count", 0)) <= 0:
        return "-"
    return f"{float(stat.get(field, 0.0)) / 1000.0:.2f}s"


def seconds_text(value: float) -> str:
    return f"{value:.2f}s"


def fixed_metric_value(value: str) -> str:
    return f"{value:>7}"


def latency_label(name: str, value: str) -> str:
    return f"{name:<23} [{fixed_metric_value(value)}]"


def metric_line(name: str, value: str) -> str:
    return f"    - {name:<28} {fixed_metric_value(value)}"


def diagnostic_line(name: str, value: str) -> str:
    return f"    * {name:<28} {fixed_metric_value(value)}"


def diagram_note(line: str, note: str = "") -> str:
    if not note:
        return line
    return f"{line:<118} # {note}"


def diagram_columns(total_col: str = "", breakdown_col: str = "", note: str = "") -> str:
    return diagram_note(f"  |               | {total_col:<36} | {breakdown_col}", note)


def duration_value(durations: dict, key: str) -> str:
    return duration_text(duration_stat(durations, key))


def spike_context_text(context: dict) -> str:
    if not context:
        return "-"
    preferred_keys = [
        "pipeline_call",
        "segment_index",
        "segment_start_frame",
        "segment_frames",
        "render_batch_index",
        "render_start_frame",
        "render_batch_frames",
        "segment_excess_s",
        "audio_in_queue_depth",
        "video_queue_depth",
        "audio_out_buffer_s",
    ]
    parts = []
    for key in preferred_keys:
        if key not in context:
            continue
        value = context[key]
        if isinstance(value, float):
            value = round(value, 3)
        parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else str(context)


def render_pipeline_diagnostics(pipeline: ARTalkPipeline) -> None:
    snapshot = pipeline.metrics_snapshot()
    counters = snapshot["counters"]
    durations = snapshot["durations"]
    spikes = snapshot.get("spikes", [])

    st.subheader("Pipeline diagnostics")
    total_latency = duration_stat(durations, "frame_audio_to_video_served_latency")
    published_latency = duration_stat(durations, "frame_audio_to_video_midpoint_latency")
    frame_first_latency = duration_stat(durations, "frame_audio_to_video_first_latency")
    frame_last_latency = duration_stat(durations, "frame_audio_to_video_last_latency")
    post_artalk_latency = duration_stat(durations, "post_model_latency")
    post_artalk_excess_latency = duration_stat(durations, "post_model_excess_latency")
    pre_render_excess_latency = duration_stat(
        durations,
        "pre_render_wait_segment_excess_over_media",
    )
    render_latency = duration_stat(durations, "segment_render_latency")
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

    with st.expander("Overview", expanded=False):
        overview_cols = st.columns(4)
        overview_cols[0].metric("Frame served latency", duration_text(total_latency))
        overview_cols[1].metric(
            "Render realtime ratio",
            f"{metric_value(counters, 'last_render_realtime_ratio'):.2f}x",
        )
        overview_cols[2].metric("Video real FPS", f"{video_real_fps:.1f}")
        overview_cols[3].metric(
            "Audio out buffer",
            seconds_text(metric_value(counters, "audio_out_buffer_samples") / 16000.0),
        )

        latency_cols = st.columns(4)
        latency_cols[0].metric("Frame published", duration_text(published_latency))
        latency_cols[1].metric("Published first", duration_text(frame_first_latency))
        latency_cols[2].metric("Published last", duration_text(frame_last_latency))
        latency_cols[3].metric("Post-ARTalk", duration_text(post_artalk_latency))

        post_artalk_cols = st.columns(4)
        post_artalk_cols[0].metric(
            "Post-ARTalk excess",
            duration_text(post_artalk_excess_latency),
        )
        post_artalk_cols[1].metric(
            "Pre-render excess",
            duration_text(pre_render_excess_latency),
        )
        post_artalk_cols[2].metric("Render", duration_text(render_latency))
        post_artalk_cols[3].metric(
            "Serve wait",
            duration_text(duration_stat(durations, "frame_publish_to_serve_latency")),
        )

        segment_cols = st.columns(4)
        segment_cols[0].metric(
            "Segment",
            "#{} f{}+{}".format(
                int(metric_value(counters, "last_pre_render_wait_segment_index")),
                int(metric_value(counters, "last_pre_render_wait_segment_start_frame")),
                int(metric_value(counters, "last_pre_render_wait_segment_frames")),
            ),
        )
        segment_cols[1].metric(
            "Audio underruns",
            int(metric_value(counters, "audio_playback_underrun_frames")),
        )
        segment_cols[2].metric(
            "Render to publish",
            duration_text(duration_stat(durations, "frame_render_to_publish_latency")),
        )
        segment_cols[3].metric(
            "Render to serve",
            duration_text(duration_stat(durations, "frame_render_to_serve_latency")),
        )

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
        duration_row("Frame audio to video", durations, "frame_audio_to_video_latency"),
        duration_row("Frame audio to video first", durations, "frame_audio_to_video_first_latency"),
        duration_row("Frame audio to video midpoint", durations, "frame_audio_to_video_midpoint_latency"),
        duration_row("Frame audio to video last", durations, "frame_audio_to_video_last_latency"),
        duration_row("Frame audio to video served", durations, "frame_audio_to_video_served_latency"),
        duration_row("Frame publish to serve", durations, "frame_publish_to_serve_latency"),
        duration_row("Frame pre-ARTalk", durations, "frame_pre_model_latency"),
        duration_row("Frame pre-ARTalk first", durations, "frame_pre_model_first_latency"),
        duration_row("Frame pre-ARTalk midpoint", durations, "frame_pre_model_midpoint_latency"),
        duration_row("Frame pre-ARTalk last", durations, "frame_pre_model_last_latency"),
        duration_row("Frame pre-ARTalk wait first", durations, "frame_pre_model_wait_first"),
        duration_row("Frame pre-ARTalk wait midpoint", durations, "frame_pre_model_wait_midpoint"),
        duration_row("Frame pre-ARTalk wait last", durations, "frame_pre_model_wait_last"),
        duration_row("Frame render", durations, "frame_render_latency"),
        duration_row("Frame render midpoint", durations, "frame_render_midpoint_latency"),
        duration_row("Frame render to publish", durations, "frame_render_to_publish_latency"),
        duration_row("Frame render to publish midpoint", durations, "frame_render_to_publish_midpoint_latency"),
        duration_row("Frame render to serve", durations, "frame_render_to_serve_latency"),
        duration_row("Audio to video publish", durations, "audio_to_video_latency"),
        duration_row("Midpoint audio to video", durations, "audio_midpoint_to_video_latency"),
        duration_row("Pre-ARTalk latency", durations, "pre_model_latency"),
        duration_row("Pre-ARTalk chunk wait", durations, "pre_model_chunk_wait"),
        duration_row("ARTalk streamer compute", durations, "pre_model_streamer_compute"),
        duration_row("Post-ARTalk latency", durations, "post_model_latency"),
        duration_row("Post-ARTalk excess latency", durations, "post_model_excess_latency"),
        duration_row("Post-ARTalk render window", durations, "post_model_render_window"),
        duration_row("Post-ARTalk publish overhead", durations, "post_model_publish_overhead"),
        duration_row("Post-ARTalk smoother", durations, "post_model_smoother_feed"),
        duration_row("Post-ARTalk audio emit prepare", durations, "post_model_audio_emit_prepare"),
        duration_row("Post-ARTalk segment audio pairing", durations, "post_model_segment_audio_pairing"),
        duration_row("Post-ARTalk publish segment", durations, "post_model_publish_segment"),
        duration_row("Post-ARTalk publish video queue", durations, "post_model_publish_video_queue"),
        duration_row("Post-ARTalk publish audio buffer", durations, "post_model_publish_audio_buffer"),
        duration_row("Post-ARTalk publish metrics", durations, "post_model_publish_metrics_update"),
        duration_row("Pre-render wait", durations, "pre_render_wait_latency"),
        duration_row("Pre-render first render start", durations, "pre_render_wait_first_render_start"),
        duration_row("Pre-render segment offset", durations, "pre_render_wait_segment_start_offset"),
        duration_row("Pre-render segment media offset", durations, "pre_render_wait_segment_media_offset"),
        duration_row("Pre-render excess over media", durations, "pre_render_wait_segment_excess_over_media"),
        duration_row("Pre-render prior segment backlog", durations, "pre_render_wait_prior_segment_backlog"),
        duration_row("Segment render latency", durations, "segment_render_latency"),
    ]
    output_rows = [
        {"name": "audio frames pushed", "value": int(metric_value(counters, "audio_frames_pushed"))},
        {"name": "audio samples fed", "value": int(metric_value(counters, "audio_samples_fed_to_streamer"))},
        {"name": "silence samples queued", "value": int(metric_value(counters, "silence_pump_samples"))},
        {"name": "silence pump chunks", "value": int(metric_value(counters, "silence_pump_chunks"))},
        {"name": "silence pump skips", "value": int(metric_value(counters, "silence_pump_backpressure_skips"))},
        {"name": "silence pump input-queue skips", "value": int(metric_value(counters, "silence_pump_input_queue_skips"))},
        {"name": "silence pump worker-busy skips", "value": int(metric_value(counters, "silence_pump_worker_busy_skips"))},
        {"name": "silence pump recent-input skips", "value": int(metric_value(counters, "silence_pump_recent_input_skips"))},
        {"name": "silence pump pacing skips", "value": int(metric_value(counters, "silence_pump_pacing_skips"))},
        {"name": "silence pump input idle seconds", "value": round(metric_value(counters, "silence_pump_input_idle_s"), 3)},
        {"name": "silence pump idle threshold", "value": round(metric_value(counters, "silence_pump_idle_seconds"), 3)},
        {"name": "silence pump audio buffer", "value": int(metric_value(counters, "silence_pump_audio_buffer_samples"))},
        {"name": "silence pump video queue", "value": int(metric_value(counters, "silence_pump_video_queue_depth"))},
        {"name": "silence pump worker busy", "value": int(metric_value(counters, "silence_pump_worker_busy"))},
        {"name": "motion chunks", "value": int(metric_value(counters, "motion_chunks_produced"))},
        {"name": "motion frames", "value": int(metric_value(counters, "motion_frames_produced"))},
        {"name": "smoothed frames", "value": int(metric_value(counters, "smoothed_frames_produced"))},
        {"name": "render batch size", "value": int(metric_value(counters, "render_batch_size"))},
        {"name": "render batches", "value": int(metric_value(counters, "render_batches"))},
        {"name": "render batch frames", "value": int(metric_value(counters, "render_batch_frames"))},
        {"name": "last render chunk frames", "value": int(metric_value(counters, "last_render_chunk_frames"))},
        {"name": "last render chunk seconds", "value": round(metric_value(counters, "last_render_chunk_s"), 3)},
        {"name": "last render media seconds", "value": round(metric_value(counters, "last_render_chunk_media_s"), 3)},
        {"name": "render realtime ratio", "value": round(metric_value(counters, "last_render_realtime_ratio"), 3)},
        {"name": "last frame audio-to-video first latency seconds", "value": round(metric_value(counters, "last_frame_audio_to_video_first_latency_s"), 3)},
        {"name": "last frame audio-to-video midpoint latency seconds", "value": round(metric_value(counters, "last_frame_audio_to_video_midpoint_latency_s"), 3)},
        {"name": "last frame audio-to-video last latency seconds", "value": round(metric_value(counters, "last_frame_audio_to_video_last_latency_s"), 3)},
        {"name": "last frame audio-to-video served latency seconds", "value": round(metric_value(counters, "last_frame_audio_to_video_served_latency_s"), 3)},
        {"name": "last frame pre-ARTalk first latency seconds", "value": round(metric_value(counters, "last_frame_pre_model_first_latency_s"), 3)},
        {"name": "last frame pre-ARTalk midpoint latency seconds", "value": round(metric_value(counters, "last_frame_pre_model_midpoint_latency_s"), 3)},
        {"name": "last frame pre-ARTalk last latency seconds", "value": round(metric_value(counters, "last_frame_pre_model_last_latency_s"), 3)},
        {"name": "last frame pre-ARTalk wait first seconds", "value": round(metric_value(counters, "last_frame_pre_model_wait_first_s"), 3)},
        {"name": "last frame pre-ARTalk wait midpoint seconds", "value": round(metric_value(counters, "last_frame_pre_model_wait_midpoint_s"), 3)},
        {"name": "last frame pre-ARTalk wait last seconds", "value": round(metric_value(counters, "last_frame_pre_model_wait_last_s"), 3)},
        {"name": "last frame render midpoint latency seconds", "value": round(metric_value(counters, "last_frame_render_midpoint_latency_s"), 3)},
        {"name": "last frame render to publish midpoint latency seconds", "value": round(metric_value(counters, "last_frame_render_to_publish_midpoint_latency_s"), 3)},
        {"name": "last frame publish to serve latency seconds", "value": round(metric_value(counters, "last_frame_publish_to_serve_latency_s"), 3)},
        {"name": "last frame render to serve latency seconds", "value": round(metric_value(counters, "last_frame_render_to_serve_latency_s"), 3)},
        {"name": "min frame audio-to-video latency seconds", "value": round(metric_value(counters, "min_frame_audio_to_video_latency_s"), 3)},
        {"name": "max frame audio-to-video latency seconds", "value": round(metric_value(counters, "max_frame_audio_to_video_latency_s"), 3)},
        {"name": "min frame audio-to-video served latency seconds", "value": round(metric_value(counters, "min_frame_audio_to_video_served_latency_s"), 3)},
        {"name": "max frame audio-to-video served latency seconds", "value": round(metric_value(counters, "max_frame_audio_to_video_served_latency_s"), 3)},
        {"name": "last audio-to-video latency seconds", "value": round(metric_value(counters, "last_audio_to_video_latency_s"), 3)},
        {"name": "last midpoint audio-to-video latency seconds", "value": round(metric_value(counters, "last_midpoint_audio_to_video_latency_s"), 3)},
        {"name": "last pre-ARTalk latency seconds", "value": round(metric_value(counters, "last_pre_model_latency_s"), 3)},
        {"name": "last post-ARTalk latency seconds", "value": round(metric_value(counters, "last_post_model_latency_s"), 3)},
        {"name": "last post-ARTalk excess latency seconds", "value": round(metric_value(counters, "last_post_model_excess_latency_s"), 3)},
        {"name": "last pre-render wait latency seconds", "value": round(metric_value(counters, "last_pre_render_wait_latency_s"), 3)},
        {"name": "last pre-render segment index", "value": int(metric_value(counters, "last_pre_render_wait_segment_index"))},
        {"name": "last pre-render segment start frame", "value": int(metric_value(counters, "last_pre_render_wait_segment_start_frame"))},
        {"name": "last pre-render segment frames", "value": int(metric_value(counters, "last_pre_render_wait_segment_frames"))},
        {"name": "last pre-render segment render offset seconds", "value": round(metric_value(counters, "last_pre_render_wait_segment_render_offset_s"), 3)},
        {"name": "last pre-render segment media offset seconds", "value": round(metric_value(counters, "last_pre_render_wait_segment_media_offset_s"), 3)},
        {"name": "last pre-render excess over media seconds", "value": round(metric_value(counters, "last_pre_render_wait_segment_excess_s"), 3)},
        {"name": "last segment render latency seconds", "value": round(metric_value(counters, "last_segment_render_latency_s"), 3)},
        {"name": "min audio-to-video latency seconds", "value": round(metric_value(counters, "min_audio_to_video_latency_s"), 3)},
        {"name": "max audio-to-video latency seconds", "value": round(metric_value(counters, "max_audio_to_video_latency_s"), 3)},
        {"name": "last audio samples emitted", "value": int(metric_value(counters, "last_audio_samples_emitted"))},
        {"name": "audio playback started", "value": int(metric_value(counters, "audio_playback_started"))},
        {"name": "audio prebuffer seconds", "value": round(metric_value(counters, "output_audio_prebuffer_seconds"), 3)},
        {"name": "audio prebuffer samples", "value": int(metric_value(counters, "output_audio_prebuffer_samples"))},
        {"name": "output segment seconds", "value": round(metric_value(counters, "output_segment_seconds"), 3)},
        {"name": "output segment min frames", "value": int(metric_value(counters, "output_segment_min_frames"))},
        {"name": "synced audio samples served", "value": int(metric_value(counters, "synced_audio_samples_served"))},
        {"name": "synced audio frame index", "value": int(metric_value(counters, "synced_audio_frame_index"))},
        {"name": "cumulative rendered FPS", "value": round(cumulative_rendered_fps, 1)},
        {"name": "rendered frames", "value": int(metric_value(counters, "rendered_frames"))},
        {"name": "output segments", "value": int(metric_value(counters, "output_segments_published"))},
        {"name": "output segment frames", "value": int(metric_value(counters, "output_segment_frames"))},
        {"name": "output segment audio samples", "value": int(metric_value(counters, "output_segment_audio_samples"))},
        {"name": "last output segment frames", "value": int(metric_value(counters, "last_output_segment_frames"))},
        {"name": "last output segment audio samples", "value": int(metric_value(counters, "last_output_segment_audio_samples"))},
        {"name": "video callbacks", "value": int(metric_value(counters, "video_callbacks"))},
        {"name": "video callback FPS", "value": round(video_callback_fps, 1)},
        {"name": "video frames served", "value": int(metric_value(counters, "video_frames_served"))},
        {"name": "video real-frame FPS", "value": round(video_real_fps, 1)},
        {"name": "video placeholders", "value": int(metric_value(counters, "video_placeholder_frames"))},
        {"name": "video placeholder FPS", "value": round(video_placeholder_fps, 1)},
        {"name": "video frames dropped", "value": int(metric_value(counters, "video_frames_dropped"))},
        {"name": "video frames dropped for sync", "value": int(metric_value(counters, "video_frames_dropped_for_sync"))},
        {"name": "video target frame index", "value": int(metric_value(counters, "last_video_target_frame_index"))},
        {"name": "video frame index served", "value": int(metric_value(counters, "last_video_frame_index_served"))},
        {"name": "video frame index enqueued", "value": int(metric_value(counters, "last_video_frame_index_enqueued"))},
        {"name": "video lead frames", "value": int(metric_value(counters, "video_lead_frames"))},
        {"name": "video lead seconds", "value": round(metric_value(counters, "video_lead_seconds"), 3)},
        {"name": "min video lead frames", "value": int(metric_value(counters, "min_video_lead_frames"))},
        {"name": "max video lead frames", "value": int(metric_value(counters, "max_video_lead_frames"))},
        {"name": "video queue first frame", "value": int(metric_value(counters, "video_queue_first_frame_index"))},
        {"name": "video queue last frame", "value": int(metric_value(counters, "video_queue_last_frame_index"))},
        {"name": "video queue span frames", "value": int(metric_value(counters, "video_queue_span_frames"))},
        {"name": "video queue span seconds", "value": round(metric_value(counters, "video_queue_span_seconds"), 3)},
        {"name": "video served lag frames", "value": int(metric_value(counters, "last_video_served_lag_frames"))},
        {"name": "video served lag seconds", "value": round(metric_value(counters, "last_video_served_lag_seconds"), 3)},
        {"name": "max video served lag frames", "value": int(metric_value(counters, "max_video_served_lag_frames"))},
        {"name": "video no-ready callbacks", "value": int(metric_value(counters, "video_no_ready_frame_callbacks"))},
        {"name": "audio callbacks", "value": int(metric_value(counters, "audio_callbacks"))},
        {"name": "audio callback FPS", "value": round(audio_callback_fps, 1)},
        {"name": "audio frames served", "value": int(metric_value(counters, "audio_frames_served"))},
        {"name": "audio pre-playback silence frames", "value": int(metric_value(counters, "audio_preplayback_silence_frames"))},
        {"name": "audio short-buffer frames", "value": int(metric_value(counters, "audio_short_buffer_frames"))},
        {"name": "audio playback underrun frames", "value": int(metric_value(counters, "audio_playback_underrun_frames"))},
        {"name": "audio out buffer seconds", "value": round(metric_value(counters, "audio_out_buffer_seconds"), 3)},
        {"name": "min audio out buffer seconds", "value": round(metric_value(counters, "min_audio_out_buffer_seconds"), 3)},
        {"name": "max audio out buffer seconds", "value": round(metric_value(counters, "max_audio_out_buffer_seconds"), 3)},
        {"name": "input audio time", "value": round(metric_value(counters, "last_input_audio_time_s"), 3)},
        {"name": "video source time", "value": round(metric_value(counters, "last_video_source_time_s"), 3)},
        {"name": "audio source time", "value": round(metric_value(counters, "last_audio_source_time_s"), 3)},
    ]
    latency_rows = [
        duration_row("Frame audio to video", durations, "frame_audio_to_video_latency"),
        duration_row("Frame audio to video first", durations, "frame_audio_to_video_first_latency"),
        duration_row("Frame audio to video midpoint", durations, "frame_audio_to_video_midpoint_latency"),
        duration_row("Frame audio to video last", durations, "frame_audio_to_video_last_latency"),
        duration_row("Frame audio to video served", durations, "frame_audio_to_video_served_latency"),
        duration_row("Frame publish to serve", durations, "frame_publish_to_serve_latency"),
        duration_row("Frame pre-ARTalk", durations, "frame_pre_model_latency"),
        duration_row("Frame pre-ARTalk midpoint", durations, "frame_pre_model_midpoint_latency"),
        duration_row("Frame pre-ARTalk wait midpoint", durations, "frame_pre_model_wait_midpoint"),
        duration_row("Frame render", durations, "frame_render_latency"),
        duration_row("Frame render to publish", durations, "frame_render_to_publish_latency"),
        duration_row("Frame render to serve", durations, "frame_render_to_serve_latency"),
        duration_row("Midpoint audio to video", durations, "audio_midpoint_to_video_latency"),
        duration_row("Audio to video publish", durations, "audio_to_video_latency"),
        duration_row("Pre-ARTalk latency", durations, "pre_model_latency"),
        duration_row("ARTalk streamer feed", durations, "artalk_streamer_feed"),
        duration_row("Savgol smoother", durations, "smoother_feed"),
        duration_row("Post-ARTalk latency", durations, "post_model_latency"),
        duration_row("Pre-render wait", durations, "pre_render_wait_latency"),
        duration_row("Segment render latency", durations, "segment_render_latency"),
        duration_row("Avatar prepare batch", durations, "avatar_prepare_batch"),
        duration_row("Avatar forward batch", durations, "avatar_forward_batch"),
        duration_row("Avatar GPU to CPU batch", durations, "avatar_gpu_to_cpu_batch"),
        duration_row("Avatar render batch", durations, "avatar_render_batch"),
        duration_row("RGB batch to ndarray", durations, "rgb_batch_to_numpy"),
    ]
    throughput_rows = [
        {"name": "render realtime ratio", "value": round(metric_value(counters, "last_render_realtime_ratio"), 3)},
        {"name": "render chunk FPS", "value": round(rendered_fps, 1)},
        {"name": "cumulative rendered FPS", "value": round(cumulative_rendered_fps, 1)},
        {"name": "video real-frame FPS", "value": round(video_real_fps, 1)},
        {"name": "video callback FPS", "value": round(video_callback_fps, 1)},
        {"name": "audio callback FPS", "value": round(audio_callback_fps, 1)},
        {"name": "render batch size", "value": int(metric_value(counters, "render_batch_size"))},
        {"name": "render batches", "value": int(metric_value(counters, "render_batches"))},
        {"name": "render batch frames", "value": int(metric_value(counters, "render_batch_frames"))},
        {"name": "last render chunk frames", "value": int(metric_value(counters, "last_render_chunk_frames"))},
        {"name": "last render chunk seconds", "value": round(metric_value(counters, "last_render_chunk_s"), 3)},
        {"name": "last render media seconds", "value": round(metric_value(counters, "last_render_chunk_media_s"), 3)},
        {"name": "output segment seconds", "value": round(metric_value(counters, "output_segment_seconds"), 3)},
        {"name": "output segment min frames", "value": int(metric_value(counters, "output_segment_min_frames"))},
        {"name": "rendered frames", "value": int(metric_value(counters, "rendered_frames"))},
    ]
    buffer_rows = [
        {"name": "audio out buffer seconds", "value": round(metric_value(counters, "audio_out_buffer_seconds"), 3)},
        {"name": "min audio out buffer seconds", "value": round(metric_value(counters, "min_audio_out_buffer_seconds"), 3)},
        {"name": "max audio out buffer seconds", "value": round(metric_value(counters, "max_audio_out_buffer_seconds"), 3)},
        {"name": "audio in queue", "value": int(metric_value(counters, "audio_in_queue_depth"))},
        {"name": "video queue", "value": int(metric_value(counters, "video_queue_depth"))},
        {"name": "video lead seconds", "value": round(metric_value(counters, "video_lead_seconds"), 3)},
        {"name": "video queue span seconds", "value": round(metric_value(counters, "video_queue_span_seconds"), 3)},
        {"name": "video served lag seconds", "value": round(metric_value(counters, "last_video_served_lag_seconds"), 3)},
        {"name": "audio playback underrun frames", "value": int(metric_value(counters, "audio_playback_underrun_frames"))},
        {"name": "audio short-buffer frames", "value": int(metric_value(counters, "audio_short_buffer_frames"))},
        {"name": "video placeholders", "value": int(metric_value(counters, "video_placeholder_frames"))},
        {"name": "video frames dropped for sync", "value": int(metric_value(counters, "video_frames_dropped_for_sync"))},
        {"name": "video frames dropped", "value": int(metric_value(counters, "video_frames_dropped"))},
        {"name": "silence pump worker-busy skips", "value": int(metric_value(counters, "silence_pump_worker_busy_skips"))},
        {"name": "silence pump recent-input skips", "value": int(metric_value(counters, "silence_pump_recent_input_skips"))},
    ]
    known_stage_keys = {
        "resample",
        "artalk_streamer_feed",
        "smoother_feed",
        "renderer_warmup",
        "warmup_avatar_prepare_frame",
        "warmup_avatar_forward_model",
        "warmup_avatar_gpu_to_cpu_copy",
        "warmup_rgb_tensor_to_numpy",
        "avatar_prepare_frame",
        "avatar_forward_model",
        "avatar_gpu_to_cpu_copy",
        "avatar_render_frame",
        "rgb_tensor_to_numpy",
        "avatar_prepare_batch",
        "avatar_forward_batch",
        "avatar_gpu_to_cpu_batch",
        "avatar_render_batch",
        "rgb_batch_to_numpy",
        "render_chunk_total",
        "frame_audio_to_video_latency",
        "frame_audio_to_video_first_latency",
        "frame_audio_to_video_midpoint_latency",
        "frame_audio_to_video_last_latency",
        "frame_audio_to_video_served_latency",
        "frame_publish_to_serve_latency",
        "frame_pre_model_latency",
        "frame_pre_model_first_latency",
        "frame_pre_model_midpoint_latency",
        "frame_pre_model_last_latency",
        "frame_pre_model_wait_first",
        "frame_pre_model_wait_midpoint",
        "frame_pre_model_wait_last",
        "frame_render_latency",
        "frame_render_midpoint_latency",
        "frame_render_to_publish_latency",
        "frame_render_to_publish_midpoint_latency",
        "frame_render_to_serve_latency",
        "audio_to_video_latency",
        "audio_midpoint_to_video_latency",
        "pre_model_latency",
        "pre_model_chunk_wait",
        "pre_model_streamer_compute",
        "post_model_latency",
        "post_model_excess_latency",
        "post_model_render_window",
        "post_model_publish_overhead",
        "post_model_smoother_feed",
        "post_model_audio_emit_prepare",
        "post_model_segment_audio_pairing",
        "post_model_publish_segment",
        "post_model_publish_video_queue",
        "post_model_publish_audio_buffer",
        "post_model_publish_metrics_update",
        "pre_render_wait_latency",
        "pre_render_wait_first_render_start",
        "pre_render_wait_segment_start_offset",
        "pre_render_wait_segment_media_offset",
        "pre_render_wait_segment_excess_over_media",
        "pre_render_wait_prior_segment_backlog",
        "segment_render_latency",
    }
    all_stage_rows = stage_rows + [
        duration_row(key, durations, key)
        for key in sorted(durations)
        if key not in known_stage_keys
    ]
    artalk_internal_rows = duration_rows_for_prefixes(durations, ("artalk_",))
    renderer_internal_rows = duration_rows_for_prefixes(
        durations,
        ("renderer_", "flame_"),
    )
    gagavatar_internal_rows = duration_rows_for_keys(
        durations,
        [
            key
            for key in sorted(durations)
            if key.startswith(("renderer_gagavatar_", "renderer_gagavatar_flame_"))
            or key.startswith("gaussian_")
        ],
    )
    recent_spike_rows = [
        {
            "age_s": round(float(spike.get("age_s", 0.0)), 1),
            "metric": display_metric_name(str(spike.get("metric", ""))),
            "elapsed_ms": round(float(spike.get("elapsed_ms", 0.0)), 1),
            "baseline_ms": round(float(spike.get("baseline_ms", 0.0)), 1),
            "delta_ms": round(float(spike.get("delta_ms", 0.0)), 1),
            "ratio": round(float(spike.get("ratio", 0.0)), 2),
            "context": spike_context_text(spike.get("context", {})),
        }
        for spike in reversed(spikes[-25:])
    ]
    spike_summary: dict[str, dict] = {}
    for spike in spikes:
        metric = str(spike.get("metric", ""))
        if not metric:
            continue
        elapsed_ms = float(spike.get("elapsed_ms", 0.0))
        ratio = float(spike.get("ratio", 0.0))
        summary = spike_summary.setdefault(
            metric,
            {
                "metric": metric,
                "count": 0,
                "max_ms": 0.0,
                "latest_ms": 0.0,
                "latest_ratio": 0.0,
            },
        )
        summary["count"] += 1
        summary["max_ms"] = max(summary["max_ms"], elapsed_ms)
        summary["latest_ms"] = elapsed_ms
        summary["latest_ratio"] = ratio
    spike_summary_rows = [
        {
            "metric": display_metric_name(row["metric"]),
            "count": row["count"],
            "max_ms": round(row["max_ms"], 1),
            "latest_ms": round(row["latest_ms"], 1),
            "latest_ratio": round(row["latest_ratio"], 2),
        }
        for row in sorted(
            spike_summary.values(),
            key=lambda item: (item["count"], item["max_ms"]),
            reverse=True,
        )
    ]

    latency_tab, throughput_tab, buffers_tab, spikes_tab, internals_tab, raw_tab = st.tabs(
        ["Latency", "Throughput", "Buffers / sync", "Spikes", "Internals", "Raw"]
    )

    with latency_tab:
        frame_served_total = duration_text(duration_stat(durations, "frame_audio_to_video_served_latency"))
        frame_first_total = duration_text(duration_stat(durations, "frame_audio_to_video_first_latency"))
        frame_midpoint_total = duration_text(duration_stat(durations, "frame_audio_to_video_midpoint_latency"))
        frame_last_total = duration_text(duration_stat(durations, "frame_audio_to_video_last_latency"))
        publish_to_serve = duration_text(duration_stat(durations, "frame_publish_to_serve_latency"))
        render_to_publish = duration_text(duration_stat(durations, "frame_render_to_publish_latency"))
        render_to_serve = duration_text(duration_stat(durations, "frame_render_to_serve_latency"))
        legacy_first_sample_total = duration_text(duration_stat(durations, "audio_to_video_latency"))
        legacy_midpoint_total = duration_text(duration_stat(durations, "audio_midpoint_to_video_latency"))
        pre_artalk = duration_text(duration_stat(durations, "frame_pre_model_midpoint_latency"))
        pre_artalk_wait = duration_text(duration_stat(durations, "frame_pre_model_wait_midpoint"))
        legacy_pre_artalk = duration_text(duration_stat(durations, "pre_model_latency"))
        post_artalk = duration_text(duration_stat(durations, "post_model_latency"))
        post_artalk_excess = duration_text(duration_stat(durations, "post_model_excess_latency"))
        pre_render_wait = duration_text(duration_stat(durations, "pre_render_wait_latency"))
        render = duration_text(duration_stat(durations, "segment_render_latency"))
        pre_streamer_compute = duration_value(durations, "pre_model_streamer_compute")
        post_render_window = duration_value(durations, "post_model_render_window")
        post_publish_overhead = duration_value(durations, "post_model_publish_overhead")
        post_smoother = duration_value(durations, "post_model_smoother_feed")
        post_audio_emit = duration_value(durations, "post_model_audio_emit_prepare")
        post_audio_pairing = duration_value(durations, "post_model_segment_audio_pairing")
        post_publish_segment = duration_value(durations, "post_model_publish_segment")
        post_publish_video = duration_value(durations, "post_model_publish_video_queue")
        post_publish_audio = duration_value(durations, "post_model_publish_audio_buffer")
        post_publish_metrics = duration_value(durations, "post_model_publish_metrics_update")
        pre_first_render = duration_value(durations, "pre_render_wait_first_render_start")
        pre_segment_offset = duration_value(durations, "pre_render_wait_segment_start_offset")
        pre_segment_media_offset = duration_value(durations, "pre_render_wait_segment_media_offset")
        pre_segment_excess = duration_value(durations, "pre_render_wait_segment_excess_over_media")
        pre_prior_backlog = duration_value(durations, "pre_render_wait_prior_segment_backlog")
        pre_segment_context = "#{} f{}+{}".format(
            int(metric_value(counters, "last_pre_render_wait_segment_index")),
            int(metric_value(counters, "last_pre_render_wait_segment_start_frame")),
            int(metric_value(counters, "last_pre_render_wait_segment_frames")),
        )
        ar_audio_to_device = duration_value(durations, "artalk_streamer_audio_to_device")
        ar_audio_encoder = duration_value(durations, "artalk_streamer_audio_encoder")
        ar_audio_resample = duration_value(
            durations,
            "artalk_streamer_audio_condition_resample",
        )
        ar_decode = duration_value(durations, "artalk_streamer_ar_decode")
        ar_vq_to_motion = duration_value(durations, "artalk_streamer_vq_to_motion")
        ar_state_update = duration_value(durations, "artalk_streamer_state_update")
        smoother_cpu = duration_value(durations, "artalk_smoother_motion_to_cpu")
        smoother_filter = duration_value(durations, "artalk_smoother_savgol")
        renderer_to_device = duration_value(durations, "renderer_motion_batch_to_device")
        mesh_flame = duration_value(durations, "renderer_mesh_flame_vertices_batch")
        mesh_forward = duration_value(durations, "renderer_mesh_pytorch3d_forward_batch")
        gag_prepare = duration_value(durations, "renderer_gagavatar_prepare_batch")
        gag_forward = duration_value(durations, "renderer_gagavatar_forward_batch")
        gag_flame = duration_value(durations, "renderer_gagavatar_flame_lbs")
        gag_raster = duration_value(durations, "renderer_gagavatar_model_gaussian_rasterize")
        gag_upsampler = duration_value(durations, "renderer_gagavatar_model_upsampler")
        render_gpu_cpu = duration_value(durations, "avatar_gpu_to_cpu_batch")
        rgb_to_numpy = duration_value(durations, "rgb_batch_to_numpy")
        st.caption("Metric relationships")
        st.code(
            "legend: [-] additive elapsed window, [*] diagnostic/non-additive sample\n"
            "timeline          frame-based elapsed latency           breakdown                               meaning\n"
            "----------------  ------------------------------------  ----------------------------------------  ----------------------------\n"
            "audio accepted    +                                    +\n"
            f"{diagram_columns(latency_label('Frame served total', frame_served_total), latency_label('Frame pre-ARTalk', pre_artalk), 'audio midpoint -> video callback')}\n"
            f"{diagram_columns(latency_label('Frame published midpoint', frame_midpoint_total), '', 'audio midpoint -> output queue')}\n"
            f"{diagram_columns(latency_label('Frame published first', frame_first_total), '', 'first frame in segment')}\n"
            f"{diagram_columns(latency_label('Frame published last', frame_last_total), '', 'last frame in segment')}\n"
            f"{diagram_columns(diagnostic_line('Segment total from first audio', legacy_first_sample_total), '', 'legacy segment-level view')}\n"
            f"{diagram_columns(diagnostic_line('Segment total from midpoint', legacy_midpoint_total), '', 'legacy segment-level view')}\n"
            f"{diagram_columns('', diagnostic_line('Segment pre-ARTalk', legacy_pre_artalk), 'legacy segment-level view')}\n"
            f"{diagram_columns('', metric_line('wait for ARTalk chunk', pre_artalk_wait), 'chunk fill/model floor')}\n"
            f"{diagram_columns('', metric_line('ARTalk compute total', pre_streamer_compute), 'model execution')}\n"
            f"  |               |                                    | {diagnostic_line('audio to device', ar_audio_to_device)}\n"
            f"  |               |                                    | {diagnostic_line('audio encoder', ar_audio_encoder)}\n"
            f"  |               |                                    | {diagnostic_line('audio cond resample', ar_audio_resample)}\n"
            f"  |               |                                    | {diagnostic_line('AR decode', ar_decode)}\n"
            f"  |               |                                    | {diagnostic_line('VQ to motion', ar_vq_to_motion)}\n"
            f"  |               |                                    | {diagnostic_line('state update', ar_state_update)}\n"
            f"  |               |                                    | {diagnostic_line('smoother to CPU', smoother_cpu)}\n"
            f"  |               |                                    | {diagnostic_line('smoother filter', smoother_filter)}\n"
            "motion produced   |                                    +\n"
            f"{diagram_columns('', latency_label('Post-ARTalk', post_artalk), 'motion produced -> published')}\n"
            f"{diagram_columns('', latency_label('Post-ARTalk excess', post_artalk_excess), 'minus expected segment offset')}\n"
            f"{diagram_columns('', metric_line('wait before render starts', pre_render_wait), 'queue/backlog before render')}\n"
            f"{diagram_columns('', metric_line('render window', post_render_window), 'render start -> publish')}\n"
            f"{diagram_columns('', metric_line('publish residual', post_publish_overhead), 'unattributed publish time')}\n"
            f"  |               |                                    | {diagnostic_line('smoother', post_smoother)}\n"
            f"  |               |                                    | {diagnostic_line('audio emit prepare', post_audio_emit)}\n"
            f"  |               |                                    | {diagnostic_line('segment audio pairing', post_audio_pairing)}\n"
            f"  |               |                                    | {diagnostic_line('publish segment', post_publish_segment)}\n"
            f"  |               |                                    | {diagnostic_line('publish video queue', post_publish_video)}\n"
            f"  |               |                                    | {diagnostic_line('publish audio buffer', post_publish_audio)}\n"
            f"  |               |                                    | {diagnostic_line('publish metrics', post_publish_metrics)}\n"
            f"{diagram_columns('', diagnostic_line('first render start delay', pre_first_render), 'chunk first render delay')}\n"
            f"{diagram_columns('', diagnostic_line('segment render start offset', pre_segment_offset), 'motion -> this segment render')}\n"
            f"{diagram_columns('', diagnostic_line('segment position in chunk', pre_segment_media_offset), 'expected media offset')}\n"
            f"{diagram_columns('', diagnostic_line('extra render-start delay', pre_segment_excess), 'start offset minus position')}\n"
            f"{diagram_columns('', diagnostic_line('render backlog from first segment', pre_prior_backlog), 'alternate non-additive view')}\n"
            f"{diagram_columns('', diagnostic_line('segment index / frames', pre_segment_context), 'chunk segment identity')}\n"
            "render/publish    |                                    +\n"
            f"  |               |                                    | {latency_label('Render', render)}\n"
            f"  |               |                                    | {diagnostic_line('motion batch to device', renderer_to_device)}\n"
            f"  |               |                                    | {diagnostic_line('mesh FLAME vertices', mesh_flame)}\n"
            f"  |               |                                    | {diagnostic_line('mesh PyTorch3D forward', mesh_forward)}\n"
            f"  |               |                                    | {diagnostic_line('GAGAvatar prepare', gag_prepare)}\n"
            f"  |               |                                    | {diagnostic_line('GAGAvatar forward', gag_forward)}\n"
            f"  |               |                                    | {diagnostic_line('GAGAvatar FLAME LBS', gag_flame)}\n"
            f"  |               |                                    | {diagnostic_line('Gaussian rasterize', gag_raster)}\n"
            f"  |               |                                    | {diagnostic_line('upsampler', gag_upsampler)}\n"
            f"  |               |                                    | {diagnostic_line('GPU to CPU', render_gpu_cpu)}\n"
            f"  |               |                                    | {diagnostic_line('RGB to ndarray', rgb_to_numpy)}\n"
            "published         |                                    +\n"
            f"{diagram_columns('', metric_line('publish to serve', publish_to_serve), 'output queue -> callback')}\n"
            f"{diagram_columns('', diagnostic_line('render to publish', render_to_publish), 'frame render end -> enqueue')}\n"
            f"{diagram_columns('', diagnostic_line('render to serve', render_to_serve), 'frame render end -> callback')}\n"
            "served            +                                    +",
            language="text",
        )
        st.caption(
            "Frame served total is the closest end-to-end per-frame latency: "
            "audio midpoint to the frame returned by the video source callback. "
            "Frame published values stop earlier, when the frame enters the output "
            "queue, and are useful for separating pipeline latency from playback "
            "queueing. "
            "Frame pre-ARTalk plus Post-ARTalk is the top-level additive breakdown. "
            "Rows marked with '-' are additive child windows for their parent. "
            "Rows marked with '*' are diagnostic samples or alternate views; "
            "for example, segment start offset and prior-render backlog describe "
            "the same delayed render start from different anchors and should not "
            "be added together. Excess over media offset is the part of a segment "
            "start delay that remains after accounting for how far that segment is "
            "inside the current ARTalk chunk, so it is the main pre-render spike "
            "signal to watch. Post-ARTalk excess subtracts that segment media "
            "offset from Post-ARTalk and is the main actionable post-ARTalk signal. "
            "Legacy segment totals use the first or midpoint audio sample in the "
            "published segment and are kept for comparison, but they can become "
            "very large when backlog accumulates."
        )
        st.caption("Latency breakdown")
        st.dataframe(
            latency_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_latency_breakdown",
        )

    with throughput_tab:
        st.caption("Throughput breakdown")
        st.dataframe(
            throughput_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_throughput_breakdown",
        )

    with buffers_tab:
        st.caption("Buffer and sync health")
        st.dataframe(
            buffer_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_buffer_health",
        )

    with spikes_tab:
        st.caption(
            "Recent duration spikes. Detection uses each metric's rolling median "
            "baseline, ignores the first few samples, and records only bounded "
            "event data instead of a full timeline."
        )
        st.dataframe(
            recent_spike_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_recent_spikes",
        )
        st.caption("Spike summary by metric")
        st.dataframe(
            spike_summary_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_spike_summary",
        )

    with internals_tab:
        st.caption("ARTalk internals")
        st.dataframe(
            artalk_internal_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_artalk_internal_timings",
        )

        st.caption("Renderer / FLAME internals")
        st.dataframe(
            renderer_internal_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_renderer_internal_timings",
        )

        st.caption("GAGAvatar / Gaussian internals")
        st.dataframe(
            gagavatar_internal_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_gagavatar_internal_timings",
        )

    with raw_tab:
        st.caption("All stage timings")
        st.dataframe(
            all_stage_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_stage_metrics",
        )

        st.caption("Raw counters")
        st.dataframe(
            output_rows,
            hide_index=True,
            width="stretch",
            key="pipeline_output_counters",
        )

        st.caption("All counters")
        st.dataframe(
            [
                {"name": key, "value": value}
                for key, value in sorted(counters.items())
            ],
            hide_index=True,
            width="stretch",
            key="pipeline_all_counters",
        )
