"""Streamlit diagnostics for the realtime pipeline."""

from __future__ import annotations

import streamlit as st
from artalk.realtime_pipeline import ARTalkPipeline


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
        duration_row("Audio to video publish", durations, "audio_to_video_latency"),
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
        {"name": "last audio-to-video latency seconds", "value": round(metric_value(counters, "last_audio_to_video_latency_s"), 3)},
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
    st.dataframe(
        output_rows,
        hide_index=True,
        width="stretch",
        key="pipeline_output_counters",
    )
