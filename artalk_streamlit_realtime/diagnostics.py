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


def duration_value(durations: dict, key: str) -> str:
    return duration_text(duration_stat(durations, key))


def render_pipeline_diagnostics(pipeline: ARTalkPipeline) -> None:
    snapshot = pipeline.metrics_snapshot()
    counters = snapshot["counters"]
    durations = snapshot["durations"]

    st.subheader("Pipeline diagnostics")
    total_latency = duration_stat(durations, "audio_midpoint_to_video_latency")
    post_model_latency = duration_stat(durations, "post_model_latency")
    post_model_excess_latency = duration_stat(durations, "post_model_excess_latency")
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

    st.caption("Overview")
    overview_cols = st.columns(4)
    overview_cols[0].metric("Latency", duration_text(total_latency))
    overview_cols[1].metric(
        "Render realtime ratio",
        f"{metric_value(counters, 'last_render_realtime_ratio'):.2f}x",
    )
    overview_cols[2].metric("Video real FPS", f"{video_real_fps:.1f}")
    overview_cols[3].metric(
        "Audio out buffer",
        seconds_text(metric_value(counters, "audio_out_buffer_samples") / 16000.0),
    )

    detail_cols = st.columns(6)
    detail_cols[0].metric("Post-model", duration_text(post_model_latency))
    detail_cols[1].metric("Post-model excess", duration_text(post_model_excess_latency))
    detail_cols[2].metric("Pre-render excess", duration_text(pre_render_excess_latency))
    detail_cols[3].metric("Render", duration_text(render_latency))
    detail_cols[4].metric(
        "Segment",
        "#{} f{}+{}".format(
            int(metric_value(counters, "last_pre_render_wait_segment_index")),
            int(metric_value(counters, "last_pre_render_wait_segment_start_frame")),
            int(metric_value(counters, "last_pre_render_wait_segment_frames")),
        ),
    )
    detail_cols[5].metric(
        "Audio underruns",
        int(metric_value(counters, "audio_playback_underrun_frames")),
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
        duration_row("Audio to video publish", durations, "audio_to_video_latency"),
        duration_row("Midpoint audio to video", durations, "audio_midpoint_to_video_latency"),
        duration_row("Pre-model latency", durations, "pre_model_latency"),
        duration_row("Pre-model chunk wait", durations, "pre_model_chunk_wait"),
        duration_row("Pre-model streamer compute", durations, "pre_model_streamer_compute"),
        duration_row("Post-model latency", durations, "post_model_latency"),
        duration_row("Post-model excess latency", durations, "post_model_excess_latency"),
        duration_row("Post-model render window", durations, "post_model_render_window"),
        duration_row("Post-model publish overhead", durations, "post_model_publish_overhead"),
        duration_row("Post-model smoother", durations, "post_model_smoother_feed"),
        duration_row("Post-model audio emit prepare", durations, "post_model_audio_emit_prepare"),
        duration_row("Post-model segment audio pairing", durations, "post_model_segment_audio_pairing"),
        duration_row("Post-model publish segment", durations, "post_model_publish_segment"),
        duration_row("Post-model publish video queue", durations, "post_model_publish_video_queue"),
        duration_row("Post-model publish audio buffer", durations, "post_model_publish_audio_buffer"),
        duration_row("Post-model publish metrics", durations, "post_model_publish_metrics_update"),
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
        {"name": "last audio-to-video latency seconds", "value": round(metric_value(counters, "last_audio_to_video_latency_s"), 3)},
        {"name": "last midpoint audio-to-video latency seconds", "value": round(metric_value(counters, "last_midpoint_audio_to_video_latency_s"), 3)},
        {"name": "last pre-model latency seconds", "value": round(metric_value(counters, "last_pre_model_latency_s"), 3)},
        {"name": "last post-model latency seconds", "value": round(metric_value(counters, "last_post_model_latency_s"), 3)},
        {"name": "last post-model excess latency seconds", "value": round(metric_value(counters, "last_post_model_excess_latency_s"), 3)},
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
        duration_row("Midpoint audio to video", durations, "audio_midpoint_to_video_latency"),
        duration_row("Audio to video publish", durations, "audio_to_video_latency"),
        duration_row("Pre-model latency", durations, "pre_model_latency"),
        duration_row("ARTalk streamer feed", durations, "artalk_streamer_feed"),
        duration_row("Savgol smoother", durations, "smoother_feed"),
        duration_row("Post-model latency", durations, "post_model_latency"),
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

    latency_tab, throughput_tab, buffers_tab, internals_tab, raw_tab = st.tabs(
        ["Latency", "Throughput", "Buffers / sync", "Internals", "Raw"]
    )

    with latency_tab:
        first_sample_total = duration_text(duration_stat(durations, "audio_to_video_latency"))
        midpoint_total = duration_text(duration_stat(durations, "audio_midpoint_to_video_latency"))
        pre_model = duration_text(duration_stat(durations, "pre_model_latency"))
        post_model = duration_text(duration_stat(durations, "post_model_latency"))
        post_model_excess = duration_text(duration_stat(durations, "post_model_excess_latency"))
        pre_render_wait = duration_text(duration_stat(durations, "pre_render_wait_latency"))
        render = duration_text(duration_stat(durations, "segment_render_latency"))
        pre_chunk_wait = duration_value(durations, "pre_model_chunk_wait")
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
            "timeline          elapsed latency window                breakdown\n"
            "----------------  ------------------------------------  ----------------------------------------\n"
            "audio accepted    +                                    +\n"
            f"  |               | {latency_label('Midpoint total', midpoint_total)} | {latency_label('Pre-model', pre_model)}\n"
            f"  |               | {latency_label('First-sample total', first_sample_total)} |\n"
            f"  |               |                                    | {metric_line('chunk fill / model floor wait', pre_chunk_wait)}\n"
            f"  |               |                                    | {metric_line('streamer compute total', pre_streamer_compute)}\n"
            f"  |               |                                    | {diagnostic_line('audio to device', ar_audio_to_device)}\n"
            f"  |               |                                    | {diagnostic_line('audio encoder', ar_audio_encoder)}\n"
            f"  |               |                                    | {diagnostic_line('audio cond resample', ar_audio_resample)}\n"
            f"  |               |                                    | {diagnostic_line('AR decode', ar_decode)}\n"
            f"  |               |                                    | {diagnostic_line('VQ to motion', ar_vq_to_motion)}\n"
            f"  |               |                                    | {diagnostic_line('state update', ar_state_update)}\n"
            f"  |               |                                    | {diagnostic_line('smoother to CPU', smoother_cpu)}\n"
            f"  |               |                                    | {diagnostic_line('smoother filter', smoother_filter)}\n"
            "motion produced   |                                    +\n"
            f"  |               |                                    | {latency_label('Post-model', post_model)}\n"
            f"  |               |                                    | {latency_label('Post-model excess', post_model_excess)}\n"
            f"  |               |                                    | {metric_line('pre-render wait', pre_render_wait)}\n"
            f"  |               |                                    | {metric_line('render window', post_render_window)}\n"
            f"  |               |                                    | {metric_line('publish overhead residual', post_publish_overhead)}\n"
            f"  |               |                                    | {diagnostic_line('smoother', post_smoother)}\n"
            f"  |               |                                    | {diagnostic_line('audio emit prepare', post_audio_emit)}\n"
            f"  |               |                                    | {diagnostic_line('segment audio pairing', post_audio_pairing)}\n"
            f"  |               |                                    | {diagnostic_line('publish segment', post_publish_segment)}\n"
            f"  |               |                                    | {diagnostic_line('publish video queue', post_publish_video)}\n"
            f"  |               |                                    | {diagnostic_line('publish audio buffer', post_publish_audio)}\n"
            f"  |               |                                    | {diagnostic_line('publish metrics', post_publish_metrics)}\n"
            f"  |               |                                    | {diagnostic_line('first render start delay', pre_first_render)}\n"
            f"  |               |                                    | {diagnostic_line('segment start offset', pre_segment_offset)}\n"
            f"  |               |                                    | {diagnostic_line('segment media offset', pre_segment_media_offset)}\n"
            f"  |               |                                    | {diagnostic_line('excess over media offset', pre_segment_excess)}\n"
            f"  |               |                                    | {diagnostic_line('prior-render backlog view', pre_prior_backlog)}\n"
            f"  |               |                                    | {diagnostic_line('segment index / frames', pre_segment_context)}\n"
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
            "published         +                                    +",
            language="text",
        )
        st.caption(
            "Midpoint audio to video is the representative total latency. "
            "It is an elapsed window from audio acceptance to publication, so it "
            "includes queueing and chunk-fill time, not just compute time. "
            "Pre-model plus Post-model is the top-level additive breakdown. "
            "Rows marked with '-' are additive child windows for their parent. "
            "Rows marked with '*' are diagnostic samples or alternate views; "
            "for example, segment start offset and prior-render backlog describe "
            "the same delayed render start from different anchors and should not "
            "be added together. Excess over media offset is the part of a segment "
            "start delay that remains after accounting for how far that segment is "
            "inside the current ARTalk chunk, so it is the main pre-render spike "
            "signal to watch. Post-model excess subtracts that segment media "
            "offset from Post-model and is the main actionable post-model signal. "
            "Audio to video publish is another total latency variant using the first "
            "audio sample in the segment, so do not add it to the others."
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
