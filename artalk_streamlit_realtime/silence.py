"""Caller-owned idle audio filler for ARTalk streaming sessions."""

from __future__ import annotations

import threading
import time

from artalk.realtime_pipeline import ARTalkPipeline

from .config import (
    SILENCE_PUMP_CHUNK_SAMPLES,
    SILENCE_PUMP_CHUNK_SECONDS,
    SILENCE_PUMP_IDLE_SECONDS,
    SILENCE_PUMP_MAX_AUDIO_BUFFER_SAMPLES,
    SILENCE_PUMP_MAX_VIDEO_FRAMES,
)


class PipelineSilencePump:
    """Fill final partial ARTalk chunks when the upstream audio source idles.

    OpenAI Realtime and browser loopback both deliver audio only while the
    upstream source is producing sound. ARTalk emits motion only after enough
    samples have accumulated for its model chunk. If the source goes silent, the
    final partial chunk can sit below that threshold forever. This belongs in
    the caller layer because it is glue between the upstream transport and the
    ARTalk package's sample input API.
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
        self._pipeline.metrics.event("silence_real_input")

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
                metrics.event("silence_pump_skip", reason="pacing", idle_s=idle_s)
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
                metrics.event("silence_pump_skip", reason="worker_busy", idle_s=idle_s)
                continue
            if audio_in_depth > 0:
                metrics.inc("silence_pump_input_queue_skips")
                metrics.event(
                    "silence_pump_skip",
                    reason="input_queue",
                    idle_s=idle_s,
                    audio_in_depth=audio_in_depth,
                )
                continue
            if (
                audio_buffered >= SILENCE_PUMP_MAX_AUDIO_BUFFER_SAMPLES
                or video_depth >= SILENCE_PUMP_MAX_VIDEO_FRAMES
            ):
                metrics.inc("silence_pump_backpressure_skips")
                metrics.event(
                    "silence_pump_skip",
                    reason="backpressure",
                    idle_s=idle_s,
                    audio_buffered=audio_buffered,
                    video_depth=video_depth,
                )
                continue

            with self._lock:
                self._last_pump_s = now
            metrics.inc("silence_pump_chunks")
            metrics.inc("silence_pump_samples", SILENCE_PUMP_CHUNK_SAMPLES)
            metrics.event(
                "silence_pump",
                samples=SILENCE_PUMP_CHUNK_SAMPLES,
                idle_s=idle_s,
            )
            self._pipeline.push_silence(SILENCE_PUMP_CHUNK_SECONDS)
