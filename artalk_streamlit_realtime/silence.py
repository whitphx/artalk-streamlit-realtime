"""Caller-owned idle audio filler for ARTalk streaming sessions."""

from __future__ import annotations

import threading
import time

from artalk.metrics import current_pipeline_metrics
from artalk.realtime_pipeline import AUDIO_OUT_SAMPLES_PER_FRAME, ARTalkPipeline

from .config import (
    ARTALK_SAMPLE_RATE,
    IDLE_MOTION_LEAD_SAMPLES,
    SILENCE_PUMP_CHUNK_SAMPLES,
    SILENCE_PUMP_CHUNK_SECONDS,
    SILENCE_PUMP_IDLE_SECONDS,
    SILENCE_PUMP_MAX_AUDIO_BUFFER_SAMPLES,
    SILENCE_PUMP_MAX_VIDEO_FRAMES,
)


class PipelineSilencePump:
    """Fill partial ARTalk chunks when the upstream audio source is silent.

    OpenAI Realtime and browser loopback both deliver audio only while the
    upstream source is producing sound. ARTalk emits motion only after enough
    samples have accumulated for its model chunk. If the source goes silent, the
    final partial chunk can sit below that threshold forever. This belongs in
    the caller layer because it is glue between the upstream transport and the
    ARTalk package's sample input API.

    The silence before the first response counts as such a gap, so pumping
    starts with the session rather than with the first real audio: the avatar
    then animates while the conversation is still idle.

    Pumping stops after one chunk's worth of silence per idle period: that is
    enough to flush any partial chunk containing trailing real audio, and
    pumping beyond it keeps a multi-second silence backlog in the output
    buffer that the strictly-realtime playback never drains — queueing every
    subsequent response behind it and adding directly to turn latency.

    The same padding also covers the silence before the first response, so
    the avatar idles from the moment the pipeline is ready rather than
    holding its resting frame until the conversation starts.

    With ``idle_motion`` the pump keeps going after that flush, one whole
    model chunk at a time, whenever the output has nearly drained, so the
    face keeps moving and blinking through a long silence. That idle output
    is only filler, and the next real input must not wait behind it, so the
    first real input after it drops it, including the chunk the worker is
    rendering (see ``mark_input``).
    """

    def __init__(self, pipeline: ARTalkPipeline, idle_motion: bool = False) -> None:
        self._pipeline = pipeline
        self._idle_motion = idle_motion
        self._flush_budget_samples = int(
            pipeline.metrics_snapshot()["counters"].get(
                "streamer_chunk_samples",
                4 * ARTALK_SAMPLE_RATE,
            )
        )
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._last_real_input_s = 0.0
        self._last_pump_s = 0.0
        self._pumped_since_input_samples = 0
        self._idle_output_queued = False
        self._thread: threading.Thread | None = None

        with self._pipeline.metrics_context() as metrics:
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
        with self._lock:
            # Pad from the moment the session opens, not from the first
            # response, so the opening silence is treated like any other gap.
            self._last_real_input_s = time.perf_counter()
            self._pumped_since_input_samples = 0
        self._thread = threading.Thread(
            target=self._run,
            name="ARTalkSilencePump",
            daemon=True,
        )
        self._thread.start()
        self._prime_idle_avatar()

    def _prime_idle_avatar(self) -> None:
        """Give the model a chunk to chew on before any response arrives.

        The whole chunk goes in at once rather than through the paced loop
        below, which would take a chunk's worth of realtime to accumulate. It
        also leaves the streamer exactly on a chunk boundary, so the first
        real speech opens a fresh chunk instead of being appended to a
        half-filled one and waiting for the rest of it.
        """
        with self._lock:
            if self._pumped_since_input_samples:
                return
            self._pumped_since_input_samples = self._flush_budget_samples
            self._last_pump_s = time.perf_counter()
        with self._pipeline.metrics_context():
            current_pipeline_metrics().inc("silence_pump_priming_chunks")
        self._pipeline.push_silence(self._flush_budget_samples / ARTALK_SAMPLE_RATE)

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    def mark_input(self) -> None:
        """Call before pushing each piece of real input audio."""
        with self._lock:
            self._last_real_input_s = time.perf_counter()
            self._pumped_since_input_samples = 0
            drop_idle_output = self._idle_output_queued
            self._idle_output_queued = False
        with self._pipeline.metrics_context():
            current_pipeline_metrics().inc("silence_pump_real_input_marks")
            if drop_idle_output:
                # The input pushed next still waits for the worker to finish
                # the render batch or model step it is in.
                self._pipeline.flush_output(discard_in_flight=True)
                current_pipeline_metrics().inc("silence_pump_idle_output_drops")

    def _pump_idle_chunk(self, metrics) -> None:
        """Queue silence up to the next model chunk boundary once the output
        has nearly drained."""
        snapshot = self._pipeline.output_buffer_snapshot()
        if snapshot["worker_busy"] or snapshot["audio_in_queue_depth"] > 0:
            metrics.inc("silence_pump_idle_motion_busy_skips")
            return
        with self._lock:
            continuing = self._idle_output_queued
        # mark_input drops everything queued once idle output is, so the first
        # idle chunk waits for the output to play out completely: the tail of
        # the last response may still be in it. Playback never takes less than
        # a whole frame, so a shorter remainder counts as played out.
        lead = IDLE_MOTION_LEAD_SAMPLES if continuing else AUDIO_OUT_SAMPLES_PER_FRAME
        if snapshot["audio_out_buffer_samples"] >= lead or (
            not continuing and snapshot["video_queue_depth"] > 0
        ):
            metrics.inc("silence_pump_idle_motion_lead_skips")
            return
        buffered = int(
            self._pipeline.metrics_snapshot()["counters"].get(
                "streamer_buffer_samples", 0
            )
        )
        # Whole chunks keep the streamer on a chunk boundary, so the context
        # reset at the next turn start discards no idle audio that is already
        # staged for output.
        n_samples = self._flush_budget_samples - buffered % self._flush_budget_samples
        with self._lock:
            # Checked under the lock so a concurrent mark_input either sees
            # this chunk as queued or stops it from being pushed.
            if time.perf_counter() - self._last_real_input_s < SILENCE_PUMP_IDLE_SECONDS:
                return
            self._idle_output_queued = True
            self._pipeline.push_silence(n_samples / ARTALK_SAMPLE_RATE)
        metrics.inc("silence_pump_idle_motion_chunks")
        metrics.inc("silence_pump_idle_motion_samples", n_samples)

    def _run(self) -> None:
        with self._pipeline.metrics_context():
            while not self._stop_event.wait(0.1):
                if self._pipeline.is_stopped:
                    return
                with self._lock:
                    last_real_input_s = self._last_real_input_s
                    last_pump_s = self._last_pump_s
                    pumped_since_input = self._pumped_since_input_samples

                now = time.perf_counter()
                idle_s = now - last_real_input_s if last_real_input_s else 0.0
                metrics = current_pipeline_metrics()
                metrics.set("silence_pump_input_idle_s", idle_s)
                metrics.set(
                    "silence_pump_samples_since_input",
                    pumped_since_input,
                )
                if idle_s < SILENCE_PUMP_IDLE_SECONDS:
                    metrics.inc("silence_pump_recent_input_skips")
                    continue
                if pumped_since_input >= self._flush_budget_samples:
                    if self._idle_motion:
                        self._pump_idle_chunk(metrics)
                    else:
                        metrics.inc("silence_pump_flush_complete_skips")
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
                    self._pumped_since_input_samples += SILENCE_PUMP_CHUNK_SAMPLES
                metrics.inc("silence_pump_chunks")
                metrics.inc("silence_pump_samples", SILENCE_PUMP_CHUNK_SAMPLES)
                self._pipeline.push_silence(SILENCE_PUMP_CHUNK_SECONDS)
