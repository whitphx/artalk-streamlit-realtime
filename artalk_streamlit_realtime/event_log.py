"""Session event timeline shared by the console and the diagnostics UI.

The metrics panel answers "what is the state now"; this answers "in what
order did things happen", which is what turn-boundary bugs are made of.
Events are emitted from the OpenAI bridge (transport milestones) and from
``PipelineEventWatcher`` (model/render/playback milestones derived from
pipeline counters), so one timeline covers the whole path from the API
socket to a served video frame.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field

from artalk.realtime_pipeline import ARTalkPipeline

logger = logging.getLogger("artalk_streamlit_realtime.events")

MAX_EVENTS = 400
# Underruns and placeholder frames arrive in bursts of hundreds; collapse
# each burst into one line per this interval instead of flooding the log.
BURST_REPORT_INTERVAL_S = 2.0
# Below this, a busy spell is just the worker dequeuing an input frame.
MIN_BUSY_REPORT_S = 0.25


@dataclass(frozen=True)
class SessionEvent:
    monotonic_s: float
    wall_s: float
    category: str
    message: str
    detail: str = ""

    def format_line(self, epoch_s: float) -> str:
        stamp = f"{self.monotonic_s - epoch_s:8.2f}s"
        detail = f"  {self.detail}" if self.detail else ""
        return f"{stamp} [{self.category}] {self.message}{detail}"


@dataclass
class SessionEventLog:
    """Bounded, thread-safe event ring shared across the session."""

    events: deque[SessionEvent] = field(
        default_factory=lambda: deque(maxlen=MAX_EVENTS)
    )
    _lock: threading.Lock = field(default_factory=threading.Lock)
    epoch_s: float = field(default_factory=time.monotonic)

    def record(self, category: str, message: str, detail: str = "") -> None:
        event = SessionEvent(
            monotonic_s=time.monotonic(),
            wall_s=time.time(),
            category=category,
            message=message,
            detail=detail,
        )
        with self._lock:
            self.events.append(event)
        logger.warning("[event] %s", event.format_line(self.epoch_s))

    def snapshot(self) -> list[SessionEvent]:
        with self._lock:
            return list(self.events)

    def clear(self) -> None:
        with self._lock:
            self.events.clear()
            self.epoch_s = time.monotonic()


class PipelineEventWatcher:
    """Turns pipeline counter transitions into timeline events.

    Sampling counters keeps the ARTalk package free of UI-facing event
    plumbing: every milestone worth logging is already reflected in a
    counter the pipeline maintains.
    """

    def __init__(
        self,
        pipeline: ARTalkPipeline,
        event_log: SessionEventLog,
        interval_s: float = 0.05,
    ) -> None:
        self._pipeline = pipeline
        self._log = event_log
        self._interval_s = interval_s
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev: dict = {}
        self._burst_reported_at = 0.0
        self._burst_baseline: dict = {}
        self._busy_since: float | None = None
        self._busy_reported = False

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="ARTalkPipelineEventWatcher",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.wait(self._interval_s):
            if self._pipeline.is_stopped:
                return
            try:
                self._sample()
            except Exception:
                logger.debug("pipeline event watcher sample failed", exc_info=True)

    def _sample(self) -> None:
        counters = self._pipeline.metrics_snapshot()["counters"]
        prev = self._prev
        state = self._state_detail(counters)

        def value(key: str, default: float = 0.0) -> float:
            return float(counters.get(key, default) or 0.0)

        def rose(key: str) -> bool:
            return value(key) > float(prev.get(key, 0) or 0.0)

        if prev:
            self._report_worker_busy(bool(value("worker_busy")), state)
            if rose("motion_chunks_produced"):
                frames = int(value("last_motion_frames"))
                self._log.record(
                    "model",
                    f"motion chunk produced ({frames} frames)",
                    state,
                )
            if rose("turns_started"):
                self._log.record(
                    "turn",
                    f"turn {int(value('turns_started'))} started (first real audio)",
                    state,
                )
            if rose("turns_served"):
                latency = value("last_turn_first_frame_latency_s")
                self._log.record(
                    "turn",
                    f"turn {int(value('turns_served'))} first frame served",
                    f"latency={latency:.2f}s  {state}",
                )
            if rose("audio_playback_starts"):
                self._log.record("playback", "output playback started", state)
            if rose("output_flushes"):
                self._log.record(
                    "playback",
                    "output flushed (barge-in)",
                    f"dropped={int(value('flushed_video_frames'))} frames  {state}",
                )
            if rose("chunk_flush_requests"):
                self._log.record("model", "chunk flush requested", state)
            if rose("video_frames_dropped"):
                self._log.record(
                    "playback",
                    "video frames evicted (queue cap)",
                    state,
                )
            self._report_bursts(counters, state)

        self._prev = dict(counters)

    def _report_worker_busy(self, busy: bool, state: str) -> None:
        """One event per sustained work period, not per queued audio item.

        The worker flips busy for every 20 ms input frame it dequeues, so raw
        transitions would bury the timeline; only spells long enough to be
        real model work (a chunk decode plus its rendering) are worth a line.
        """
        now = time.monotonic()
        if busy:
            if self._busy_since is None:
                self._busy_since = now
            elif (
                not self._busy_reported
                and now - self._busy_since >= MIN_BUSY_REPORT_S
            ):
                self._busy_reported = True
                self._log.record("model", "inference/render running", state)
            return
        if self._busy_since is not None:
            elapsed = now - self._busy_since
            if self._busy_reported:
                self._log.record(
                    "model",
                    f"inference/render finished ({elapsed:.2f}s)",
                    state,
                )
            self._busy_since = None
            self._busy_reported = False

    def _report_bursts(self, counters: dict, state: str) -> None:
        """Rate-limited summary for counters that move in large bursts."""
        now = time.monotonic()
        keys = {
            "audio_playback_underrun_frames": "playback underruns",
            "video_placeholder_frames": "video placeholders (no ready frame)",
            "video_frames_dropped_for_sync": "video frames dropped to catch up",
        }
        if not self._burst_baseline:
            self._burst_baseline = {k: counters.get(k, 0) or 0 for k in keys}
            self._burst_reported_at = now
            return
        if now - self._burst_reported_at < BURST_REPORT_INTERVAL_S:
            return
        # Between turns there is nothing to play, so placeholder frames are the
        # expected state rather than a symptom; reporting them would bury the
        # gap in noise exactly where turn-boundary bugs are read.
        idle = (
            not float(counters.get("audio_out_buffer_seconds", 0) or 0.0)
            and not int(counters.get("video_queue_depth", 0) or 0)
            and not float(counters.get("worker_busy", 0) or 0.0)
        )
        parts = []
        for key, label in keys.items():
            delta = (counters.get(key, 0) or 0) - self._burst_baseline.get(key, 0)
            if delta > 0 and not (idle and key == "video_placeholder_frames"):
                parts.append(f"{label}: +{int(delta)}")
            self._burst_baseline[key] = counters.get(key, 0) or 0
        self._burst_reported_at = now
        if parts:
            self._log.record("playback", ", ".join(parts), state)

    @staticmethod
    def _state_detail(counters: dict) -> str:
        """Compact A/V state, attached to every event so the timeline shows
        how buffers and the media clock moved between milestones."""
        audio_s = float(counters.get("audio_out_buffer_seconds", 0) or 0.0)
        video_depth = int(counters.get("video_queue_depth", 0) or 0)
        clock_index = int(counters.get("synced_audio_frame_index", 0) or 0)
        served_index = int(counters.get("last_video_frame_index_served", 0) or 0)
        enqueued_index = int(counters.get("last_video_frame_index_enqueued", 0) or 0)
        return (
            f"audio_buf={audio_s:.2f}s video_q={video_depth} "
            f"clock={clock_index} served={served_index} enqueued={enqueued_index}"
        )
