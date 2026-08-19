"""Catch stalls of the event loop that drives the WebRTC tracks.

When the loop stops running, both media tracks stop being pumped and the
browser receives nothing at all — audio conceals, video freezes, and the
two drift apart. The stall is invisible from the pipeline's own counters,
which keep looking healthy because the worker thread is unaffected.

Detection deliberately runs on a plain thread rather than on the loop: a
coroutine can only notice a stall once the loop is running again, by which
time the stack that caused it is gone. Here a heartbeat coroutine stamps a
timestamp, a watcher thread notices the timestamp going stale, and the
thread dumps every thread's stack *while the loop is still stuck* — so the
dump names the blocking frame. A dump showing no thread doing anything
points outward instead, at the process being descheduled.
"""

from __future__ import annotations

import asyncio
import faulthandler
import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

HEARTBEAT_S = 0.1


class EventLoopStallWatchdog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._last_beat: float | None = None
        self._installed = False
        self._threshold_s = 0.3
        self._dump_dir = Path("diagnostics_snapshots")

    def install(
        self,
        loop: asyncio.AbstractEventLoop,
        *,
        threshold_s: float = 0.3,
        dump_dir: Path | str = "diagnostics_snapshots",
    ) -> None:
        with self._lock:
            if self._installed:
                return
            self._installed = True
            self._threshold_s = float(threshold_s)
            self._dump_dir = Path(dump_dir)
            self._last_beat = time.monotonic()
        asyncio.run_coroutine_threadsafe(self._heartbeat(), loop)
        threading.Thread(
            target=self._watch, name="EventLoopStallWatchdog", daemon=True
        ).start()
        logger.warning(
            "[loop-watchdog] armed: dumping thread stacks when the event loop "
            "stalls over %.0f ms",
            self._threshold_s * 1000,
        )

    async def _heartbeat(self) -> None:
        while True:
            with self._lock:
                self._last_beat = time.monotonic()
            await asyncio.sleep(HEARTBEAT_S)

    def _watch(self) -> None:
        stall_started_at: float | None = None
        while True:
            time.sleep(HEARTBEAT_S / 2)
            with self._lock:
                last_beat = self._last_beat
                threshold_s = self._threshold_s
            if last_beat is None:
                continue
            lag_s = time.monotonic() - last_beat - HEARTBEAT_S
            # Edge-triggered: one dump and one log pair per stall, however
            # many polls it spans.
            if lag_s >= threshold_s:
                if stall_started_at is None:
                    stall_started_at = last_beat
                    self._dump(lag_s)
            elif stall_started_at is not None:
                logger.warning(
                    "[loop-watchdog] event loop recovered after ~%.0f ms",
                    (time.monotonic() - stall_started_at) * 1000,
                )
                stall_started_at = None

    def _dump(self, lag_s: float) -> None:
        path = self._dump_dir / f"loop-stall-{time.strftime('%Y%m%d-%H%M%S')}.txt"
        try:
            self._dump_dir.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as fh:
                fh.write(
                    f"event loop stalled {lag_s * 1000:.0f} ms\n"
                    f"captured {time.strftime('%Y-%m-%dT%H:%M:%S%z')} "
                    f"(while the loop is still stuck)\n\n"
                )
                faulthandler.dump_traceback(file=fh, all_threads=True)
        except OSError:
            logger.warning(
                "[loop-watchdog] event loop stalled %.0f ms; writing the dump "
                "to %s failed", lag_s * 1000, path, exc_info=True,
            )
            return
        logger.warning(
            "[loop-watchdog] event loop stalled %.0f ms — thread stacks in %s",
            lag_s * 1000,
            path,
        )


loop_stall_watchdog = EventLoopStallWatchdog()
