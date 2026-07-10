"""Process-wide GC pause instrumentation.

CPython's cyclic collector runs on whichever thread happens to trigger a
collection and holds the GIL for the entire pass, so a long gen-2 pass
freezes every pipeline thread at once. This probe times each collection
via ``gc.callbacks`` so the diagnostics UI can correlate pipeline stalls
with collector pauses.
"""

from __future__ import annotations

import gc
import threading
import time
from collections import deque

PAUSE_EVENT_LIMIT = 100
PAUSE_EVENT_MIN_MS = 10.0


class GcPauseProbe:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        # Collections never overlap (the collector holds the GIL from the
        # "start" callback through "stop"), so this needs no locking.
        self._pass_start_s: float | None = None
        self._stats = {
            generation: {"count": 0, "total_ms": 0.0, "max_ms": 0.0, "last_ms": 0.0}
            for generation in range(3)
        }
        self._recent: deque[dict] = deque(maxlen=PAUSE_EVENT_LIMIT)
        self._installed = False

    def install(self) -> None:
        with self._lock:
            if self._installed:
                return
            self._installed = True
        gc.callbacks.append(self._on_gc_event)

    def _on_gc_event(self, phase: str, info: dict) -> None:
        if phase == "start":
            self._pass_start_s = time.perf_counter()
            return
        start_s = self._pass_start_s
        if start_s is None:
            return
        self._pass_start_s = None
        now = time.perf_counter()
        elapsed_ms = (now - start_s) * 1000.0
        generation = int(info.get("generation", 0))
        with self._lock:
            stat = self._stats[generation]
            stat["count"] += 1
            stat["total_ms"] += elapsed_ms
            stat["max_ms"] = max(stat["max_ms"], elapsed_ms)
            stat["last_ms"] = elapsed_ms
            if elapsed_ms >= PAUSE_EVENT_MIN_MS:
                self._recent.append(
                    {
                        "timestamp_s": now,
                        "generation": generation,
                        "elapsed_ms": elapsed_ms,
                        "collected": int(info.get("collected", 0)),
                        "uncollectable": int(info.get("uncollectable", 0)),
                    }
                )

    def snapshot(self) -> dict:
        now = time.perf_counter()
        with self._lock:
            generations = {gen: dict(stat) for gen, stat in self._stats.items()}
            recent = [dict(event) for event in self._recent]
        for event in recent:
            event["age_s"] = now - event.pop("timestamp_s")
        return {
            "generations": generations,
            "recent": recent,
            "enabled": gc.isenabled(),
            "thresholds": gc.get_threshold(),
            "counts": gc.get_count(),
            "frozen": gc.get_freeze_count(),
        }


gc_pause_probe = GcPauseProbe()
