"""Auto-dump thread stacks when a Streamlit script run wedges.

The app's hangs so far have been unreproducible headless and the hung
process lives on a host without interactive access, so waiting for a
`kill -USR1` was not workable. A daemon thread watches script-run
duration and, past a threshold, writes every thread's Python stack both
to stderr (the launcher terminal) and to a file in the diagnostics
snapshot directory on the shared filesystem, where it can be analyzed
from any host.
"""

from __future__ import annotations

import faulthandler
import sys
import threading
import time
from pathlib import Path

HANG_THRESHOLD_S = 90.0
REDUMP_INTERVAL_S = 120.0
MAX_DUMPS_PER_INCIDENT = 4
DUMP_DIR = Path("diagnostics_snapshots")


class ScriptHangWatchdog:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._run_started_s: float | None = None
        # Armed only after one run completed: the first run legitimately
        # blocks for minutes loading models.
        self._armed = False
        self._dumps_this_incident = 0
        self._last_dump_s = 0.0
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run, name="ScriptHangWatchdog", daemon=True
            )
            self._thread.start()

    def mark_run_start(self) -> None:
        with self._lock:
            self._run_started_s = time.monotonic()
            self._dumps_this_incident = 0

    def beat(self) -> None:
        """Push the deadline forward during legitimately long blocking work
        (e.g. avatar tracking)."""
        with self._lock:
            if self._run_started_s is not None:
                self._run_started_s = time.monotonic()

    def mark_run_end(self) -> None:
        with self._lock:
            self._run_started_s = None
            self._armed = True

    def _run(self) -> None:
        while True:
            time.sleep(5.0)
            with self._lock:
                started = self._run_started_s
                armed = self._armed
                dumps = self._dumps_this_incident
                last_dump = self._last_dump_s
            if not armed or started is None:
                continue
            now = time.monotonic()
            if now - started < HANG_THRESHOLD_S:
                continue
            if dumps >= MAX_DUMPS_PER_INCIDENT:
                continue
            if dumps and now - last_dump < REDUMP_INTERVAL_S:
                continue
            self._dump(now - started)
            with self._lock:
                self._dumps_this_incident += 1
                self._last_dump_s = now

    def _dump(self, stuck_s: float) -> None:
        header = f"\n[hang watchdog] script run stuck for {stuck_s:.0f} s; dumping all thread stacks\n"
        try:
            sys.stderr.write(header)
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        except Exception:
            pass
        try:
            DUMP_DIR.mkdir(exist_ok=True)
            path = DUMP_DIR / time.strftime("hang-%Y%m%d-%H%M%S.txt")
            with open(path, "w") as f:
                f.write(header)
                faulthandler.dump_traceback(file=f, all_threads=True)
        except Exception:
            # If the shared filesystem itself is the hang, the stderr copy
            # above is the surviving evidence.
            pass


script_hang_watchdog = ScriptHangWatchdog()
