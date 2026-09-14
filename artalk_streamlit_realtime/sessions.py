"""Process-wide accounting of the browser sessions holding a GPU pipeline.

Every session builds its own pipeline as soon as its page renders, so on a
shared host the number of open tabs is the number of render loops on the
GPU, and a tab that closes without pressing STOP leaves its pipeline running
with nothing to stop it: Streamlit offers no session-end hook, and the
callbacks that would stop it only run inside a script run.
"""

from __future__ import annotations

import logging
import threading
import time

from streamlit.runtime import Runtime
from streamlit.runtime.scriptrunner import get_script_run_ctx

logger = logging.getLogger(__name__)

# A closed tab drops its session at once, but a reload or a flaky link can
# bring the same session back within Streamlit's reconnect window, so leave a
# margin before tearing its pipeline down.
DISCONNECT_GRACE_SECONDS = 20.0
REAP_INTERVAL_SECONDS = 5.0


def current_session_id() -> str:
    ctx = get_script_run_ctx()
    return ctx.session_id if ctx is not None else "no-session"


def _active_session_ids() -> set[str] | None:
    if not Runtime.exists():
        return None
    session_mgr = Runtime.instance()._session_mgr
    return {info.session.id for info in session_mgr.list_active_sessions()}


class PipelineSlots:
    """Admits at most `limit` sessions (0: unlimited) and stops what a vanished
    session left behind. Attached objects are stopped through ``stop()``."""

    def __init__(self) -> None:
        self.limit = 0
        self._lock = threading.Lock()
        self._holders: dict[str, dict[str, object]] = {}
        self._missing_since: dict[str, float] = {}
        self._reaper: threading.Thread | None = None

    def acquire(self, session_id: str) -> bool:
        with self._lock:
            if session_id in self._holders:
                return True
            if self.limit and len(self._holders) >= self.limit:
                return False
            self._holders[session_id] = {}
            if self._reaper is None:
                self._reaper = threading.Thread(
                    target=self._reap_loop, name="pipeline-slot-reaper", daemon=True
                )
                self._reaper.start()
            return True

    def attach(self, session_id: str, name: str, obj) -> None:
        with self._lock:
            self._holders.setdefault(session_id, {})[name] = obj

    def release(self, session_id: str) -> None:
        with self._lock:
            self._holders.pop(session_id, None)
            self._missing_since.pop(session_id, None)

    def _reap_loop(self) -> None:
        while True:
            time.sleep(REAP_INTERVAL_SECONDS)
            try:
                self._reap_once()
            except Exception:
                logger.warning("[slots] reap failed", exc_info=True)

    def _reap_once(self) -> None:
        active = _active_session_ids()
        if active is None:
            return
        now = time.monotonic()
        doomed = []
        with self._lock:
            for session_id in list(self._holders):
                if session_id in active:
                    self._missing_since.pop(session_id, None)
                    continue
                since = self._missing_since.setdefault(session_id, now)
                if now - since >= DISCONNECT_GRACE_SECONDS:
                    doomed.append((session_id, self._holders.pop(session_id)))
                    self._missing_since.pop(session_id, None)
        for session_id, objects in doomed:
            logger.info("[slots] session %s is gone; stopping its pipeline", session_id)
            # The pump feeds the pipeline and the watcher observes it, so both
            # go before it.
            for name in ("pump", "watcher", "pipeline"):
                obj = objects.get(name)
                if obj is None:
                    continue
                try:
                    obj.stop()
                except Exception:
                    logger.warning("[slots] stopping %s failed", name, exc_info=True)


slots = PipelineSlots()
