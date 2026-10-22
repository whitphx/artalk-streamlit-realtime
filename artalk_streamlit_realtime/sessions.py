"""Process-wide accounting of the browser sessions holding a GPU pipeline.

Every session builds its own pipeline as soon as its page renders, so on a
shared host the number of open tabs is the number of render loops on the
GPU, and a tab that closes without pressing STOP leaves its pipeline running
with nothing to stop it: Streamlit offers no session-end hook, and the
callbacks that would stop it only run inside a script run.
"""

from __future__ import annotations

import heapq
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

ENDED_TIME_LIMIT = "time_limit"
ENDED_NOT_STARTED = "not_started"


def current_session_id() -> str:
    ctx = get_script_run_ctx()
    return ctx.session_id if ctx is not None else "no-session"


def live_session_ids() -> set[str] | None:
    """Sessions Streamlit can still resume: connected, or disconnected but
    within its reconnect window. None outside a Streamlit server."""
    if not Runtime.exists():
        return None
    session_mgr = Runtime.instance()._session_mgr
    return {info.session.id for info in session_mgr.list_sessions()}


def _active_session_ids() -> set[str] | None:
    if not Runtime.exists():
        return None
    session_mgr = Runtime.instance()._session_mgr
    return {info.session.id for info in session_mgr.list_active_sessions()}


class PipelineSlots:
    """Admits at most `limit` sessions (0: unlimited), queues the rest in
    arrival order, and stops what a vanished session, one past its time
    limit, or one admitted that is not playing while others wait left
    behind. Attached objects are stopped through ``stop()``."""

    def __init__(self) -> None:
        self.limit = 0
        self.start_grace = 30.0
        self._lock = threading.Lock()
        self._holders: dict[str, dict[str, object]] = {}
        self._queue: list[str] = []
        self._missing_since: dict[str, float] = {}
        self._deadlines: dict[str, float] = {}
        self._time_left: dict[str, float] = {}
        self._start_by: dict[str, float] = {}
        self._started: set[str] = set()
        self._ended: dict[str, str] = {}
        self._wake = threading.Event()
        self._reaper: threading.Thread | None = None

    def acquire(self, session_id: str) -> bool:
        """Whether this session holds a slot. A session refused one keeps
        its place in the queue across calls, and the reaper hands it a slot
        once it reaches the head and one is free. A session whose end
        ``pop_ended`` has not reported yet is refused without queueing."""
        with self._lock:
            if session_id in self._holders:
                return True
            full = bool(self.limit) and len(self._holders) >= self.limit
        if full:
            # A reload is a new session asking while its predecessor is still
            # inside the grace period; a holder whose browser is gone yields
            # at once when someone is waiting.
            self._stop(self._take_gone(grace_seconds=0.0), "its browser is gone")
        active = _active_session_ids()
        with self._lock:
            # A script run that checked for an end just before the reaper
            # took the slot would otherwise queue the session, and the
            # visitor would never see why their session stopped.
            if session_id in self._ended:
                return False
            # The reaper may have admitted this session since the check above.
            if session_id not in self._holders and session_id not in self._queue:
                self._queue.append(session_id)
            self._admit(active)
            self._start_reaper()
            return session_id in self._holders

    def holds(self, session_id: str) -> bool:
        with self._lock:
            return session_id in self._holders

    def queue_position(self, session_id: str) -> int | None:
        """1 for the head of the queue, None for a session not in it.
        Entries whose browser is gone are not counted, since admission
        passes over them."""
        active = _active_session_ids()
        with self._lock:
            if session_id not in self._queue:
                return None
            ahead = self._queue[: self._queue.index(session_id)]
        return 1 + sum(1 for other in ahead if active is None or other in active)

    def estimated_wait(self, session_id: str, session_seconds: float) -> float | None:
        """Seconds until this queued session should get a slot, if every
        session runs its full `session_seconds` and starts as soon as it is
        admitted. None without a time limit, or for a session not queued."""
        position = self.queue_position(session_id)
        if not session_seconds or not self.limit or position is None:
            return None
        now = time.monotonic()
        with self._lock:
            frees = []
            for holder in self._holders:
                if holder in self._deadlines:
                    frees.append(self._deadlines[holder] - now)
                else:
                    start_by = self._start_by.get(holder, now + self.start_grace)
                    time_left = self._time_left.get(holder, session_seconds)
                    frees.append(start_by - now + time_left)
            frees.extend([0.0] * max(self.limit - len(self._holders), 0))
        heapq.heapify(frees)
        for _ in range(position - 1):
            heapq.heappush(frees, max(heapq.heappop(frees), 0.0) + session_seconds)
        return max(heapq.heappop(frees), 0.0)

    def take_over(self, session_id: str) -> None:
        """Stop every other holder and hand this session the slot.

        The escape hatch for a holder the reaper will not collect because its
        session is still connected, such as a tab left open on a machine that
        went to sleep.
        """
        with self._lock:
            evicted = [
                (other, self._holders.pop(other))
                for other in list(self._holders)
                if other != session_id
            ]
            for other, _ in evicted:
                self._forget_holder(other)
            if session_id in self._queue:
                self._queue.remove(session_id)
            self._holders.setdefault(session_id, {})
            self._start_reaper()
        self._stop(evicted, "taken over by another session")

    def attach(self, session_id: str, name: str, obj) -> bool:
        """Put an object under this session's slot, to be stopped when the
        slot is reclaimed. False means the slot was taken meanwhile, and the
        object is the caller's to dispose of; attaching must not resurrect a
        holder that was already evicted."""
        with self._lock:
            holder = self._holders.get(session_id)
            if holder is None:
                return False
            holder[name] = obj
            return True

    def release(self, session_id: str) -> None:
        with self._lock:
            if self._holders.pop(session_id, None) is None:
                return
            self._forget_holder(session_id)
        # Called from WebRTC threads as well as script runs, so the queue is
        # served from the reaper rather than from whoever freed the slot.
        self._wake.set()

    def mark_started(self, session_id: str) -> None:
        """Record that this session pressed START, which ends its START grace
        period. Counted from the press rather than from playing media, since
        connecting can take seconds of the grace period."""
        with self._lock:
            if session_id in self._holders:
                self._started.add(session_id)
                self._start_by.pop(session_id, None)

    def mark_stopped(self, session_id: str) -> None:
        """Record that this session pressed STOP. It keeps its slot as an
        admitted session that has not started does, so its START grace
        period runs while someone is waiting, and its time limit pauses."""
        active = _active_session_ids()
        with self._lock:
            if session_id not in self._holders:
                return
            self._started.discard(session_id)
            deadline = self._deadlines.pop(session_id, None)
            if deadline is not None:
                self._time_left[session_id] = max(deadline - time.monotonic(), 0.0)
            self._arm_start_grace(active)
        self._wake.set()

    def start_clock(self, session_id: str, seconds: float) -> None:
        """Give this session's slot a deadline `seconds` from now, or the
        time it had left when it stopped, unless it already has one.

        The reaper stops the slot's objects once the deadline passes, so the
        limit holds even when no script run is left to enforce it, such as
        for a tab in the background.
        """
        self.mark_started(session_id)
        with self._lock:
            if session_id not in self._holders:
                return
            if session_id not in self._deadlines:
                seconds = self._time_left.pop(session_id, seconds)
                self._deadlines[session_id] = time.monotonic() + seconds
        self._wake.set()

    def deadline(self, session_id: str) -> float | None:
        """When this session's time limit runs out (``time.monotonic()``)."""
        with self._lock:
            return self._deadlines.get(session_id)

    def time_left(self, session_id: str) -> float | None:
        """Seconds of this session's time limit left while it is stopped."""
        with self._lock:
            return self._time_left.get(session_id)

    def start_by(self, session_id: str) -> float | None:
        """When this session's slot passes to the queue unless it presses
        START; None while nobody is waiting for it."""
        with self._lock:
            return self._start_by.get(session_id)

    def pop_ended(self, session_id: str) -> str | None:
        """Why the reaper took this session's slot since the last call:
        ``ENDED_TIME_LIMIT``, ``ENDED_NOT_STARTED``, or None."""
        with self._lock:
            return self._ended.pop(session_id, None)

    def _forget_holder(self, session_id: str) -> None:
        """Caller holds the lock and has removed the holder."""
        self._missing_since.pop(session_id, None)
        self._deadlines.pop(session_id, None)
        self._time_left.pop(session_id, None)
        self._start_by.pop(session_id, None)
        self._started.discard(session_id)

    def _admit(self, active: set[str] | None) -> None:
        """Caller holds the lock. Hand free slots to the queue in order,
        passing over sessions whose browser is gone."""
        for session_id in list(self._queue):
            if self.limit and len(self._holders) >= self.limit:
                break
            if active is not None and session_id not in active:
                continue
            self._queue.remove(session_id)
            self._missing_since.pop(session_id, None)
            self._holders[session_id] = {}
            logger.info("[slots] admitting session %s", session_id)
        self._arm_start_grace(active)

    def _arm_start_grace(self, active: set[str] | None) -> None:
        """Caller holds the lock. A holder that has not started, or has
        stopped, gets `start_grace` seconds from when someone began waiting
        for its slot or from its STOP, whichever is later, so a tab left open
        without pressing START does not stall the queue, while one nobody is
        waiting for keeps its slot."""
        if not any(active is None or other in active for other in self._queue):
            self._start_by.clear()
            return
        start_by = time.monotonic() + self.start_grace
        for session_id in self._holders:
            if session_id not in self._started:
                self._start_by.setdefault(session_id, start_by)

    def _start_reaper(self) -> None:
        """Caller holds the lock."""
        if self._reaper is None:
            self._reaper = threading.Thread(
                target=self._reap_loop, name="pipeline-slot-reaper", daemon=True
            )
            self._reaper.start()

    def _reap_loop(self) -> None:
        while True:
            with self._lock:
                next_deadline = min(
                    [*self._deadlines.values(), *self._start_by.values()],
                    default=None,
                )
            timeout = REAP_INTERVAL_SECONDS
            if next_deadline is not None:
                timeout = min(timeout, max(next_deadline - time.monotonic(), 0.0))
            self._wake.wait(timeout)
            self._wake.clear()
            try:
                self._reap_once()
            except Exception:
                logger.warning("[slots] reap failed", exc_info=True)

    def _reap_once(self) -> None:
        self._stop(self._take_gone(DISCONNECT_GRACE_SECONDS), "its browser is gone")
        self._stop(
            self._take_overdue(self._deadlines, ENDED_TIME_LIMIT),
            "its time limit ran out",
        )
        self._stop(
            self._take_overdue(self._start_by, ENDED_NOT_STARTED),
            "it did not start before its grace period ran out",
        )
        active = _active_session_ids()
        live = live_session_ids()
        with self._lock:
            self._admit(active)
            if live is not None:
                for session_id in list(self._ended):
                    if session_id not in live:
                        del self._ended[session_id]

    def _take_overdue(
        self, deadlines: dict[str, float], reason: str
    ) -> list[tuple[str, dict[str, object]]]:
        now = time.monotonic()
        overdue = []
        with self._lock:
            for session_id, deadline in list(deadlines.items()):
                if now < deadline:
                    continue
                del deadlines[session_id]
                holder = self._holders.pop(session_id, None)
                if holder is not None:
                    self._forget_holder(session_id)
                    overdue.append((session_id, holder))
                    self._ended[session_id] = reason
        return overdue

    def _take_gone(self, grace_seconds: float) -> list[tuple[str, dict[str, object]]]:
        """Remove and return the holders whose session has been gone for at
        least `grace_seconds`, and drop the queue entries gone for the
        disconnect grace period."""
        active = _active_session_ids()
        if active is None:
            return []
        now = time.monotonic()
        gone = []
        with self._lock:
            for session_id in [*self._holders, *self._queue]:
                if session_id in active:
                    self._missing_since.pop(session_id, None)
                    continue
                since = self._missing_since.setdefault(session_id, now)
                if session_id in self._holders:
                    if now - since >= grace_seconds:
                        gone.append((session_id, self._holders.pop(session_id)))
                        self._forget_holder(session_id)
                elif now - since >= DISCONNECT_GRACE_SECONDS:
                    self._queue.remove(session_id)
                    self._missing_since.pop(session_id, None)
        return gone

    def _stop(self, gone: list[tuple[str, dict[str, object]]], reason: str) -> None:
        for session_id, objects in gone:
            logger.info("[slots] stopping session %s: %s", session_id, reason)
            # The bridge and the pump feed the pipeline, and the watcher and
            # the recorder observe it, so they go before it.
            for name in ("recorder", "bridge", "pump", "watcher", "pipeline"):
                obj = objects.get(name)
                if obj is None:
                    continue
                try:
                    obj.stop()
                except Exception:
                    logger.warning("[slots] stopping %s failed", name, exc_info=True)


slots = PipelineSlots()
