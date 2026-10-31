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
from typing import TYPE_CHECKING

from streamlit import config
from streamlit.runtime import Runtime
from streamlit.runtime.app_session import AppSession
from streamlit.runtime.scriptrunner import get_script_run_ctx

from .gc_probe import collect_frozen_garbage

if TYPE_CHECKING:
    from streamlit.runtime.session_manager import SessionInfo

logger = logging.getLogger(__name__)

# A closed tab drops its session at once, but a reload or a flaky link can
# bring the same session back within Streamlit's reconnect window, so leave a
# margin before tearing its pipeline down.
DISCONNECT_GRACE_SECONDS = 20.0
REAP_INTERVAL_SECONDS = 5.0

# How long before Streamlit forgets a disconnected session to close it. The
# closer only notices a disconnect on its next pass, so this must exceed the
# pass interval with room to spare.
SESSION_CLOSE_MARGIN_SECONDS = 15.0
SESSION_CLOSE_INTERVAL_SECONDS = 5.0
# Each collection over the frozen objects stalls every thread for a few
# hundred milliseconds, so sessions closed in a busy minute share one, and
# it waits for a moment when no session is playing, up to the cap.
FROZEN_COLLECT_INTERVAL_SECONDS = 60.0
FROZEN_COLLECT_MAX_DEFER_SECONDS = 600.0

# How recently a pipeline must have handed a track media for the process to
# count as playing.
PLAYING_WINDOW_SECONDS = 2.0

# A session already past its time limit when someone starts waiting gets
# this long rather than ending at once, and so does one that is within this
# long of the limit, so the visitor sees the end coming on the page.
FINAL_COUNTDOWN_SECONDS = 30.0

ENDED_TIME_LIMIT = "time_limit"
ENDED_NOT_STARTED = "not_started"


_last_media_served = float("-inf")


def note_media_served() -> None:
    """Record that a pipeline just handed one of its tracks a frame."""
    global _last_media_served
    _last_media_served = time.monotonic()


def media_playing() -> bool:
    """Whether any session's media is flowing. The slots' record of START
    and STOP lags: a script run records START seconds after the browser
    starts playing. The frames themselves stop while a connection is set
    up or the event loop stalls. Neither alone covers a playing session."""
    return time.monotonic() - _last_media_served < PLAYING_WINDOW_SECONDS


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
    arrival order, and stops what a vanished session left behind, and,
    while others wait, what one past its time limit or one admitted that
    is not playing left behind. Attached objects are stopped through
    ``stop()``.

    The time limit, `session_seconds` of playing time (0: none), only
    holds while someone is waiting: a slot nobody is waiting for serves
    nobody else, so cutting its session short would only leave the GPU
    idle."""

    def __init__(self) -> None:
        self.limit = 0
        self.start_grace = 30.0
        self.session_seconds = 0.0
        self._lock = threading.Lock()
        self._holders: dict[str, dict[str, object]] = {}
        self._queue: list[str] = []
        self._missing_since: dict[str, float] = {}
        # Playing time before the current stretch, and when the current
        # stretch began for a session playing now.
        self._played: dict[str, float] = {}
        self._playing_since: dict[str, float] = {}
        # Playing time each holder may reach, set only while someone waits.
        self._allowance: dict[str, float] = {}
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
            # Queueing can give the holders deadlines the reaper has not seen.
            self._wake.set()
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

    def estimated_wait(self, session_id: str) -> float | None:
        """Seconds until this queued session should get a slot, if every
        session runs until its time limit and starts as soon as it is
        admitted. None without a time limit, or for a session not queued."""
        position = self.queue_position(session_id)
        if not self.session_seconds or not self.limit or position is None:
            return None
        now = time.monotonic()
        with self._lock:
            frees = []
            for holder in self._holders:
                allowance = self._allowance.get(
                    holder, self._default_allowance(holder, now)
                )
                remaining = allowance - self._played_total(holder, now)
                if holder in self._playing_since:
                    frees.append(remaining)
                else:
                    start_by = self._start_by.get(holder, now + self.start_grace)
                    frees.append(start_by - now + remaining)
            frees.extend([0.0] * max(self.limit - len(self._holders), 0))
        heapq.heapify(frees)
        for _ in range(position - 1):
            heapq.heappush(frees, max(heapq.heappop(frees), 0.0) + self.session_seconds)
        return max(heapq.heappop(frees), 0.0)

    def take_over(self, session_id: str) -> None:
        """Stop every other holder and hand this session the slot.

        The escape hatch for a holder the reaper will not collect because its
        session is still connected, such as a tab left open on a machine that
        went to sleep.
        """
        active = _active_session_ids()
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
            self._update_waiting(active)
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

    def any_started(self) -> bool:
        """Whether some session holding a slot has pressed START and not
        STOP since."""
        with self._lock:
            return bool(self._started)

    def mark_stopped(self, session_id: str) -> None:
        """Record that this session pressed STOP. It keeps its slot as an
        admitted session that has not started does, so its START grace
        period runs while someone is waiting, and its playing time stops
        counting."""
        active = _active_session_ids()
        with self._lock:
            if session_id not in self._holders:
                return
            self._started.discard(session_id)
            since = self._playing_since.pop(session_id, None)
            if since is not None:
                self._played[session_id] = self._played.get(session_id, 0.0) + (
                    time.monotonic() - since
                )
            self._update_waiting(active)
        self._wake.set()

    def start_clock(self, session_id: str) -> None:
        """Count this session's playing time from now, unless it is already
        counting.

        The reaper stops the slot's objects once its time runs out while
        someone waits, so the limit holds even when no script run is left
        to enforce it, such as for a tab in the background.
        """
        self.mark_started(session_id)
        with self._lock:
            if session_id not in self._holders:
                return
            self._playing_since.setdefault(session_id, time.monotonic())
        self._wake.set()

    def deadline(self, session_id: str) -> float | None:
        """When this playing session's time runs out
        (``time.monotonic()``); None while nobody is waiting."""
        with self._lock:
            return self._deadline(session_id)

    def time_left(self, session_id: str) -> float | None:
        """Seconds of playing time this stopped session has left; None
        while nobody is waiting."""
        with self._lock:
            if session_id not in self._allowance or session_id in self._playing_since:
                return None
            return max(
                self._allowance[session_id] - self._played.get(session_id, 0.0), 0.0
            )

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
        self._played.pop(session_id, None)
        self._playing_since.pop(session_id, None)
        self._allowance.pop(session_id, None)
        self._start_by.pop(session_id, None)
        self._started.discard(session_id)

    def _played_total(self, session_id: str, now: float) -> float:
        """Caller holds the lock."""
        played = self._played.get(session_id, 0.0)
        since = self._playing_since.get(session_id)
        return played if since is None else played + now - since

    def _default_allowance(self, session_id: str, now: float) -> float:
        """Caller holds the lock. Counted from all the playing time so far,
        so a long session does not get a fresh limit when someone starts
        waiting."""
        return max(
            self.session_seconds,
            self._played_total(session_id, now) + FINAL_COUNTDOWN_SECONDS,
        )

    def _deadline(self, session_id: str) -> float | None:
        """Caller holds the lock."""
        allowance = self._allowance.get(session_id)
        since = self._playing_since.get(session_id)
        if allowance is None or since is None:
            return None
        return since + allowance - self._played.get(session_id, 0.0)

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
        self._update_waiting(active)

    def _update_waiting(self, active: set[str] | None) -> None:
        """Caller holds the lock. While someone is waiting, a holder that
        has not started, or has stopped, gets `start_grace` seconds from
        when someone began waiting for its slot or from its STOP, whichever
        is later, so a tab left open without pressing START does not stall
        the queue, and every holder gets its allowance of playing time.
        With nobody waiting, both are lifted."""
        if not any(active is None or other in active for other in self._queue):
            self._start_by.clear()
            self._allowance.clear()
            return
        now = time.monotonic()
        start_by = now + self.start_grace
        for session_id in self._holders:
            if session_id not in self._started:
                self._start_by.setdefault(session_id, start_by)
            if self.session_seconds and session_id not in self._allowance:
                self._allowance[session_id] = self._default_allowance(session_id, now)

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
                deadlines = [self._deadline(holder) for holder in self._holders]
                next_deadline = min(
                    [d for d in deadlines if d is not None]
                    + list(self._start_by.values()),
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
        # Unstarted holders go first, so the slots they free count against
        # the sessions out of time.
        self._stop(
            self._take_unstarted(),
            "it did not start before its grace period ran out",
        )
        self._stop(self._take_out_of_time(), "its time limit ran out")
        active = _active_session_ids()
        live = live_session_ids()
        with self._lock:
            self._admit(active)
            if live is not None:
                for session_id in list(self._ended):
                    if session_id not in live:
                        del self._ended[session_id]

    def _take_unstarted(self) -> list[tuple[str, dict[str, object]]]:
        now = time.monotonic()
        overdue = []
        with self._lock:
            for session_id, start_by in list(self._start_by.items()):
                if now < start_by:
                    continue
                del self._start_by[session_id]
                holder = self._holders.pop(session_id, None)
                if holder is not None:
                    self._forget_holder(session_id)
                    overdue.append((session_id, holder))
                    self._ended[session_id] = ENDED_NOT_STARTED
        return overdue

    def _take_out_of_time(self) -> list[tuple[str, dict[str, object]]]:
        """Remove and return the holders out of playing time, but no more
        than the waiting sessions need. The rest keep playing, since the
        queue then empties and their limit is lifted."""
        active = _active_session_ids()
        now = time.monotonic()
        with self._lock:
            waiting = sum(
                1 for other in self._queue if active is None or other in active
            )
            free = self.limit - len(self._holders) if self.limit else waiting
            due = [
                session_id
                for session_id in self._holders
                if (deadline := self._deadline(session_id)) is not None
                and now >= deadline
            ]
            # Over-limit sessions given the same final countdown run out
            # together, and the one that has played longest ends first.
            due.sort(
                key=lambda session_id: self._played_total(session_id, now), reverse=True
            )
            taken = []
            for session_id in due[: max(waiting - free, 0)]:
                taken.append((session_id, self._holders.pop(session_id)))
                self._forget_holder(session_id)
                self._ended[session_id] = ENDED_TIME_LIMIT
        return taken

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
            # the recorder observe it, so they go before it. The recorder
            # goes before the WebRTC worker too, whose closing connection
            # would otherwise end the record as a disconnect.
            for name in ("recorder", "webrtc", "bridge", "pump", "watcher", "pipeline"):
                obj = objects.get(name)
                if obj is None:
                    continue
                try:
                    obj.stop()
                except Exception:
                    logger.warning("[slots] stopping %s failed", name, exc_info=True)


class DisconnectedSessionCloser:
    """Shuts down each disconnected session shortly before Streamlit forgets
    it, then collects what the session left behind.

    Streamlit keeps a disconnected session for
    ``server.disconnectedSessionTTL`` seconds so that its browser can come
    back, and then drops it from ``MemorySessionStorage`` without calling
    ``AppSession.shutdown()``. Only ``close_session`` shuts a session down,
    and nothing in the server calls it for a closed tab (see
    ``WebsocketSessionManager.disconnect_session`` and ``close_session`` in
    streamlit/runtime/websocket_session_manager.py, Streamlit 1.65.0). A
    session that is never shut down keeps its script event loop and its
    file descriptors open. Worse, streamlit-webrtc's ``ShutdownPolling``
    threads wait for the shutdown and keep a strong reference to the session
    while they wait, so the session, its session state and the pipeline in
    it are never freed.

    The session's objects reference each other in cycles that
    ``freeze_loaded_objects`` may already have frozen, so freeing them takes
    a collection over the frozen objects as well.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._started = False
        # Only read and written on the Streamlit event loop. The SessionInfo
        # tells disconnects apart: Streamlit stores a new one at each, so a
        # reconnect and a new disconnect between two passes restart the
        # clock.
        self._disconnected_since: dict[str, tuple[SessionInfo, float]] = {}
        self._closed_lock = threading.Lock()
        self._closed: list[AppSession] = []

    def start(self) -> None:
        with self._lock:
            if self._started or not Runtime.exists():
                return
            self._started = True
            missing = _missing_streamlit_internals()
            if missing:
                logger.warning(
                    "[sessions] not closing disconnected sessions: this "
                    "Streamlit lacks %s, so their pipelines are kept until "
                    "the process exits",
                    ", ".join(missing),
                )
                return
            threading.Thread(
                target=self._run, name="session-closer", daemon=True
            ).start()

    def _run(self) -> None:
        runtime = Runtime.instance()
        last_collect = -FROZEN_COLLECT_INTERVAL_SECONDS
        # When the first session closed since the last collection was
        # emptied, or None when nothing awaits a collection.
        garbage_since: float | None = None
        while True:
            time.sleep(SESSION_CLOSE_INTERVAL_SECONDS)
            now = time.monotonic()
            # Sessions are closed at the end of a pass, so they are a pass old
            # here, and the ShutdownPolling threads, which check once a
            # second, have let go of them.
            with self._closed_lock:
                closed, self._closed = self._closed, []
            if closed and garbage_since is None:
                garbage_since = now
            # Popped rather than iterated, so no loop variable keeps the
            # last session alive until the next pass.
            while closed:
                _empty_closed_session(closed.pop())
            if (
                garbage_since is not None
                and now - last_collect >= FROZEN_COLLECT_INTERVAL_SECONDS
            ):
                forced = now - garbage_since >= FROZEN_COLLECT_MAX_DEFER_SECONDS
                if forced or not (media_playing() or slots.any_started()):
                    elapsed_ms = collect_frozen_garbage()
                    logger.info(
                        "[sessions] collected closed sessions in %.0f ms%s",
                        elapsed_ms,
                        " (forced: sessions kept playing)" if forced else "",
                    )
                    last_collect = now
                    garbage_since = None
            try:
                # The session manager is only safe to use on its event loop.
                runtime._get_async_objs().eventloop.call_soon_threadsafe(
                    self._close_expiring, runtime
                )
            except Exception:
                logger.warning("[sessions] closer pass failed", exc_info=True)

    def _close_expiring(self, runtime: Runtime) -> None:
        session_mgr = runtime._session_mgr
        close_after = (
            config.get_option("server.disconnectedSessionTTL")
            - SESSION_CLOSE_MARGIN_SECONDS
        )
        now = time.monotonic()
        previous = self._disconnected_since
        self._disconnected_since = {}
        for info in session_mgr.list_sessions():
            session_id = info.session.id
            if session_mgr.is_active_session(session_id):
                continue
            seen_info, since = previous.get(session_id, (None, now))
            if seen_info is not info:
                since = now
            if now - since < close_after:
                self._disconnected_since[session_id] = (info, since)
                continue
            logger.info("[sessions] closing disconnected session %s", session_id)
            runtime.close_session(session_id)
            with self._closed_lock:
                self._closed.append(info.session)


def _missing_streamlit_internals() -> list[str]:
    """The private Streamlit attributes the closer relies on that this
    Streamlit lacks. Checked against the session running the current
    script, since some are only set on instances."""
    runtime = Runtime.instance()
    missing = [
        name
        for name in ("_get_async_objs", "_session_mgr", "close_session")
        if not hasattr(runtime, name)
    ]
    session_mgr = getattr(runtime, "_session_mgr", None)
    missing += [
        f"_session_mgr.{name}"
        for name in ("list_sessions", "is_active_session", "get_session_info")
        if not hasattr(session_mgr, name)
    ]
    try:
        from streamlit.runtime.state.session_state import SessionState, WStates
    except ImportError:
        missing.append("streamlit.runtime.state.session_state.SessionState")
    else:
        if "_new_widget_state" not in getattr(SessionState, "__dataclass_fields__", {}):
            missing.append("SessionState._new_widget_state")
        elif "widget_metadata" not in getattr(WStates, "__dataclass_fields__", {}):
            missing.append("WStates.widget_metadata")
    ctx = get_script_run_ctx()
    info = (
        session_mgr.get_session_info(ctx.session_id)
        if ctx is not None and hasattr(session_mgr, "get_session_info")
        else None
    )
    if info is not None and not hasattr(info.session, "_fragment_storage"):
        missing.append("AppSession._fragment_storage")
    return missing


def _empty_closed_session(session: AppSession) -> None:
    # What a closed session leaves can outlive it: the runtime loop keeps
    # the last session it served in a local until another session sends
    # something, and aiortc's MediaRelay keeps some source tracks, whose
    # streamlit-webrtc reset callback holds the session state. Emptying the
    # state, its widget callbacks and the fragments lets the pipeline go
    # either way. Not done at close time, because the ShutdownPolling
    # callbacks still pop their keys then.
    try:
        state = session.session_state
        state.clear()
        state._new_widget_state.widget_metadata.clear()
        session._fragment_storage.clear()
    except Exception:
        logger.warning("[sessions] emptying a closed session failed", exc_info=True)


slots = PipelineSlots()
session_closer = DisconnectedSessionCloser()
