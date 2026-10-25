"""Optional session logging and a feedback widget."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import streamlit as st

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
COMMENT_MAX_CHARS = 500
# st.feedback("faces") returns the index of the face, 0 (worst) to 4 (best).
RATING_SCALE = 5

# The scheduled uploads guide cited on HubRecordStore recommends
# committing no more often than every 5 minutes.
HUB_COMMIT_EVERY_MIN = 5

SubjectKind = Literal["user", "anonymous"]
EndReason = Literal["stopped", "disconnected", "server"]


@dataclass(frozen=True)
class RecordBase:
    record: str
    subject_kind: SubjectKind
    subject_id: str
    browser_session: str
    time_utc: str
    schema_version: int


@dataclass(frozen=True)
class ConsentRecord(RecordBase):
    notice_sha256: str


@dataclass(frozen=True)
class MediaSessionStart(RecordBase):
    media_session: str
    mode: str
    backend: str | None
    motion_model: str
    appearance_kind: Literal["mesh", "photoreal-builtin", "photoreal-registered"]
    style: str
    session_time_limit_s: float


@dataclass(frozen=True)
class MediaSessionEnd(RecordBase):
    media_session: str
    end_reason: EndReason
    duration_s: float
    # Counters of the pipeline in use when the session ended; switching the
    # appearance or motion model mid-session builds a new pipeline whose
    # counters start from zero.
    pipeline: dict[str, float | int | None]


@dataclass(frozen=True)
class FeedbackRecord(RecordBase):
    rating: int
    rating_scale: int
    comment: str
    # The media session rated, when session logging is on as well.
    media_session: str | None


class RecordStore(Protocol):
    """Where records go.

    Another destination, such as a remote database or bucket, implements
    ``append`` and is chosen in ``open_record_store``. ``append`` is called
    from the script thread and from WebRTC and reaper threads, so it must be
    thread-safe, and it should not block for long: a session end is written
    from the thread that stops the session.
    """

    def append(self, record: dict) -> None: ...


class JsonlRecordStore:
    def __init__(self, directory: str | os.PathLike) -> None:
        self.directory = Path(directory)
        self._lock = threading.Lock()

    def path_for(self, now: datetime) -> Path:
        return self.directory / f"records-{now:%Y-%m-%d}.jsonl"

    def append(self, record: dict) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self._lock:
            self.directory.mkdir(parents=True, exist_ok=True)
            # One write per line on an O_APPEND descriptor, so lines from
            # several processes on one host do not interleave. NFS gives no
            # such guarantee across hosts.
            fd = os.open(
                self.path_for(datetime.now(timezone.utc)),
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
                0o600,
            )
            try:
                os.write(fd, line.encode())
            finally:
                os.close(fd)


class HubRecordStore(JsonlRecordStore):
    """Appends to JSON Lines files in a local folder that a
    ``huggingface_hub.CommitScheduler`` commits to a dataset repo.

    Follows the guide's scheduled uploads recipe:
    https://huggingface.co/docs/huggingface_hub/v0.36.2/en/guides/upload#scheduled-uploads
    """

    def __init__(self, directory: str | os.PathLike, repo_id: str, token: str | None) -> None:
        super().__init__(directory)
        self.repo_id = repo_id
        self._token = token
        # A restart on a fresh disk starts its files from empty; a name of
        # its own keeps the next commit from replacing the previous
        # process's file in the repo, and keeps replicas apart.
        self._process = uuid.uuid4().hex[:12]
        self._scheduler = None
        threading.Thread(
            target=self._start_scheduler, name="session-records-hub", daemon=True
        ).start()

    def path_for(self, now: datetime) -> Path:
        return self.directory / f"records-{now:%Y-%m-%d}-{self._process}.jsonl"

    def append(self, record: dict) -> None:
        scheduler = self._scheduler
        # The scheduler caps each file at the size it saw while scanning, so
        # without its lock a commit could end on half a line.
        with scheduler.lock if scheduler else contextlib.nullcontext():
            super().append(record)

    def _start_scheduler(self) -> None:
        from huggingface_hub import CommitScheduler

        # The constructor reaches the Hub to ensure the repo exists; until it
        # succeeds, records collect in the folder and the first commit
        # uploads them.
        while True:
            try:
                self._scheduler = CommitScheduler(
                    repo_id=self.repo_id,
                    repo_type="dataset",
                    folder_path=self.directory,
                    every=HUB_COMMIT_EVERY_MIN,
                    private=True,
                    token=self._token,
                    # Local-store files in the same folder, named per day
                    # only, would replace other servers' files of that name.
                    allow_patterns="records-????-??-??-*.jsonl",
                )
                return
            except Exception:
                logger.warning(
                    "[session-records] cannot reach dataset repo %s; retrying in "
                    "%d minutes",
                    self.repo_id,
                    HUB_COMMIT_EVERY_MIN,
                    exc_info=True,
                )
                time.sleep(HUB_COMMIT_EVERY_MIN * 60)


@functools.cache
def _hub_record_store(directory: str, repo_id: str) -> HubRecordStore:
    token = os.environ.get("ARTALK_SESSION_DATA_TOKEN") or None
    return HubRecordStore(directory, repo_id, token)


def open_record_store(directory: str, repo_id: str | None = None) -> RecordStore:
    if repo_id:
        # One scheduler per process: each browser session opens a store.
        return _hub_record_store(directory, repo_id)
    return JsonlRecordStore(directory)


@dataclass(frozen=True)
class ConsentNotice:
    text: str
    sha256: str


def load_consent_notice(path: str | None, text: str | None) -> ConsentNotice | None:
    if text is None:
        if not path:
            _warn_once(
                "session logging / feedback is enabled but neither "
                "--consent-notice nor --consent-notice-text is set; nothing "
                "will be collected"
            )
            return None
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            _warn_once(
                f"cannot read --consent-notice {path!r} ({exc}); nothing will be collected"
            )
            return None
        if not text.strip():
            _warn_once(f"--consent-notice {path!r} is empty; nothing will be collected")
            return None
    return ConsentNotice(text, hashlib.sha256(text.encode("utf-8")).hexdigest())


@functools.cache
def _warn_once(message: str) -> None:
    logger.warning("[session-records] %s", message)


def appearance_kind(appearance: str, registered_ids: list[str]) -> str:
    """The kind of appearance only: a registered avatar's id is a name the
    visitor chose, often their photo's file name."""
    if not appearance.startswith("gagavatar:"):
        return "mesh"
    if appearance.split(":", 1)[1] in registered_ids:
        return "photoreal-registered"
    return "photoreal-builtin"


_PIPELINE_DURATIONS = (
    "frame_audio_to_video_served_latency",
    "segment_render_latency",
)
_PIPELINE_COUNTERS = (
    "first_motion_latency_s",
    "last_render_realtime_ratio",
    "rendered_frames",
    "video_frames_served",
    "video_placeholder_frames",
    "audio_playback_underrun_frames",
    "audio_rebuffer_events",
)


def pipeline_summary(snapshot: dict) -> dict[str, float | int | None]:
    counters = snapshot.get("counters", {})
    durations = snapshot.get("durations", {})
    summary: dict[str, float | int | None] = {
        key: counters.get(key) for key in _PIPELINE_COUNTERS
    }
    for key in _PIPELINE_DURATIONS:
        stat = durations.get(key, {})
        summary[f"{key}_count"] = int(stat.get("count", 0))
        summary[f"{key}_avg_ms"] = stat.get("avg_ms")
        summary[f"{key}_max_ms"] = stat.get("max_ms")
    return summary


class _OpenMediaSession(NamedTuple):
    media_session: str
    started: float
    recorded: bool


class SessionRecorder:
    def __init__(
        self,
        store: RecordStore,
        notice: ConsentNotice,
        *,
        login_id: str | None,
        log_sessions: bool,
        feedback: bool,
    ) -> None:
        self.store = store
        self.notice = notice
        self.log_sessions = log_sessions
        self.feedback = feedback
        self.browser_session = uuid.uuid4().hex
        self.subject_kind: SubjectKind = "user" if login_id else "anonymous"
        self.subject_id = login_id or self.browser_session
        self.consent: Literal["pending", "accepted", "declined"] = "pending"
        self.media_sessions_started = 0
        self.last_media_session: str | None = None
        self.pipeline = None
        self._lock = threading.Lock()
        self._open: _OpenMediaSession | None = None

    @property
    def collecting(self) -> bool:
        return self.consent == "accepted"

    def _base(self, record: str) -> dict:
        return {
            "record": record,
            "subject_kind": self.subject_kind,
            "subject_id": self.subject_id,
            "browser_session": self.browser_session,
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "schema_version": SCHEMA_VERSION,
        }

    def _write(self, record_type: type[RecordBase], record: str, **fields) -> None:
        # Building the dataclass rejects a missing or unknown field.
        payload = dataclasses.asdict(record_type(**self._base(record), **fields))
        try:
            self.store.append(payload)
        except Exception:
            logger.warning("[session-records] could not store a record", exc_info=True)

    def accept(self) -> None:
        self.consent = "accepted"
        self._write(ConsentRecord, "consent", notice_sha256=self.notice.sha256)

    def decline(self) -> None:
        self.consent = "declined"

    def start_media(
        self,
        *,
        mode: str,
        backend: str | None,
        motion_model: str,
        appearance_kind: str,
        style: str,
        session_time_limit_s: float,
    ) -> None:
        with self._lock:
            if self._open is not None:
                return
            media_session = uuid.uuid4().hex
            # A session started before the visitor accepted stays unrecorded
            # to its end, so no end record lacks its start.
            recorded = self.collecting and self.log_sessions
            self._open = _OpenMediaSession(media_session, time.monotonic(), recorded)
            self.media_sessions_started += 1
            self.last_media_session = media_session if recorded else None
        if recorded:
            self._write(
                MediaSessionStart,
                "media_session_start",
                media_session=media_session,
                mode=mode,
                backend=backend,
                motion_model=motion_model,
                appearance_kind=appearance_kind,
                style=style,
                session_time_limit_s=session_time_limit_s,
            )

    def end_media(self, reason: EndReason) -> None:
        # Called whenever the session's pipeline is stopped, and this
        # reference would otherwise keep a stopped pipeline's GPU memory
        # alive until the session's next script run.
        pipeline, self.pipeline = self.pipeline, None
        with self._lock:
            if self._open is None:
                return
            media_session, started, recorded = self._open
            self._open = None
        if not recorded:
            return
        summary = {}
        if pipeline is not None:
            try:
                summary = pipeline_summary(pipeline.metrics_snapshot())
            except Exception:
                logger.warning("[session-records] no pipeline metrics", exc_info=True)
        self._write(
            MediaSessionEnd,
            "media_session_end",
            media_session=media_session,
            end_reason=reason,
            duration_s=round(time.monotonic() - started, 3),
            pipeline=summary,
        )

    def stop(self) -> None:
        """Called by the session slot when the server stops the session."""
        self.end_media("server")

    def submit_feedback(self, rating: int, comment: str) -> None:
        if not (self.collecting and self.feedback):
            return
        if not 0 <= rating < RATING_SCALE:
            raise ValueError(f"rating must be in [0, {RATING_SCALE}), got {rating}")
        self._write(
            FeedbackRecord,
            "feedback",
            rating=rating,
            rating_scale=RATING_SCALE,
            comment=comment.strip()[:COMMENT_MAX_CHARS],
            media_session=self.last_media_session,
        )


def render_consent_gate(recorder: SessionRecorder) -> None:
    if recorder.consent != "pending":
        return
    with st.container(border=True):
        st.markdown(recorder.notice.text)
        accept_col, decline_col = st.columns(2)
        accept_col.button(
            "Accept", key="consent_accept", type="primary", on_click=recorder.accept
        )
        decline_col.button("No thanks", key="consent_decline", on_click=recorder.decline)


def render_feedback_form(recorder: SessionRecorder) -> None:
    if not (recorder.collecting and recorder.feedback):
        return
    if recorder.media_sessions_started == 0:
        return
    with st.form("feedback_form", clear_on_submit=True, border=True):
        st.caption("How was it?")
        rating = st.feedback("faces", key="feedback_rating")
        comment = st.text_input(
            "Comment (optional)", max_chars=COMMENT_MAX_CHARS, key="feedback_comment"
        )
        if st.form_submit_button("Send feedback"):
            if rating is None:
                st.warning("Pick a face first.")
            else:
                recorder.submit_feedback(rating, comment)
                st.success("Thanks, feedback sent.")
