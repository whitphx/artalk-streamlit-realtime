"""Optional session logging and a feedback widget."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import hashlib
import json
import logging
import os
import shutil
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, NamedTuple, Protocol

import streamlit as st

from .voice_records import (
    VOICE_DIR,
    VoiceSession,
    delete_voice_recordings,
    has_voice_recordings,
    start_voice_session,
    subject_key,
    voice_codec,
)

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
class VoiceConsentRecord(RecordBase):
    notice_sha256: str
    granted: bool


@dataclass(frozen=True)
class MediaSettings:
    mode: str
    backend: str | None
    motion_model: str
    appearance_kind: Literal["mesh", "photoreal-builtin", "photoreal-registered"]
    style: str
    session_time_limit_s: float


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

    def voice_dir(self, subject_key: str) -> Path: ...

    def delete_voice(self, subject_key: str) -> None:
        """Deletes ``voice_dir(subject_key)`` everywhere this store keeps
        it. Blocks; called from a thread of its own."""


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

    def voice_dir(self, subject_key: str) -> Path:
        return self.directory / VOICE_DIR / subject_key

    def delete_voice(self, subject_key: str) -> None:
        with contextlib.suppress(FileNotFoundError):
            shutil.rmtree(self.voice_dir(subject_key))


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

    def delete_voice(self, subject_key: str) -> None:
        scheduler = self._scheduler
        # Under the lock, a scan either sees all of the files or none.
        with scheduler.lock if scheduler else contextlib.nullcontext():
            super().delete_voice(subject_key)
        if scheduler is None:
            from huggingface_hub import HfApi

            self._delete_voice_on_hub(HfApi(token=self._token), subject_key)
            return
        # The scheduler's commits run one at a time on its HfApi's executor.
        # Queued there, the deletion lands after any commit already under way,
        # which may still be uploading these files.
        scheduler.api.run_as_future(
            self._delete_voice_on_hub, scheduler.api, subject_key
        ).result()

    def _delete_voice_on_hub(self, api, subject_key: str) -> None:
        from huggingface_hub.errors import EntryNotFoundError

        path = f"{VOICE_DIR}/{subject_key}"
        tree = api.list_repo_tree(self.repo_id, path_in_repo=path, repo_type="dataset")
        try:
            next(iter(tree), None)
        except EntryNotFoundError:
            pass
        else:
            api.delete_folder(
                path,
                self.repo_id,
                repo_type="dataset",
                commit_message="Delete voice recordings at a visitor's request",
            )
        # Even with nothing left to delete, a previous deletion may have
        # stopped before squashing.
        # https://huggingface.co/docs/huggingface_hub/v0.36.2/en/package_reference/hf_api#huggingface_hub.HfApi.super_squash_history
        api.super_squash_history(
            self.repo_id,
            repo_type="dataset",
            commit_message="Squash history after deleting voice recordings",
        )

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
                    allow_patterns=[
                        # Local-store files in the same folder, named per
                        # day only, would replace other servers' files of
                        # that name.
                        "records-????-??-??-*.jsonl",
                        *voice_upload_patterns(),
                    ],
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


def voice_upload_patterns() -> list[str]:
    extension = voice_codec().extension
    return [
        f"{VOICE_DIR}/*/*.mic.{extension}",
        f"{VOICE_DIR}/*/*.reply.{extension}",
        f"{VOICE_DIR}/*/*.json",
    ]


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


def load_consent_notice(
    path: str | None, text: str | None, *, option: str = "--consent-notice"
) -> ConsentNotice | None:
    if text is None:
        if not path:
            _warn_once(
                f"collection is enabled but neither {option} nor "
                f"{option}-text is set; nothing will be collected"
            )
            return None
        try:
            text = Path(path).read_text(encoding="utf-8")
        except OSError as exc:
            _warn_once(f"cannot read {option} {path!r} ({exc}); nothing will be collected")
            return None
        if not text.strip():
            _warn_once(f"{option} {path!r} is empty; nothing will be collected")
            return None
    return ConsentNotice(text, hashlib.sha256(text.encode("utf-8")).hexdigest())


def load_voice_consent_notice(path: str | None, text: str | None) -> ConsentNotice | None:
    if not path and text is None:
        return None
    return load_consent_notice(path, text, option="--voice-consent-notice")


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
    settings: MediaSettings
    started: float
    recorded: bool
    start_written: bool
    voice: VoiceSession | None


class VoiceDeletion(NamedTuple):
    state: Literal["running", "done", "failed"]
    message: str


class SessionRecorder:
    def __init__(
        self,
        store: RecordStore,
        notice: ConsentNotice,
        *,
        login_id: str | None,
        log_sessions: bool,
        feedback: bool,
        voice_notice: ConsentNotice | None = None,
    ) -> None:
        self.store = store
        self.notice = notice
        self.log_sessions = log_sessions
        self.feedback = feedback
        self.voice_notice = voice_notice
        self.browser_session = uuid.uuid4().hex
        self.subject_kind: SubjectKind = "user" if login_id else "anonymous"
        self.subject_id = login_id or self.browser_session
        self.voice_key = subject_key(self.subject_id)
        self.consent: Literal["pending", "accepted", "declined"] = "pending"
        self.voice_consent = False
        self.voice_deletion: VoiceDeletion | None = None
        self.media_sessions_ended = 0
        # The value of media_sessions_ended when feedback was last sent.
        self.feedback_sent_for = 0
        self.last_media_session: str | None = None
        self.pipeline = None
        self._lock = threading.Lock()
        self._open: _OpenMediaSession | None = None
        self._settings: MediaSettings | None = None
        self._armed = False
        # Read without the lock from media threads; replaced, never mutated.
        self._voice: VoiceSession | None = None

    @property
    def collecting(self) -> bool:
        return self.consent == "accepted"

    @property
    def media_open(self) -> bool:
        return self._open is not None

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

    @property
    def voice_enabled(self) -> bool:
        return self.voice_notice is not None

    def set_voice_consent(self, granted: bool) -> None:
        if not (self.voice_enabled and self.collecting) or granted == self.voice_consent:
            return
        self.voice_consent = granted
        self._write(
            VoiceConsentRecord,
            "voice_consent",
            notice_sha256=self.voice_notice.sha256,
            granted=granted,
        )
        if not granted:
            # Turning it off stops the recording under way at once; what it
            # holds so far is kept, and the delete button removes it.
            voice, self._voice = self._voice, None
            if voice is not None:
                voice.close("voice_consent_withdrawn")

    def record_mic(self, frame) -> None:
        voice = self._voice
        if voice is None and self._armed:
            # The page run that sees the session playing comes seconds
            # after its media starts, so waiting for it would cut the
            # start of what the visitor says.
            voice = self._open_media(
                self._settings, time.monotonic() - frame.samples / frame.sample_rate
            )
        if voice is not None:
            voice.push_mic(frame)

    def record_reply(self, samples, sample_rate: int) -> None:
        voice = self._voice
        if voice is not None:
            voice.push_reply(samples, sample_rate)

    def has_voice_recordings(self) -> bool:
        return has_voice_recordings(self.store, self.voice_key)

    def delete_voice_recordings(self) -> None:
        if self.voice_deletion is not None and self.voice_deletion.state == "running":
            return
        self._voice = None
        self.voice_deletion = VoiceDeletion("running", "")

        def run() -> None:
            try:
                delete_voice_recordings(self.store, self.voice_key)
            except Exception as exc:
                logger.warning("[session-records] voice deletion failed", exc_info=True)
                self.voice_deletion = VoiceDeletion("failed", f"{type(exc).__name__}: {exc}")
            else:
                self.voice_deletion = VoiceDeletion("done", "")

        threading.Thread(target=run, name="voice-records-delete", daemon=True).start()

    def prepare_media(self, settings: MediaSettings, *, playing: bool) -> None:
        """Called on each page run. Opens the media session if it plays
        already, and otherwise arms the next microphone frame to open it."""
        self._settings = settings
        if playing:
            self._open_media(settings, time.monotonic())
            self._write_start()
        else:
            self._armed = True

    def _open_media(self, settings: MediaSettings, started: float) -> VoiceSession | None:
        """Runs on media threads too, so it does no file or network I/O;
        the start record waits for the next page run or the session end."""
        with self._lock:
            # Only a page run that finds the session stopped arms it again.
            # Frames still arriving after the server stopped a session
            # would otherwise open a new one.
            self._armed = False
            if self._open is not None:
                return self._voice
            media_session = uuid.uuid4().hex
            # A session started before the visitor accepted stays unrecorded
            # to its end, so no end record lacks its start.
            recorded = self.collecting and self.log_sessions
            voice = None
            if self.collecting and self.voice_enabled and self.voice_consent:
                voice = start_voice_session(
                    self.store,
                    self.voice_key,
                    media_session,
                    # Loopback plays the microphone back, so its reply is
                    # the mic track already.
                    record_reply=settings.mode != "loopback",
                    started=started,
                    metadata={
                        "subject_kind": self.subject_kind,
                        "mode": settings.mode,
                        "backend": settings.backend,
                        "voice_notice_sha256": self.voice_notice.sha256,
                    },
                )
            self._open = _OpenMediaSession(
                media_session, settings, started, recorded, False, voice
            )
            self._voice = voice
            self.last_media_session = media_session if recorded else None
        return voice

    def _write_start(self) -> None:
        with self._lock:
            session = self._open
            if session is None or not session.recorded or session.start_written:
                return
            self._open = session._replace(start_written=True)
        self._write_start_record(session)

    def _write_start_record(self, session: _OpenMediaSession) -> None:
        self._write(
            MediaSessionStart,
            "media_session_start",
            media_session=session.media_session,
            **dataclasses.asdict(session.settings),
        )

    def end_media(self, reason: EndReason) -> None:
        # Called whenever the session's pipeline is stopped, and this
        # reference would otherwise keep a stopped pipeline's GPU memory
        # alive until the session's next script run.
        pipeline, self.pipeline = self.pipeline, None
        with self._lock:
            session, self._open = self._open, None
            self._armed = False
            self._voice = None
            if session is not None:
                self.media_sessions_ended += 1
        if session is None:
            return
        if session.voice is not None:
            session.voice.close(reason)
        if not session.recorded:
            return
        if not session.start_written:
            self._write_start_record(session)
        summary = {}
        if pipeline is not None:
            try:
                summary = pipeline_summary(pipeline.metrics_snapshot())
            except Exception:
                logger.warning("[session-records] no pipeline metrics", exc_info=True)
        self._write(
            MediaSessionEnd,
            "media_session_end",
            media_session=session.media_session,
            end_reason=reason,
            duration_s=round(time.monotonic() - session.started, 3),
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
        self.feedback_sent_for = self.media_sessions_ended


def render_feedback_slot(recorder: SessionRecorder) -> None:
    """The consent panel while the visitor has not answered it, then the
    feedback form, in one place so that accepting swaps one for the other."""
    if recorder.consent == "pending":
        _render_consent_panel(recorder)
    else:
        render_feedback_form(recorder)
    render_voice_controls(recorder)


def _render_consent_panel(recorder: SessionRecorder) -> None:
    kept = []
    if recorder.feedback:
        kept.append("the feedback you send")
    if recorder.log_sessions:
        kept.append("records of your sessions")
    if recorder.voice_enabled:
        kept.append("voice recordings if you turn them on")
    listed = kept[0] if len(kept) == 1 else f"{', '.join(kept[:-1])} and {kept[-1]}"
    with st.container(border=True):
        st.markdown("##### Help improve this demo")
        st.markdown(f"May this demo keep {listed}? Nothing is kept unless you accept.")
        with st.expander("Read the full notice"):
            st.markdown(recorder.notice.text)
        with st.container(horizontal=True):
            st.button(
                "Accept", key="consent_accept", type="primary", on_click=recorder.accept
            )
            st.button("No thanks", key="consent_decline", on_click=recorder.decline)


def render_feedback_form(recorder: SessionRecorder) -> None:
    if not (recorder.collecting and recorder.feedback):
        return
    ended = recorder.media_sessions_ended
    if recorder.media_open or ended == 0:
        return
    if recorder.feedback_sent_for == ended:
        st.success("Thanks, feedback sent.")
        return
    # Keys of their own give each session an empty form.
    rating_key = f"feedback_rating_{ended}"
    comment_key = f"feedback_comment_{ended}"
    missing_key = f"feedback_missing_rating_{ended}"

    def submit() -> None:
        rating = st.session_state[rating_key]
        if rating is None:
            st.session_state[missing_key] = True
            return
        recorder.submit_feedback(rating, st.session_state[comment_key])

    with st.form(f"feedback_form_{ended}", border=True):
        st.markdown("##### How was this session?")
        st.feedback("faces", key=rating_key)
        st.text_input("Comment (optional)", max_chars=COMMENT_MAX_CHARS, key=comment_key)
        # A callback rather than the button's return value: it runs before
        # the page, so the run that follows the click already shows thanks.
        st.form_submit_button("Send feedback", on_click=submit)
        if st.session_state.get(missing_key):
            st.warning("Pick a face first.")


def render_voice_controls(recorder: SessionRecorder) -> None:
    if not recorder.voice_enabled:
        return
    may_have_recordings = (
        recorder.voice_consent
        or recorder.voice_deletion is not None
        # Signed-in visitors' earlier sessions may have left recordings that
        # only the uploaded copy still holds.
        or recorder.subject_kind == "user"
        or recorder.has_voice_recordings()
    )
    if not (recorder.collecting or may_have_recordings):
        return
    with st.expander("Voice recording (optional)"):
        if recorder.collecting:
            st.markdown(recorder.voice_notice.text)
            st.toggle(
                "Record my voice and the avatar's spoken replies",
                value=recorder.voice_consent,
                key="voice_consent",
                on_change=lambda: recorder.set_voice_consent(st.session_state.voice_consent),
            )
            st.caption(
                "Applies from the next START. Turning it off stops a recording "
                "under way."
            )
        if not may_have_recordings:
            return
        # Not inside the polling fragment below, whose reruns lose clicks.
        st.button(
            "Delete my voice recordings",
            key="voice_delete",
            on_click=recorder.delete_voice_recordings,
        )
        if recorder.subject_kind == "user":
            scope = "Deletes every voice recording made while signed in to this account"
        else:
            scope = "Deletes the voice recordings of this browser session"
        if isinstance(recorder.store, HubRecordStore):
            scope += ", including uploaded copies and the history that still holds them"
        caption = f"{scope}. Ratings and session records are kept."
        if recorder.subject_kind != "user":
            caption += (
                " You are not signed in, so recordings from before a reload, "
                "which starts a new browser session, are not covered."
            )
        st.caption(caption)
        running = recorder.voice_deletion is not None and recorder.voice_deletion.state == "running"

        @st.fragment(run_every="1s" if running else None)
        def render_deletion_status() -> None:
            deletion = recorder.voice_deletion
            if deletion is None:
                return
            if deletion.state == "running":
                st.info("Deleting your voice recordings...")
            elif deletion.state == "done":
                st.success("Your voice recordings were deleted.")
            else:
                st.error(f"Deletion failed ({deletion.message}). Press the button to try again.")

        render_deletion_status()
