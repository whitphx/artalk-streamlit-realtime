"""User-registered GAGAvatar avatars: on-demand tracking and storage.

One directory per avatar under the registry root, holding the uploaded
source image and the tracked entry (``tracked.pt``) produced by the
GAGAvatar_track subprocess. Per-avatar files keep the built-in
``tracked.pt`` asset pristine and make writes atomic and deletion trivial.
"""

from __future__ import annotations

import hashlib
import logging
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Iterable
from pathlib import Path

import torch

from .redact import redact

logger = logging.getLogger(__name__)

TRACK_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "track_avatar.py"
TRACK_TIMEOUT_S = 600.0


class AvatarRegistrationError(RuntimeError):
    pass


def slugify_avatar_id(name: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", name.strip().lower()).strip("-.")


def owner_dir(root: Path | str, kind: str, owner_id: str) -> Path:
    """The registry root for one owner's avatars under the shared root.

    The name is a digest, so a directory listing does not reveal who
    uploaded what (owner ids can be e-mail addresses).
    """
    digest = hashlib.sha256(owner_id.encode()).hexdigest()[:32]
    return Path(root) / f"{kind}-{digest}"


_session_dirs: dict[str, Path] = {}
_session_dirs_lock = threading.Lock()


def session_owner_dir(root: Path | str, session_id: str) -> Path:
    """``owner_dir`` for a browser session, remembered so that
    ``prune_session_owners`` can delete it once the session is gone."""
    path = owner_dir(root, "session", session_id)
    with _session_dirs_lock:
        _session_dirs[session_id] = path
    return path


def prune_session_owners(live_session_ids: Iterable[str]) -> None:
    """Delete the avatars of this process's browser sessions that Streamlit
    can no longer resume.

    A reload starts a new session, so nothing can select these again, and
    they hold faces their uploader may not expect the server to keep. Only
    directories handed out by ``session_owner_dir`` in this process are
    candidates, not every ``session-*`` entry under the root: another
    process sharing the root has live sessions this one cannot see.
    """
    live = set(live_session_ids)
    with _session_dirs_lock:
        gone = [sid for sid in _session_dirs if sid not in live]
        paths = [_session_dirs.pop(sid) for sid in gone]
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)


class UserAvatarRegistry:
    def __init__(
        self,
        root: Path | str,
        track_python: str | None = None,
        track_dir: str | None = None,
        track_device: str = "cpu",
    ) -> None:
        self.root = Path(root)
        self._track_python = track_python
        self._track_dir = track_dir
        self.track_device = track_device

    @property
    def can_register(self) -> bool:
        return bool(self._track_python and self._track_dir)

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(path.parent.name for path in self.root.glob("*/tracked.pt"))

    def load(self, avatar_id: str) -> dict:
        # Entries are written by scripts/track_avatar.py as tensors only,
        # so the safe loading mode suffices.
        return torch.load(
            self.root / avatar_id / "tracked.pt",
            map_location="cpu",
            weights_only=True,
        )

    def source_path(self, avatar_id: str) -> Path | None:
        matches = sorted((self.root / avatar_id).glob("source.*"))
        return matches[0] if matches else None

    def delete(self, avatar_id: str) -> None:
        shutil.rmtree(self.root / avatar_id)

    def last_track_seconds(self) -> float | None:
        """Duration of the most recent successful tracking run, persisted so
        the UI can show a measured estimate instead of a guess."""
        try:
            return float((self.root / ".last_track_seconds").read_text())
        except (OSError, ValueError):
            return None

    def allocate_id(self, base_id: str, reserved: Iterable[str] = ()) -> str:
        """``base_id`` if free, else the first free ``base_id-N``."""
        taken = set(self.list_ids()) | set(reserved)
        if base_id not in taken:
            return base_id
        n = 2
        while f"{base_id}-{n}" in taken:
            n += 1
        return f"{base_id}-{n}"

    def register(self, avatar_id: str, image_bytes: bytes, suffix: str) -> dict:
        if not self.can_register:
            raise AvatarRegistrationError(
                "Avatar tracking is not configured; set GAGAVATAR_TRACK_PYTHON "
                "and GAGAVATAR_TRACK_DIR (or the matching CLI options)."
            )
        avatar_dir = self.root / avatar_id
        self.root.mkdir(parents=True, exist_ok=True)
        try:
            # Creating the directory reserves the id; callers pick a free one
            # via allocate_id(), and this converts races into a clean error
            # instead of an overwrite.
            avatar_dir.mkdir()
        except FileExistsError:
            raise AvatarRegistrationError(
                f"Avatar `{avatar_id}` already exists."
            ) from None
        source_path = avatar_dir / f"source{suffix}"
        source_path.write_bytes(image_bytes)
        tracked_path = avatar_dir / "tracked.pt"
        command = [
            self._track_python,
            str(TRACK_SCRIPT),
            "--image",
            str(source_path),
            "--output",
            str(tracked_path),
            "--track-dir",
            self._track_dir,
            "--device",
            self.track_device,
        ]
        track_started = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=TRACK_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired:
            shutil.rmtree(avatar_dir, ignore_errors=True)
            raise AvatarRegistrationError(
                f"Avatar tracking timed out after {TRACK_TIMEOUT_S:.0f} s."
            ) from None
        if proc.returncode != 0 or not tracked_path.exists():
            shutil.rmtree(avatar_dir, ignore_errors=True)
            logger.warning(
                "[avatars] tracking %s failed with exit code %s\n"
                "command: %s\n--- stdout ---\n%s\n--- stderr ---\n%s",
                avatar_id,
                proc.returncode,
                " ".join(command),
                proc.stdout or "",
                proc.stderr or "",
            )
            # CUDA failures span several lines and the informative one is
            # rarely last; prefer lines that name an error.
            lines = [
                line
                for line in (proc.stderr or proc.stdout or "").splitlines()
                if line.strip()
            ]
            tail = lines[-8:]
            named = [l for l in tail if "rror" in l or "failed" in l]
            detail = " | ".join(named or tail[-2:]) or f"exit code {proc.returncode}"
            raise AvatarRegistrationError(
                f"Avatar tracking failed: {redact(detail)[:400]}"
            )
        elapsed_s = time.monotonic() - track_started
        try:
            (self.root / ".last_track_seconds").write_text(f"{elapsed_s:.1f}")
        except OSError:
            pass
        return self.load(avatar_id)
