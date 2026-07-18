"""User-registered GAGAvatar avatars: on-demand tracking and storage.

One directory per avatar under the registry root, holding the uploaded
source image and the tracked entry (``tracked.pt``) produced by the
GAGAvatar_track subprocess. Per-avatar files keep the built-in
``tracked.pt`` asset pristine and make writes atomic and deletion trivial.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import torch

TRACK_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "track_avatar.py"
TRACK_TIMEOUT_S = 600.0


class AvatarRegistrationError(RuntimeError):
    pass


def slugify_avatar_id(name: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", name.strip().lower()).strip("-.")


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

    def register(self, avatar_id: str, image_bytes: bytes, suffix: str) -> dict:
        if not self.can_register:
            raise AvatarRegistrationError(
                "Avatar tracking is not configured; set GAGAVATAR_TRACK_PYTHON "
                "and GAGAVATAR_TRACK_DIR (or the matching CLI options)."
            )
        avatar_dir = self.root / avatar_id
        avatar_dir.mkdir(parents=True, exist_ok=True)
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
            lines = (proc.stderr or proc.stdout or "").strip().splitlines()
            detail = lines[-1] if lines else f"exit code {proc.returncode}"
            raise AvatarRegistrationError(f"Avatar tracking failed: {detail}")
        return self.load(avatar_id)
