"""Bounded storage for the diagnostic dumps the server writes to disk.

Stalls come in bursts under load, and a dump per stall filled the disk of a
long-running public server with nothing to stop it. Each kind of dump is
therefore rate-limited and keeps only its newest files.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

MAX_FILES_PER_KIND = 50
MIN_INTERVAL_S = 30.0


class DumpFiles:
    def __init__(
        self,
        kind: str,
        directory: Path | str = "diagnostics_snapshots",
        *,
        max_files: int = MAX_FILES_PER_KIND,
        min_interval_s: float = MIN_INTERVAL_S,
    ) -> None:
        self.kind = kind
        self.directory = Path(directory)
        self._max_files = max_files
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last_claim_s: float | None = None

    def claim(self) -> Path | None:
        """The stem for a new dump, or None while the previous dump of this
        kind is under the minimum interval old. The caller adds the suffix,
        writes the file, then calls ``prune``."""
        now = time.monotonic()
        with self._lock:
            if (
                self._last_claim_s is not None
                and now - self._last_claim_s < self._min_interval_s
            ):
                return None
            self._last_claim_s = now
        self.directory.mkdir(parents=True, exist_ok=True)
        return self.directory / f"{self.kind}-{time.strftime('%Y%m%d-%H%M%S')}"

    def prune(self) -> None:
        """Delete all but the newest files of this kind."""
        try:
            files = sorted(
                self.directory.glob(f"{self.kind}-*"),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            logger.warning("[dumps] listing %s failed", self.directory, exc_info=True)
            return
        for path in files[self._max_files :]:
            path.unlink(missing_ok=True)
