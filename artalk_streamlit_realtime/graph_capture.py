"""Process-wide coordination between CUDA graph captures and RNG draws.

A CUDA graph capture is a process-global affair twice over: device-wide
calls (``torch.cuda.synchronize``, ``empty_cache``) are forbidden while
any capture is open, and ``torch.cuda.graph`` registers the device's
default RNG generator, so a ``normal_()`` dispatched from any other
thread mid-capture raises "Offset increment outside graph capture" and
an aborted capture poisons the allocator for every later capture. Both
happen in practice: Streamlit sessions build pipelines concurrently,
each warming its own decoder and upsampler graphs on its own worker
thread while other pipelines keep rendering.

Captures therefore take the exclusive side of this coordinator and RNG
draws the shared side. A thread that opens a capture from inside its own
shared section (the upsampler captures lazily mid-render) steps its own
share aside for the duration, so the nesting cannot deadlock.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager


class GraphCaptureCoordinator:
    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._shared = 0
        self._capturing = False
        # Captures block new shared entrants while waiting, or steady
        # rendering at 25 fps would starve them indefinitely.
        self._captures_waiting = 0
        self._local = threading.local()

    @contextmanager
    def rng_work(self):
        """Shared section for GPU work that draws from the default RNG."""
        held = getattr(self._local, "shared", 0)
        if held == 0:
            with self._cond:
                while self._capturing or self._captures_waiting:
                    self._cond.wait()
                self._shared += 1
        self._local.shared = held + 1
        try:
            yield
        finally:
            self._local.shared -= 1
            if self._local.shared == 0:
                with self._cond:
                    self._shared -= 1
                    self._cond.notify_all()

    @contextmanager
    def capture(self):
        """Exclusive section for recording a CUDA graph."""
        own_share = 1 if getattr(self._local, "shared", 0) else 0
        with self._cond:
            self._captures_waiting += 1
            # Step this thread's own share aside before waiting, or two
            # threads capturing from inside their shared sections would
            # each wait forever on the other's share.
            self._shared -= own_share
            self._cond.notify_all()
            try:
                while self._capturing or self._shared > 0:
                    self._cond.wait()
                self._capturing = True
            finally:
                self._captures_waiting -= 1
        try:
            yield
        finally:
            with self._cond:
                self._capturing = False
                self._shared += own_share
                self._cond.notify_all()


coordinator = GraphCaptureCoordinator()


def serialize_gagavatar_captures() -> None:
    """Route GAGAvatar's upsampler graph captures through the coordinator.

    The pinned package predates this module, so wrap its capture method
    here; idempotent, and skipped when the class does not look like the
    code this was written against.
    """
    from gagavatar import runtime as gagavatar_runtime

    replay = getattr(gagavatar_runtime, "CudaGraphReplay", None)
    if replay is None or getattr(replay, "_capture_coordinated", False):
        return
    orig_capture = replay._capture

    def coordinated_capture(self, x):
        with coordinator.capture():
            return orig_capture(self, x)

    replay._capture = coordinated_capture
    replay._capture_coordinated = True
