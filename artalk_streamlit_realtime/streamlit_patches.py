"""Keep Streamlit's source watcher off the WebRTC event loop.

Whenever ``sys.modules`` changes, Streamlit rescans every module and
resolves its paths with ``realpath``/``isfile``/``isdir``, and it does so
on the event loop that also drives the WebRTC tracks
(``AppSession._handle_scriptrunner_event_on_event_loop`` ->
``LocalSourcesWatcher.update_watched_modules``). This app loads torch,
transformers and friends, which makes that ~3900 modules and ~600 ms per
scan on a local disk, multiple seconds on the NFS-mounted environment.
While it runs no RTP leaves the process at all, so the browser conceals
audio and freezes video, and the two drift apart by seconds. Any lazy
import mid-session triggers another scan, which is why the stalls come
and go.

``server.fileWatcherType = "none"`` does not avoid this: that setting is
checked in ``_register_watcher``, after the expensive scan has already
run.

Hot reload is not useful for a server that loads several GB of weights at
startup, so the scan is disabled outright. Set
``ARTALK_STREAMLIT_SOURCE_WATCHER=1`` to restore Streamlit's behaviour.
"""

from __future__ import annotations

import logging

from .config import _env_flag

logger = logging.getLogger(__name__)


def disable_streamlit_source_watcher() -> None:
    if _env_flag("ARTALK_STREAMLIT_SOURCE_WATCHER", False):
        return
    try:
        from streamlit.watcher.local_sources_watcher import LocalSourcesWatcher
    except ImportError:
        logger.warning(
            "[source-watcher] streamlit.watcher.local_sources_watcher moved; "
            "the module scan is still on the event loop", exc_info=True,
        )
        return
    if not hasattr(LocalSourcesWatcher, "update_watched_modules"):
        logger.warning(
            "[source-watcher] LocalSourcesWatcher.update_watched_modules is "
            "gone; leaving Streamlit alone"
        )
        return

    def update_watched_modules(self) -> None:
        return None

    LocalSourcesWatcher.update_watched_modules = update_watched_modules
    logger.warning(
        "[source-watcher] disabled: source edits no longer trigger a rerun "
        "(set ARTALK_STREAMLIT_SOURCE_WATCHER=1 to keep it)"
    )
