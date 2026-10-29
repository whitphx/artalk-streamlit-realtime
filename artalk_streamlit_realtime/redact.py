"""Scrubbing for server-side text that a visitor's page shows.

Error messages and event details come from exceptions, subprocesses and
remote APIs, so they can carry what the server knows and a public page must
not: file paths, host names, credentials and e-mail addresses. The server
log keeps the original.
"""

from __future__ import annotations

import re

_RULES = (
    # Before the path rule, which would otherwise eat a URL's path.
    (re.compile(r"\b([a-z][a-z0-9+.-]*)://[^\s'\"<>)\]]+", re.IGNORECASE), r"\1://[host]"),
    (re.compile(r"\b(?:sk|hf|xai|ghp|gho|ghs)[-_][\w*-]{6,}"), "[key]"),
    (re.compile(r"(?i)\b(bearer)\s+\S+"), r"\1 [key]"),
    (re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+"), "[email]"),
    (re.compile(r"(?<![\w.:/\[])(?:~|/)(?:[\w.@+-]+/)+[\w.@+-]*"), "[path]"),
    (re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b"), "[ip]"),
    # host:port, as connection errors name the server they failed to reach.
    (re.compile(r"\b[a-z][\w-]*(?:\.[\w-]+)*:\d{2,5}\b", re.IGNORECASE), "[host]"),
)


def redact(text: str) -> str:
    for pattern, replacement in _RULES:
        text = pattern.sub(replacement, text)
    return text
