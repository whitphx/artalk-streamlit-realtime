"""Credentials the operator supplies to the Streamlit apps."""

from __future__ import annotations

import functools
import logging
import os

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from .config import REALTIME_ENDPOINT_PRESETS

logger = logging.getLogger(__name__)

API_KEY_NAMES = tuple(
    sorted(
        {"OPENAI_API_KEY"}
        | {endpoint.secret for endpoint in REALTIME_ENDPOINT_PRESETS.values() if endpoint.secret}
    )
)


def _lookup(name: str) -> tuple[str, str]:
    try:
        value = st.secrets.get(name)
    except StreamlitSecretNotFoundError:
        value = None
    if value:
        return str(value), "Streamlit secrets"
    # A Space delivers its secrets as environment variables.
    value = os.environ.get(name)
    if value:
        return value, "environment"
    return "", ""


def get_secret(name: str, default: str = "") -> str:
    """A secret from Streamlit's secrets file, else from the environment."""
    return _lookup(name)[0] or default


# Cached rather than logged from the page script, which Streamlit runs again
# on every interaction.
@functools.cache
def log_api_key_sources() -> None:
    """Record once per process which API keys the server holds, so the
    operator can check the configuration without the page showing it."""
    for name in API_KEY_NAMES:
        source = _lookup(name)[1]
        # Nothing configures this package's loggers, so Python's fallback
        # handler prints only warnings and above.
        logger.warning(
            "[config] %s: %s", name, f"from {source}" if source else "not configured"
        )
