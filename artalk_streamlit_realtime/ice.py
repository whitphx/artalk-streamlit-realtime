"""ICE server configuration for the WebRTC streamer."""

from __future__ import annotations

import json
import os
import time
import urllib.request

import streamlit as st

STUN_ONLY = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}

CREDENTIAL_TTL_SECONDS = 3600
# Credentials are only consumed when a connection is negotiated, which can be
# long after this session first rendered, so refresh well ahead of expiry.
REFRESH_MARGIN_SECONDS = 300

_SESSION_KEY = "artalk_rtc_configuration"


def fetch_cloudflare_rtc_configuration(
    key_id: str, api_token: str, ttl: int = CREDENTIAL_TTL_SECONDS
) -> dict:
    """Short-lived relay credentials from Cloudflare Realtime TURN.

    https://developers.cloudflare.com/realtime/turn/generate-credentials/
    """
    request = urllib.request.Request(
        f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}"
        "/credentials/generate-ice-servers",
        data=json.dumps({"ttl": ttl}).encode(),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            # Cloudflare's edge answers urllib's default agent with a 403
            # (error 1010, "banned based on browser signature").
            "User-Agent": "artalk-streamlit-realtime",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=10) as response:
        config = json.load(response)
    if not config.get("iceServers"):
        raise ValueError(f"credential service returned no iceServers: {config!r}")
    return config


def resolve_rtc_configuration(provider: str) -> dict:
    """This session's rtc_configuration, refreshed before its credentials expire."""
    if provider == "stun":
        return STUN_ONLY
    if provider != "cloudflare":
        raise ValueError(f"unknown ICE provider {provider!r}")
    cached = st.session_state.get(_SESSION_KEY)
    now = time.monotonic()
    if cached is not None and cached["expires_at"] - now > REFRESH_MARGIN_SECONDS:
        return cached["config"]
    # Secrets pasted into a settings page tend to arrive with a trailing newline.
    key_id = os.environ.get("CLOUDFLARE_TURN_KEY_ID", "").strip()
    api_token = os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN", "").strip()
    if not key_id or not api_token:
        raise RuntimeError(
            "the cloudflare ICE provider needs CLOUDFLARE_TURN_KEY_ID and "
            "CLOUDFLARE_TURN_KEY_API_TOKEN"
        )
    config = fetch_cloudflare_rtc_configuration(key_id, api_token)
    st.session_state[_SESSION_KEY] = {
        "config": config,
        "expires_at": now + CREDENTIAL_TTL_SECONDS,
    }
    return config
