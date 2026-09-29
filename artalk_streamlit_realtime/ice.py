"""ICE server configuration for the WebRTC streamer."""

from __future__ import annotations

import os

from streamlit_webrtc import get_cloudflare_ice_servers

STUN_ONLY = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}


def resolve_rtc_configuration(provider: str) -> dict:
    if provider == "stun":
        return STUN_ONLY
    if provider != "cloudflare":
        raise ValueError(f"unknown ICE provider {provider!r}")
    # Secrets pasted into a settings page tend to arrive with a trailing newline.
    key_id = os.environ.get("CLOUDFLARE_TURN_KEY_ID", "").strip()
    api_token = os.environ.get("CLOUDFLARE_TURN_KEY_API_TOKEN", "").strip()
    if not key_id or not api_token:
        raise RuntimeError(
            "the cloudflare ICE provider needs CLOUDFLARE_TURN_KEY_ID and "
            "CLOUDFLARE_TURN_KEY_API_TOKEN"
        )
    # streamlit-webrtc caches these process-wide, so this needs no cache
    # of its own.
    return {"iceServers": get_cloudflare_ice_servers(key_id, api_token)}
