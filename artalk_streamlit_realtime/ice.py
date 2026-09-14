"""ICE server configuration for the WebRTC streamer."""

from __future__ import annotations

import json
import time
import urllib.request

import streamlit as st

STUN_ONLY = {"iceServers": [{"urls": "stun:stun.l.google.com:19302"}]}

# FastRTC's credential service for the Cloudflare TURN relay that Hugging
# Face accounts get 10 GB/month of; the same endpoint fastrtc's
# get_cloudflare_turn_credentials uses.
CLOUDFLARE_CREDENTIALS_URL = "https://turn.fastrtc.org/credentials"
CREDENTIAL_TTL_SECONDS = 3600
# Credentials are only consumed when a connection is negotiated, which can be
# long after this session first rendered, so refresh well ahead of expiry.
REFRESH_MARGIN_SECONDS = 300

_SESSION_KEY = "artalk_rtc_configuration"


def fetch_cloudflare_rtc_configuration(
    hf_token: str, ttl: int = CREDENTIAL_TTL_SECONDS
) -> dict:
    request = urllib.request.Request(
        f"{CLOUDFLARE_CREDENTIALS_URL}?ttl={ttl}",
        headers={"Authorization": f"Bearer {hf_token}"},
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
    from huggingface_hub import get_token

    token = get_token()
    if not token:
        raise RuntimeError(
            "the cloudflare ICE provider needs a Hugging Face token "
            "(HF_TOKEN or `hf auth login`)"
        )
    config = fetch_cloudflare_rtc_configuration(token)
    st.session_state[_SESSION_KEY] = {
        "config": config,
        "expires_at": now + CREDENTIAL_TTL_SECONDS,
    }
    return config
