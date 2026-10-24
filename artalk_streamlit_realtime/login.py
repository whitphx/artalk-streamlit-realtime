"""Optional login through Streamlit's OpenID Connect support (``st.login``).

Streamlit reads the identity provider from the ``[auth]`` section of
``secrets.toml``. On a Hugging Face Space, ``hf_oauth: true`` provides an
OIDC client through environment variables instead, which
``write_hf_oauth_secrets`` turns into that section at container start.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Collection, Mapping
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError


def parse_email_domains(value: str) -> frozenset[str]:
    return frozenset(d.strip().lower() for d in value.split(",") if d.strip())


def email_domain_admits(user: Mapping[str, object], domains: Collection[str]) -> bool:
    """Whether the logged-in ``user`` claims a verified e-mail address in one
    of ``domains`` (lowercase). An empty ``domains`` admits everyone."""
    if not domains:
        return True
    email = user.get("email")
    if not isinstance(email, str) or "@" not in email:
        return False
    # Hugging Face sends email_verified alongside email
    # (https://huggingface.co/docs/huggingface_hub/v0.36.2/en/package_reference/oauth#huggingface_hub.OAuthUserInfo),
    # but other OIDC providers may leave it out; their address is taken as
    # given. A "true" string is not a verification.
    if "email_verified" in user and user["email_verified"] is not True:
        return False
    # Exact match only, so that admitting a subdomain takes listing it.
    return email.rpartition("@")[2].lower() in domains


def require_login(email_domains: Collection[str] = frozenset()) -> str:
    """Stop the script run behind a login prompt unless the visitor is
    logged in, and with ``email_domains`` unless their verified e-mail
    address is in one of them. Return an id that is stable across their
    sessions."""
    try:
        configured = "auth" in st.secrets
    except StreamlitSecretNotFoundError:
        configured = False
    if not configured:
        st.error(
            "Login is enabled, but no identity provider is configured: "
            "`.streamlit/secrets.toml` needs an `[auth]` section."
        )
        st.stop()
    if not st.user.is_logged_in:
        st.info("Log in to use this demo.")
        st.button("Log in", on_click=st.login, type="primary")
        st.stop()
    if not email_domain_admits(st.user, email_domains):
        who = st.user.get("email") or st.user.get("preferred_username")
        # The allowed domains stay unnamed, since the page is public.
        st.error(
            f"You are logged in as **{who or st.user.get('sub')}**, which is "
            "not allowed to use this demo."
        )
        st.button("Log out", on_click=st.logout)
        st.stop()
    with st.sidebar:
        name = st.user.get("preferred_username") or st.user.get("name")
        st.caption(f"Logged in as **{name or st.user.get('sub')}**")
        st.button("Log out", on_click=st.logout)
    # `sub` is unique only per issuer.
    return f"{st.user.get('iss', '')}|{st.user.get('sub')}"


def hf_oauth_auth_section(env: Mapping[str, str]) -> str | None:
    """The ``[auth]`` section for the OIDC client a Space's HF OAuth provides,
    or None outside such a Space."""
    client_id = env.get("OAUTH_CLIENT_ID")
    client_secret = env.get("OAUTH_CLIENT_SECRET")
    provider_url = env.get("OPENID_PROVIDER_URL")
    space_host = env.get("SPACE_HOST")
    if not (client_id and client_secret and provider_url and space_host):
        return None
    values = {
        # Any path on the Space is an accepted redirect target
        # (https://huggingface.co/docs/hub/spaces-oauth#redirect-urls), and
        # Streamlit only answers on this one.
        "redirect_uri": f"https://{space_host}/oauth2callback",
        # A secret that changes on every start logs everyone out on restart;
        # set ARTALK_AUTH_COOKIE_SECRET as a Space secret to keep logins.
        "cookie_secret": env.get("ARTALK_AUTH_COOKIE_SECRET") or secrets.token_hex(32),
        "client_id": client_id,
        "client_secret": client_secret,
        "server_metadata_url": (
            f"{provider_url.rstrip('/')}/.well-known/openid-configuration"
        ),
    }
    # JSON string literals are valid TOML basic strings.
    lines = ["[auth]", *(f"{k} = {json.dumps(v)}" for k, v in values.items())]
    # Streamlit asks for e-mail by default, which the Space's OAuth app is not
    # granted unless its metadata lists it; ask for what it was granted.
    scope = env.get("OAUTH_SCOPES") or "openid profile"
    # HF includes the e-mail claims only for the email scope, and the
    # domain check turns away anyone without them.
    if parse_email_domains(env.get("ARTALK_LOGIN_EMAIL_DOMAINS", "")) and (
        "email" not in scope.split()
    ):
        scope += " email"
    lines += ["", "[auth.client_kwargs]", f"scope = {json.dumps(scope)}"]
    # Streamlit defaults to prompt="select_account", which HF rejects
    # (it accepts consent, login and none), sending the visitor back
    # logged out. TOML cannot leave the key unset, and "none" fails for
    # anyone who has not approved the app yet.
    lines.append('prompt = "consent"')
    return "\n".join(lines) + "\n"


def write_hf_oauth_secrets(path: Path, env: Mapping[str, str] = os.environ) -> bool:
    """Append the HF OAuth ``[auth]`` section to ``path`` unless the file
    already configures a provider. True when it wrote one."""
    section = hf_oauth_auth_section(env)
    if section is None:
        return False
    existing = path.read_text() if path.exists() else ""
    if any(line.strip().startswith("[auth") for line in existing.splitlines()):
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    separator = "\n" if existing and not existing.endswith("\n") else ""
    # The section carries the OAuth client secret and the cookie key.
    path.touch(mode=0o600, exist_ok=True)
    path.chmod(0o600)
    path.write_text(existing + separator + section)
    return True
