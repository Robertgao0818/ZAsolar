"""Vexcel API 2.0 auth helpers.

Vexcel-issued tokens expire within ~24h. To avoid manual re-pasting, store
``VEXCEL_USER`` + ``VEXCEL_PASSWORD`` in ``.env`` and let scripts mint a fresh
token at startup via ``resolve_token``. Static ``VEXCEL_TOKEN`` remains as
a fallback when credentials are absent.
"""

from __future__ import annotations

from pathlib import Path

import requests


DEFAULT_BASE_URL = "https://api.vexcelgroup.com/v2"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def mint_token(base_url: str, username: str, password: str, *, timeout: int = 30) -> str:
    url = f"{base_url.rstrip('/')}/auth/login"
    response = requests.post(
        url,
        json={"username": username, "password": password},
        headers={"Content-Type": "application/json; charset=utf-8"},
        timeout=timeout,
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"Vexcel auth failed: HTTP {response.status_code} {response.text[:240]}"
        )
    data = response.json()
    token = data.get("token")
    if not token:
        raise RuntimeError(f"Vexcel auth response missing 'token': {data!r}")
    return token


def resolve_token(env: dict[str, str], base_url: str) -> str:
    """Return a valid Vexcel API token.

    Prefers minting a fresh token from ``VEXCEL_USER`` + ``VEXCEL_PASSWORD``.
    Falls back to a pre-pasted ``VEXCEL_TOKEN`` if credentials are missing.
    """
    user = env.get("VEXCEL_USER")
    password = env.get("VEXCEL_PASSWORD")
    if user and password:
        return mint_token(base_url, user, password)
    static_token = env.get("VEXCEL_TOKEN")
    if static_token:
        return static_token
    raise RuntimeError(
        "No Vexcel credentials: set VEXCEL_USER + VEXCEL_PASSWORD (preferred) "
        "or VEXCEL_TOKEN in .env"
    )
