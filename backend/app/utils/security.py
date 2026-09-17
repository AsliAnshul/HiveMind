"""API key authentication.

Authentication is off when ``API_KEYS`` is empty (handy for local development)
and mandatory the moment one or more keys are configured — which you should do
before exposing the server to the internet, since both agents talk to it
without a user session.
"""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, status

from app.utils.config import get_settings


async def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """FastAPI dependency enforcing the ``X-API-Key`` header."""
    settings = get_settings()
    if not settings.auth_enabled:
        return

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-API-Key header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if not any(hmac.compare_digest(x_api_key, key) for key in settings.api_key_list):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Invalid API key."
        )
