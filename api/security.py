"""Simple token and hotkey validation helpers."""
from __future__ import annotations

from typing import Optional

from fastapi import Depends, Header, HTTPException, status

from .config import Settings, get_settings


def _normalize_token(token: Optional[str]) -> Optional[str]:
    if token is None:
        return None
    token = token.strip()
    return token or None


def verify_bearer_token(
    authorization: Optional[str] = Header(default=None),
    settings: Settings = Depends(get_settings),
) -> None:
    """Validate the provided bearer token when one is configured."""

    expected = _normalize_token(settings.api_token)
    if expected is None:
        return
    provided = None
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:]
    if provided != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token")


def enforce_miner_hotkey(hotkey: str, settings: Settings) -> None:
    allowlist = set(settings.allowed_miner_hotkeys)
    if allowlist and hotkey not in allowlist:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Miner hotkey not allowed")


def enforce_validator_hotkey(hotkey: str, settings: Settings) -> None:
    allowlist = set(settings.allowed_validator_hotkeys)
    if allowlist and hotkey not in allowlist:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Validator hotkey not allowed")


__all__ = [
    "verify_bearer_token",
    "enforce_miner_hotkey",
    "enforce_validator_hotkey",
]
