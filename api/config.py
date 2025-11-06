"""API configuration modeled as Pydantic settings."""
from __future__ import annotations

from functools import lru_cache
from typing import List, Optional

from pydantic import AnyUrl, Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Runtime configuration for the platform API."""

    database_url: str = Field(
        default="sqlite:///./epochor.db",
        description="SQLAlchemy database URL.",
    )
    api_token: Optional[str] = Field(
        default=None,
        description="Shared bearer token for privileged API routes.",
    )
    allowed_miner_hotkeys: List[str] = Field(
        default_factory=list,
        description="Optional allowlist of miner hotkeys permitted to submit.",
    )
    allowed_validator_hotkeys: List[str] = Field(
        default_factory=list,
        description="Optional allowlist of validator hotkeys permitted to lease jobs.",
    )
    shallow_pool_lease_seconds: int = Field(default=45 * 60, ge=60)
    medium_pool_lease_seconds: int = Field(default=60 * 60, ge=60)
    final_pool_lease_seconds: int = Field(default=90 * 60, ge=60)
    shallow_pool_promotion_threshold: float = Field(
        default=0.0,
        description="Minimum score required before promoting from shallow to medium.",
    )
    medium_pool_promotion_threshold: float = Field(
        default=0.0,
        description="Minimum score required before promoting from medium to final.",
    )
    max_medium_promotions: int = Field(default=5, ge=0)
    max_final_promotions: int = Field(default=3, ge=0)
    api_base_url: Optional[AnyUrl] = None
    heartbeat_timeout_seconds: int = Field(default=10 * 60, ge=60)

    class Config:
        env_prefix = "EPOCHOR_"
        case_sensitive = False


@lru_cache
def get_settings() -> Settings:
    """Return cached settings instance."""

    return Settings()  # type: ignore[call-arg]


__all__ = ["Settings", "get_settings"]
