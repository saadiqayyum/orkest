"""Configuration loaded from environment variables."""

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """All settings for the SEO Agent, loaded from .env file."""

    # API Keys
    google_api_key: str = ""
    gemini_model_id: str = "gemini-2.0-flash"
    serpapi_key: str = ""
    moz_access_id: str = ""
    moz_secret_key: str = ""
    ahrefs_api_key: str = ""  # Optional, preferred over Moz if available
    pagespeed_api_key: str = ""

    # Target
    target_domain: str = ""

    # Infrastructure
    redis_url: str = "redis://localhost:6379"

    # Tuning
    max_competitors: int = Field(default=5, ge=1, le=20)
    cache_ttl_hours: int = Field(default=24, ge=1)

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
