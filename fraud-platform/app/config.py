"""Service settings, read from environment variables (and a local .env file if present).

Every setting can be overridden by an environment variable of the same name in upper case,
e.g. MODEL_PATH=models/logreg_v1.joblib. In Docker Compose these come from .env.
"""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", protected_namespaces=())

    redis_url: str = "redis://localhost:6379/0"
    # SQLite by default so the service runs without Postgres; Compose points this at Postgres.
    database_url: str = "sqlite:///./data/decisions.db"

    # Which model tier 1 uses. "builtin:heuristic" is the hand-weighted placeholder.
    model_path: str = "builtin:heuristic"
    # rules_only: no model at all, any soft rule hit sends the transaction to review.
    # rules_then_model: hard rules can block, everything else goes to the model.
    scoring_mode: Literal["rules_only", "rules_then_model"] = "rules_then_model"
    review_threshold: float = 0.5
    block_threshold: float = 0.9

    # If set, POST /score requires the header X-API-Key with this value.
    api_key: str | None = None

    # Seconds to wait for Redis before giving up. Untuned on purpose: see ticket FRAUD-5.
    redis_timeout_s: float = 1.0
    decision_log_enabled: bool = True
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
