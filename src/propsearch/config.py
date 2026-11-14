from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    database_url: str = "postgresql+psycopg://propsearch:propsearch@localhost:5433/propsearch"

    # Query understanding: "anthropic" uses Claude tool calling and falls back to rules on any failure.
    parser: Literal["anthropic", "rules"] = "rules"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "low"
    anthropic_timeout_s: float = 20.0
    # Optional grounded narrative for the reply (numbers checked against DB fields). Off = templates only.
    llm_summary: bool = False

    # Retrieval models. "hashing" needs no download (tests, CI, small instances).
    embedder: Literal["bge-m3", "hashing"] = "hashing"
    reranker: Literal["bge", "none"] = "none"
    embed_device: str = "cpu"
    top_k: int = 8

    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_host: str = "https://cloud.langfuse.com"

    cors_origins: list[str] = ["http://localhost:3000"]

    @field_validator("database_url")
    @classmethod
    def _psycopg_driver(cls, v: str) -> str:
        # Hosted Postgres (Render, Neon, Supabase) hands out postgres:// URLs; SQLAlchemy needs the driver.
        for prefix in ("postgres://", "postgresql://"):
            if v.startswith(prefix):
                return "postgresql+psycopg://" + v[len(prefix) :]
        return v


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
