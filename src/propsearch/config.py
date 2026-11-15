from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_env: Literal["dev", "test", "prod"] = "dev"
    database_url: str = "postgresql+psycopg://propsearch:propsearch@localhost:5433/propsearch"

    # Query understanding: "anthropic" (Claude tool calling) or "llm" (any OpenAI-compatible API: Groq,
    # OpenRouter, ...). Both fall back to rules on any failure.
    parser: Literal["anthropic", "llm", "rules"] = "rules"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high"] = "low"
    anthropic_timeout_s: float = 20.0
    llm_base_url: str = "https://api.groq.com/openai/v1"
    llm_api_key: str = ""
    llm_model: str = "openai/gpt-oss-120b"
    llm_reasoning_effort: str = "low"  # empty string to omit (models without reasoning controls)
    llm_timeout_s: float = 15.0
    # "on_unknown": rules first, LLM only when the rules hit unknown terms. "always": LLM on every turn.
    llm_mode: Literal["on_unknown", "always"] = "on_unknown"
    # Optional grounded narrative for the reply (numbers checked against DB fields). Off = templates only.
    llm_summary: bool = False

    # Retrieval models. "hashing" needs no download (tests, CI, small instances).
    embedder: Literal["bge-m3", "hashing"] = "hashing"
    reranker: Literal["bge", "none"] = "none"
    # cpu / cuda: local sentence-transformers. hf: Hugging Face Inference (HF_TOKEN, needs credits).
    # precomputed: query vectors and reranker scores computed offline and read from Postgres (no model
    # at serve time; see retrieval/precomputed.py).
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
