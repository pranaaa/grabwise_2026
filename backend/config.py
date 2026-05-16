"""Central settings, loaded from .env via pydantic-settings."""
from __future__ import annotations
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # --- AWS Bedrock (primary LLM provider) ---
    aws_region: str | None = None
    aws_access_key_id: str | None = None
    aws_secret_access_key: str | None = None
    aws_session_token: str | None = None             # only for STS / temporary credentials
    # Workshop accounts don't have Anthropic Claude on Bedrock — defaults are
    # Qwen3 (strong open-weight models with reliable tool calling). Override in
    # .env if needed. The chat UI dropdown also lets users pick per-request.
    bedrock_model_sonnet: str = "qwen.qwen3-235b-a22b-2507-v1:0"   # agent reasoning
    bedrock_model_haiku: str = "qwen.qwen3-32b-v1:0"               # supervisor routing

    # --- Anthropic direct API (failsafe when Bedrock errors) ---
    anthropic_api_key: str | None = None
    anthropic_model_sonnet: str = "claude-sonnet-4-5"
    anthropic_model_haiku: str = "claude-haiku-4-5"

    # --- DB ---
    database_url: str = "sqlite:///./grabwise.db"

    # --- Misc ---
    log_level: str = "INFO"

    @property
    def use_bedrock(self) -> bool:
        """True if Bedrock has enough credentials to attempt a call.

        Session token is optional — only needed for STS / temporary creds.
        Long-term IAM users have access key + secret without a session token.
        """
        return bool(self.aws_access_key_id and self.aws_secret_access_key and self.aws_region)

    @property
    def has_anthropic_fallback(self) -> bool:
        return bool(self.anthropic_api_key)


settings = Settings()
