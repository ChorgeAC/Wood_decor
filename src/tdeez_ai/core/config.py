"""Environment-driven configuration. Nothing here is hardcoded to an AWS account."""

from functools import lru_cache
from typing import Literal, Self

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_REAL_AWS_REQUIRED: tuple[str, ...] = (
    "s3_bucket",
    "creations_table",
    "slabs_table",
    "profiles_table",
    "quotes_table",
    "opensearch_endpoint",
    "ses_sender",
    "quote_inbox",
    "cognito_user_pool_id",
    "cognito_app_client_id",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    use_mock: bool = True
    app_env: str = "dev"
    mock_public_base_url: str = "http://localhost:8000"
    cors_allow_origins: list[str] = Field(default_factory=list)
    seed_on_startup: bool = True

    aws_region: str = "us-east-1"

    # Bedrock models
    canvas_model_id: str = "amazon.nova-canvas-v1:0"
    vision_model_id: str = "amazon.nova-pro-v1:0"
    embedding_model_id: str = "amazon.titan-embed-image-v1"
    embedding_dimensions: Literal[256, 384, 1024] = 1024
    bedrock_read_timeout_s: float = Field(120.0, gt=0)
    bedrock_connect_timeout_s: float = Field(5.0, gt=0)
    bedrock_max_retries: int = Field(3, ge=0, le=5)
    bedrock_backoff_base_s: float = Field(0.5, ge=0)

    # Cost guardrails (Converse limits a single image to 3.75 MB)
    max_images_per_request: int = Field(3, ge=1, le=5)
    max_image_bytes: int = Field(3_750_000, gt=0)

    # Business rules
    species_review_threshold: float = Field(0.6, ge=0, le=1)
    room_min_confidence: float = Field(0.5, ge=0, le=1)
    pricing_rules_provided: bool = False
    pricing_rules_path: str | None = None

    # URL lifetimes (seconds)
    upload_url_expiry_s: int = Field(300, gt=0, le=900)
    read_url_expiry_s: int = Field(300, gt=0, le=900)

    # Real AWS resources
    s3_bucket: str | None = None
    creations_table: str | None = None
    slabs_table: str | None = None
    profiles_table: str | None = None
    quotes_table: str | None = None
    opensearch_endpoint: str | None = None
    opensearch_index: str = "slab-embeddings"
    ses_sender: str | None = None
    quote_inbox: str | None = None
    cognito_user_pool_id: str | None = None
    cognito_app_client_id: str | None = None

    @model_validator(mode="after")
    def _check_real_aws(self) -> Self:
        if not self.use_mock:
            missing = [n.upper() for n in _REAL_AWS_REQUIRED if not getattr(self, n)]
            if missing:
                raise ValueError("Missing required settings: " + ", ".join(missing))
        if self.pricing_rules_provided and not self.pricing_rules_path:
            raise ValueError("PRICING_RULES_PATH is required when PRICING_RULES_PROVIDED=true")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
