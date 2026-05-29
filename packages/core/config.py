import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_user: str = "v2t"
    postgres_password: str = "v2t_dev_password"
    postgres_db: str = "v2t"

    redis_host: str = "localhost"
    redis_port: int = 6379

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    siglip_model: str = "google/siglip2-base-patch16-256"
    siglip_dim: int = 768

    anthropic_api_key: str = ""

    class Config:
        env_file = ".env"

settings = Settings()