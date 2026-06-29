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

    # Cloud overrides (set in prod). When present these take precedence over the
    # host/port fields above so the same code runs locally and on Neon/Qdrant Cloud.
    database_url: str = ""        # postgresql://USER:PASS@HOST/db?sslmode=require (Neon)
    qdrant_url: str = ""          # https://xxxx.cloud.qdrant.io:6333
    qdrant_api_key: str = ""
    media_base_url: str = ""      # signed object-storage base for AIDE-derived clip media

    siglip_model: str = "google/siglip2-base-patch16-256"
    siglip_dim: int = 768

    anthropic_api_key: str = ""

    aide_fps: float = 15.0
    aide_frames_per_clip: int = 45
    aide_clip_duration_s: float = 3.0


    class Config:
        env_file = ".env"

settings = Settings()