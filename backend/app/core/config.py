from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "deepkin-dubbing-backend"
    api_prefix: str = "/api"
    app_version: str = "0.1.0"
    app_description: str = "Async video dubbing backend using transcript translation and DeepKIN TTS."

    # OpenAPI / Swagger UI
    openapi_url: str = "/openapi.json"
    docs_url: str = "/swagger"
    redoc_url: str | None = None

    # Limits
    max_upload_size_bytes: int = 10 * 1024 * 1024
    max_video_duration_seconds: int = 180
    allowed_extensions: str = "mp4,mov,m4v,webm"

    # API gateway + storage
    generate_signed_url_endpoint: str = "https://tfo7lcvae5.execute-api.us-east-1.amazonaws.com/prod/generate-upload-url"
    input_bucket: str = "cmu-aisd-input"
    output_bucket: str = "cmu-aisd-output"
    artifact_bucket: str = "cmu-aisd-artifacts"
    public_prefix: str = "public"
    public_artifact_store: str = "https://cmu-aisd-artifacts.s3.us-east-1.amazonaws.com"

    # Pipeline timeouts
    transcript_poll_interval_seconds: int = 5
    transcript_timeout_seconds: int = 600

    # Models
    translation_model_name: str = "facebook/nllb-200-distilled-600M"
    tts_model_path: str = "./kinya_flex_tts_base_trained.pt"
    speaker_id: int = 1

    # Runtime paths
    base_dir: Path = Path(__file__).resolve().parents[2]
    jobs_db_path: Path = base_dir / "data" / "jobs.db"
    temp_dir: Path = base_dir / "data" / "tmp"


settings = Settings()
