import os
from pathlib import Path
from typing import List, Optional
import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Core metadata
    PROJECT_NAME: str = "AI Vision Training Studio"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_V1_STR: str = "/api/v1"

    # Server binding
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:5173",
        "*"
    ]

    # Database
    DATABASE_URL: str = f"sqlite+aiosqlite:///{BASE_DIR / 'data' / 'studio.db'}"

    # Storage Paths
    BASE_STORAGE_DIR: Path = BASE_DIR / "data"
    UPLOAD_DIR: Path = BASE_DIR / "data" / "uploads"
    DATASET_DIR: Path = BASE_DIR / "data" / "datasets"
    RUNS_DIR: Path = BASE_DIR / "data" / "runs"
    MODELS_DIR: Path = BASE_DIR / "data" / "models"
    EXPORT_DIR: Path = BASE_DIR / "data" / "exports"

    # Security & Upload Restrictions
    MAX_UPLOAD_SIZE_MB: int = 500
    ALLOWED_IMAGE_EXTENSIONS: List[str] = [
        "jpg", "jpeg", "png", "webp", "bmp", "tiff", "tif"
    ]
    SECRET_KEY: str = "super-secret-key-change-in-production-ai-vision-studio"

    # Hardware & Training defaults
    DEVICE: str = "auto"
    MAX_CONCURRENT_JOBS: int = 1
    WORKER_POLL_INTERVAL_SEC: float = 1.0

    def init_directories(self) -> None:
        """Ensure all required runtime directories exist."""
        for path in [
            self.BASE_STORAGE_DIR,
            self.UPLOAD_DIR,
            self.DATASET_DIR,
            self.RUNS_DIR,
            self.MODELS_DIR,
            self.EXPORT_DIR,
        ]:
            path.mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.init_directories()
