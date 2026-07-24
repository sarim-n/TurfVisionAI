"""
Purpose: Application configuration management using Pydantic BaseSettings.
Dependencies: pydantic-settings, pydantic
Inputs: Environment variables or .env file
Outputs: Global application Settings instance
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """System-wide configuration settings loaded from environment or .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Core System Info
    APP_NAME: str = "TurfVision AI"
    ENVIRONMENT: Literal["development", "testing", "production"] = "development"
    LOG_LEVEL: str = "INFO"

    # API Server
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    SECRET_KEY: str = "turfvision_dev_secret_key_change_in_production"

    # Database Settings
    POSTGRES_USER: str = "turfvision"
    POSTGRES_PASSWORD: str = "turfvision_pass"
    POSTGRES_DB: str = "turfvision_db"
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    DATABASE_URL: str = (
        "postgresql+asyncpg://turfvision:turfvision_pass@localhost:5432/turfvision_db"
    )

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379

    # Vision & ML Settings
    YOLO_MODEL_PATH: str = "models/yolov8x.pt"
    BALL_YOLO_MODEL_PATH: str = "models/football_yolo.pt"
    CONFIDENCE_THRESHOLD: float = 0.35
    IOU_THRESHOLD: float = 0.45
    DEVICE: str = "cpu"

    # Video Pipeline Targets
    DEFAULT_TARGET_FPS: int = 30
    FRAME_QUEUE_MAX_SIZE: int = 100


@lru_cache
def get_settings() -> Settings:
    """Returns a cached singleton Settings instance."""
    return Settings()
