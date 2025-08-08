from pathlib import Path
import os
from typing import Optional

from pydantic import BaseSettings, Field, field_validator
from pydantic_settings import SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
# Determine the environment from the OS, defaulting to 'qa'
ENVIRONMENT = os.getenv("ENVIRONMENT", "qa").lower()
ENV_FILE = PROJECT_ROOT / f".env.{ENVIRONMENT}"


class Settings(BaseSettings):
    """Application configuration settings that automatically load from env files."""
    # This config tells Pydantic where to find the .env file
    model_config = SettingsConfigDict(
        env_file=ENV_FILE if ENV_FILE.exists() else None,
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore'  # Prevents errors if .env file has extra variables
    )

    # Environment
    ENVIRONMENT: str = Field(default=ENVIRONMENT)

    # API Configuration
    API_BASE_URL: str
    API_TIMEOUT: int = 45
    API_RETRY_COUNT: int = 3

    # Database Configuration
    DB_HOST: str
    DB_PORT: int
    DB_USER: str
    DB_PRD: str
    DB_SERVICE_NAME: str
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20

    # Certificate Configuration
    CERT_PFX_PATH: str
    CERT_PRD: str
    CERT_PEM_PATH: Optional[str] = None
    KEY_PEM_PATH: Optional[str] = None

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    ENABLE_DETAILED_LOGGING: bool = False
    ENABLE_DB_QUERY_LOGGING: bool = False
    VERIFY_SSL: bool = True

    @field_validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Ensure environment is valid."""
        valid_envs = ["development", "qa", "staging", "production"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {', '.join(valid_envs)}")
        return v.lower()

    @property
    def DB_CONNECTION_STRING(self) -> str:
        """Build Oracle connection string."""
        return (
            f"oracle+oracledb://{self.DB_USER}:{self.DB_PASSWORD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_SERVICE_NAME}"
        )
