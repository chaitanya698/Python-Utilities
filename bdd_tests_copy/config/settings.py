from pathlib import Path
import os
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Application configuration settings with automatic .env file loading."""
    
    # Environment
    ENVIRONMENT: str = Field(default="qa")
    
    # API Configuration
    API_BASE_URL: str = Field(...)
    API_TIMEOUT: int = Field(default=45)
    API_RETRY_COUNT: int = Field(default=3)
    
    # Database Configuration
    DB_HOST: str = Field(...)
    DB_PORT: int = Field(default=3203)
    DB_USER: str = Field(...)
    DB_PRD: str = Field(alias="DB_PRD")  # Using alias for password field
    DB_SERVICE_NAME: str = Field(...)
    DB_POOL_SIZE: int = Field(default=10)
    DB_MAX_OVERFLOW: int = Field(default=20)
    
    # Certificate Configuration
    CERT_PFX_PATH: str = Field(...)
    CERT_PRD: str = Field(alias="CERT_PRD")
    CERT_PEM_PATH: Optional[str] = Field(default=None)
    KEY_PEM_PATH: Optional[str] = Field(default=None)
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FILE_PATH: Optional[str] = Field(default=None)
    ENABLE_DETAILED_LOGGING: bool = Field(default=False)
    ENABLE_DB_QUERY_LOGGING: bool = Field(default=False)
    VERIFY_SSL: bool = Field(default=True)
    
    model_config = SettingsConfigDict(
        env_file=f".env.{os.getenv('ENVIRONMENT', 'qa')}",
        env_file_encoding='utf-8',
        case_sensitive=False,
        extra='ignore',
        populate_by_name=True  # Allows using field names or aliases
    )
    
    @field_validator("ENVIRONMENT")
    @classmethod
    def validate_environment(cls, v: str) -> str:
        """Ensure environment is valid."""
        valid_envs = ["development", "qa", "staging", "production", "dev"]
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {', '.join(valid_envs)}")
        return v.lower()
    
    @property
    def DB_CONNECTION_STRING(self) -> str:
        """Build Oracle connection string for SQLAlchemy."""
        return (
            f"oracle+oracledb://{self.DB_USER}:{self.DB_PRD}@"
            f"{self.DB_HOST}:{self.DB_PORT}/?service_name={self.DB_SERVICE_NAME}"
        )
    
    @property
    def DB_DSN(self) -> str:
        """Build Oracle DSN for direct oracledb connections."""
        return f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_SERVICE_NAME}"