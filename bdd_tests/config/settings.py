from pydantic import BaseModel, Field, validator
from typing import Optional
import os


class Settings(BaseModel):
    """Application configuration settings using Pydantic for validation."""
    
    # Environment
    ENVIRONMENT: str = Field(default="qa")
    
    # API Configuration
    API_BASE_URL: str
    API_TIMEOUT: int = Field(default=45)
    API_RETRY_COUNT: int = Field(default=3)
    
    # Database Configuration
    DB_HOST: str
    DB_PORT: int = Field(default=1521)
    DB_USER: str
    DB_PASSWORD: str
    DB_SERVICE_NAME: str
    DB_POOL_SIZE: int = Field(default=10)
    DB_MAX_OVERFLOW: int = Field(default=20)
    
    # Certificate Configuration
    CERT_PFX_PATH: str
    CERT_PASSWORD: str
    CERT_PEM_PATH: Optional[str] = None
    KEY_PEM_PATH: Optional[str] = None
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(
        default="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s"
    )
    LOG_FILE_PATH: Optional[str] = None
    
    # Feature Flags
    ENABLE_DETAILED_LOGGING: bool = Field(default=False)
    ENABLE_DB_QUERY_LOGGING: bool = Field(default=False)
    VERIFY_SSL: bool = Field(default=True)
    
    class Config:
        case_sensitive = False
        env_file = ".env"
        env_file_encoding = "utf-8"
    
    @validator("ENVIRONMENT")
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