from pydantic import BaseModel, Field, validator
from typing import Optional
import os

class Settings(BaseModel):
        
    # Environment
    ENVIRONMENT: str = Field(default="development")

    # API Configuration
    API_BASE_URL: str
    API_TIMEOUT: int = Field(default=45)
    API_RETRY_COUNT: int = Field(default=3)

    # Database Configuration (from environment only)
    DB_CONNECTION_STRING: Optional[str] = None
    DB_HOST: Optional[str] = None
    DB_PORT: Optional[int] = None
    DB_USER: str
    DB_PASSWORD: str
    DB_SERVICE_NAME: Optional[str] = None
    DB_POOL_SIZE: int = Field(default=10)
    DB_MAX_OVERFLOW: int = Field(default=20)

    # Certificate Configuration (from environment only)
    CERT_PFX_PATH: str
    CERT_PASSWORD: str
    CERT_PEM_PATH: Optional[str] = None
    KEY_PEM_PATH: Optional[str] = None

    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s")
    LOG_FILE_PATH: Optional[str] = None

    # Feature Flags
    ENABLE_DETAILED_LOGGING: bool = Field(default=False)
    ENABLE_DB_QUERY_LOGGING: bool = Field(default=False)
    VERIFY_SSL: bool = Field(default=True)

class Config:
    case_sensitive = False
        
    @validator('DB_CONNECTION_STRING', always=True)
    def build_connection_string(cls, v, values):
        """Build connection string if not provided directly."""
        if v:
            return v
        if all(k in values for k in ['DB_HOST', 'DB_PORT', 'DB_USER', 'DB_SERVICE_NAME']):
            return f"oracle+oracledb://{values['DB_USER']}:{values.get('DB_PASSWORD')}@{values['DB_HOST']}:{values['DB_PORT']}/{values['DB_SERVICE_NAME']}"
        return None

    @validator('ENVIRONMENT')
    def validate_environment(cls, v):
        """Ensure environment is valid."""
        valid_envs = ['development', 'qa', 'staging', 'production']
        if v.lower() not in valid_envs:
            raise ValueError(f"Environment must be one of: {', '.join(valid_envs)}")
        return v.lower()
