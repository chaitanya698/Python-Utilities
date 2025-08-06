import os
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import Optional

class Settings(BaseSettings):
    """
    Main settings class to hold all configuration variables.
    It reads from environment variables and .env files.
    """
    # API Settings
    API_BASE_URL: str
    CERT_PFX_PATH: str
    CERT_PFX_PASSWORD: str

    # Database Settings
    DB_USER: str
    DB_PASSWORD: str
    DB_HOST: str
    DB_PORT: int
    DB_SERVICE_NAME: str
    
    # Feature Flags
    ENABLE_DETAILED_LOGGING: bool = False

    class Config:
        """
        Pydantic config class.
        Specifies the .env file to be used. The loader.py will set the ENV
        environment variable which determines which file is loaded.
        """
        case_sensitive = False
        env_file = f".env.{os.getenv('ENV', 'qa')}"
        env_file_encoding = 'utf-8'

# Instantiate the settings. This object will be imported across the application.
# The `load_settings` function will ensure the correct .env file is loaded before this.
settings = Settings()
