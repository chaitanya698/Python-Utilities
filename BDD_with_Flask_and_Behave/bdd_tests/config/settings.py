from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    """
    Defines and validates all application settings using Pydantic.
    Loads variables from .env files automatically.
    """
    # Core API settings
    API_BASE_URL: str
    LOG_LEVEL: str = "INFO"

    # Certificate settings (optional)
    CERT_PASSWORD: Optional[str] = None
    CERT_PFX_PATH: Optional[str] = None
    
    # These paths will be populated by the loader after secure certificate processing
    CERT_PEM_PATH: Optional[str] = None
    KEY_PEM_PATH: Optional[str] = None

    class Config:
        """Tells Pydantic to look for and load .env files."""
        env_file = '.env'
        # ('.env.qa', '.env.dev') # Order matters; it stops at the first one found
        env_file_encoding = 'utf-8'

# Create a single, validated settings instance to be used across the app
settings = Settings()