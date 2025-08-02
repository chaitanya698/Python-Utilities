# config/settings.py

from pydantic import BaseModel
from typing import Optional

class Settings(BaseModel):
    """
    Defines the data structure for all application settings.
    This model is instantiated and populated by the config loader, which reads
    values from environment variables.
    """
    # API Settings
    API_BASE_URL: str
    CERT_PFX_PATH: Optional[str] = None
    CERT_PASSWORD: Optional[str] = None

    # Oracle Database Settings
    DB_HOST: str
    DB_PORT: int = 1521
    DB_USER: str
    DB_PASSWORD: str
    DB_SERVICE_NAME: str

    # Framework Settings
    LOG_LEVEL: str = "INFO"

    # Derived paths (these will be populated at runtime by the loader)
    CERT_PEM_PATH: Optional[str] = None
    KEY_PEM_PATH: Optional[str] = None

    class Config:
        # This tells Pydantic to be case-insensitive when reading env vars
        case_sensitive = False
