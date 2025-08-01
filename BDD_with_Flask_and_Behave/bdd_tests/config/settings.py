# bdd_tests/config/settings.py

import os
from pydantic import BaseSettings
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv

# Dynamically determine which .env file to load based on the TEST_ENV variable
active_env = os.getenv("TEST_ENV", "qa").lower()
env_file_path = Path(__file__).resolve().parent.parent / f".env.{active_env}"

# Load the .env file if it exists
if env_file_path.exists():
    load_dotenv(dotenv_path=env_file_path, override=True)
    print(f"✅ Loading environment settings from: {env_file_path}")
else:
    print(f"⚠️ Note: Environment file not found at {env_file_path}. Relying on OS environment variables.")

class Settings(BaseSettings):
    """
    This class defines all application settings. Pydantic automatically reads values
    from environment variables (loaded from the .env file or set by the system/command line).
    """
    # API Settings
    API_BASE_URL: str
    CERT_PFX_PATH: str
    CERT_PASSWORD: str

    # Oracle Database Settings
    DB_HOST: str
    DB_PORT: int = 1521
    DB_USER: str
    DB_PASSWORD: str
    DB_SERVICE_NAME: str

    # Framework Settings
    LOG_LEVEL: str = "INFO"

    # Derived paths (set at runtime by the loader)
    CERT_PEM_PATH: Optional[str] = None
    KEY_PEM_PATH: Optional[str] = None

# Create a singleton instance for other modules to import
settings = Settings()
