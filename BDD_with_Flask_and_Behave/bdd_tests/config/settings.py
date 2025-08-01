# bdd_tests/config/settings.py

import os
from pydantic import BaseSettings
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.engine.url import URL

# This is the key change: dynamically determine which .env file to load
active_env = os.getenv("TEST_ENV", "qa").lower()
env_file_path = Path(__file__).resolve().parent.parent / f".env.{active_env}"

# Ensure the selected .env file exists before trying to load it
if not env_file_path.exists():
    raise FileNotFoundError(f"Environment file not found for '{active_env}' environment: {env_file_path}")

# Load the determined .env file
load_dotenv(dotenv_path=env_file_path, override=True)
print(f"✅ Loading environment settings from: {env_file_path}")

class Settings(BaseSettings):
    """
    Defines and validates all application settings, loading values from the active .env file.
    """
    # API Settings
    API_BASE_URL: str
    CERT_PFX_PATH: str
    CERT_PASSWORD: str

    # Database Settings
    DB_HOST: str
    DB_PORT: int = 5432
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    # Framework Settings
    LOG_LEVEL: str = "INFO"

    # Derived paths
    CERT_PEM_PATH: Optional[str] = None
    KEY_PEM_PATH: Optional[str] = None

    @property
    def DATABASE_URL(self) -> str:
        """Constructs the SQLAlchemy database connection URL from individual settings."""
        return str(URL.create(
            drivername="postgresql+psycopg2",
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.DB_NAME,
        ))

# Create a singleton instance for other modules to import easily
settings = Settings()
