# bdd_tests/config/settings.py

import os
from pydantic import BaseSettings
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.engine.url import URL

# This part remains the same, it dynamically loads the correct .env file
active_env = os.getenv("TEST_ENV", "qa").lower()
env_file_path = Path(__file__).resolve().parent.parent / f".env.{active_env}"

if not env_file_path.exists():
    raise FileNotFoundError(f"Configuration Error: Environment file not found for '{active_env}' environment. Expected at: {env_file_path}")

load_dotenv(dotenv_path=env_file_path, override=True)
print(f"✅ Loading environment settings from: {env_file_path}")

class Settings(BaseSettings):
    """
    This class now defines settings for connecting to an Oracle database.
    """
    # API Settings
    API_BASE_URL: str
    CERT_PFX_PATH: str
    CERT_PASSWORD: str

    # --- Database Settings (Updated for Oracle) ---
    DB_HOST: str
    DB_PORT: int = 1521
    DB_USER: str
    DB_PASSWORD: str
    DB_SERVICE_NAME: str  # Use service name for Oracle

    # Framework Settings
    LOG_LEVEL: str = "INFO"

    # Derived paths
    CERT_PEM_PATH: Optional[str] = None
    KEY_PEM_PATH: Optional[str] = None

    @property
    def DATABASE_URL(self) -> str:
        """
        Constructs the SQLAlchemy URL for an Oracle database.
        The 'database' argument in URL.create corresponds to the service name.
        """
        return str(URL.create(
            drivername="oracle+cx_oracle",
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.DB_SERVICE_NAME,
        ))

# Create a singleton instance for other modules to import easily
settings = Settings()
