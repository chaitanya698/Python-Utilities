# bdd_tests/config/settings.py

from pydantic import BaseSettings
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.engine.url import URL # Import SQLAlchemy's URL builder

class Settings(BaseSettings):
    """
    Defines and validates all application settings, loading values from environment variables.
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
        """
        Constructs the SQLAlchemy database connection URL from individual settings.
        Example: postgresql+psycopg2://user:password@host:port/dbname
        """
        return str(URL.create(
            drivername="postgresql+psycopg2", # Specify the DB dialect and driver
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.DB_NAME,
        ))

    class Config:
        """Pydantic configuration to load from a .env file."""
        env_file = Path(__file__).resolve().parent.parent / ".env.qa"
        env_file_encoding = "utf-8"

# Load the .env file before creating the Settings instance
load_dotenv(dotenv_path=Settings.Config.env_file, override=True)

# Create a singleton instance for other modules to import easily
settings = Settings()
