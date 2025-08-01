# bdd_tests/config/settings.py

import os
from pydantic import BaseSettings
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.engine.url import URL

# --- Step 1: Determine the Active Environment ---
# Read the 'TEST_ENV' environment variable, which is set by conftest.py from the --env command-line arg.
# If the variable is not set (e.g., when running without --env), it defaults to 'qa'.
active_env = os.getenv("TEST_ENV", "qa").lower()

# --- Step 2: Construct the Path to the Correct .env File ---
# Dynamically build the full path to the environment file (e.g., /path/to/project/bdd_tests/.env.dev)
env_file_path = Path(__file__).resolve().parent.parent / f".env.{active_env}"

# --- Step 3: Load the Environment File ---
# Check if the file exists to provide a clear error message if it doesn't.
if not env_file_path.exists():
    raise FileNotFoundError(f"Configuration Error: Environment file not found for '{active_env}' environment. Expected at: {env_file_path}")

# Use python-dotenv to load the key-value pairs from the selected .env file
# into the operating system's environment variables. This makes them available to Pydantic.
load_dotenv(dotenv_path=env_file_path, override=True)
print(f"✅ Loading environment settings from: {env_file_path}")

# --- Step 4: Define and Populate the Settings Model ---
class Settings(BaseSettings):
    """
    This class defines all the settings your application needs.
    Pydantic automatically reads from the environment variables (which were just loaded by load_dotenv)
    and populates the attributes of this class. It also performs type validation.
    """
    # API Settings - Pydantic will find API_BASE_URL, etc., in the environment.
    API_BASE_URL: str
    CERT_PFX_PATH: str
    CERT_PASSWORD: str

    # Database Settings - Pydantic will find DB_HOST, etc., in the environment.
    DB_HOST: str
    DB_PORT: int = 5432
    DB_USER: str
    DB_PASSWORD: str
    DB_NAME: str

    # Framework Settings
    LOG_LEVEL: str = "INFO"

    # Derived paths (not loaded from .env, will be set later by the loader)
    CERT_PEM_PATH: Optional[str] = None
    KEY_PEM_PATH: Optional[str] = None

    @property
    def DATABASE_URL(self) -> str:
        """Constructs the SQLAlchemy database connection URL from the loaded settings."""
        return str(URL.create(
            drivername="postgresql+psycopg2",
            username=self.DB_USER,
            password=self.DB_PASSWORD,
            host=self.DB_HOST,
            port=self.DB_PORT,
            database=self.DB_NAME,
        ))

# --- Step 5: Create a Singleton Instance ---
# Create a single, globally accessible instance of the populated Settings class.
# Other modules in the framework will import this 'settings' object.
settings = Settings()
