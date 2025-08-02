# config/config_loader.py

import os
import json
import logging
from pathlib import Path
from pydantic import BaseModel, Field
from typing import Optional

logger = logging.getLogger(__name__)

class DBSettings(BaseModel):
    """Pydantic model for database settings, loaded from environment variables."""
    host: str = Field(..., env="DB_HOST")
    port: int = Field(..., env="DB_PORT")
    user: str = Field(..., env="DB_USER")
    password: str = Field(..., env="DB_PASSWORD")
    service_name: str = Field(..., env="DB_SERVICE_NAME")

class AppSettings(BaseModel):
    """Pydantic model for application-level settings loaded from JSON config."""
    api_base_url: str
    log_level: str = "INFO"
    # Optional certificate settings, may not be in all configs
    cert_pfx_path: Optional[str] = None
    cert_password: Optional[str] = None

class Config:
    """A centralized class to hold all configuration."""
    def __init__(self):
        self.db = DBSettings()
        
        # Load app settings from the environment-specific JSON file
        env = os.getenv("TEST_ENV", "qa")
        config_path = Path(__file__).parent / f"{env}.json"
        
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file for environment '{env}' not found at {config_path}")
            
        logger.info(f"Loading application configuration from: {config_path}")
        with open(config_path) as f:
            app_data = json.load(f)
            self.app = AppSettings(**app_data)

# Create a singleton instance to be imported by other modules
try:
    config = Config()
    # Set the log level for the application based on the loaded config
    logging.getLogger().setLevel(config.app.log_level.upper())
except Exception as e:
    logger.critical(f"FATAL: Failed to load configuration. Error: {e}")
    # Re-raise the exception to halt execution if config fails to load
    raise

