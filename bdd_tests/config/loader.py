import os
import sys
import tempfile
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import pkcs12, Encoding, PrivateFormat, NoEncryption
from cryptography.hazmat.backends import default_backend

from .settings import Settings
from .config_validator import ConfigValidator
from core.utils.logger import get_logger

class ConfigLoader:

    def __init__(self):
        self.logger = get_logger(__name__)
        self._settings: Optional[Settings] = None
        
    def load(self) -> Settings:
        """Load configuration with proper precedence: ENV vars > .env file > defaults."""
        if self._settings:
            return self._settings
            
        # 1. Determine environment
        environment = os.getenv("ENVIRONMENT", "development").lower()
        self.logger.info(f"Loading configuration for environment: {environment}")
        
        # 2. Load .env file for the environment
        self._load_env_file(environment)
        
        # 3. Validate required environment variables
        ConfigValidator.validate_required_vars()
        
        # 4. Create settings instance
        self._settings = Settings()
        
        # 5. Process certificate if needed
        self._process_certificate()
        
        # 6. Log configuration (without sensitive data)
        self._log_configuration()
        
        return self._settings

    def _load_env_file(self, environment: str):
        """Load environment-specific .env file."""
        env_file = Path(f".env.{environment}")
        
        if not env_file.exists():
            self.logger.warning(f"Environment file {env_file} not found. Using .env.example as fallback.")
            env_file = Path(".env.example")
        
        if env_file.exists():
            load_dotenv(dotenv_path=env_file, override=False)
            self.logger.info(f"Loaded environment file: {env_file}")
        else:
            self.logger.error("No environment file found. Relying on system environment variables.")

    def _process_certificate(self):
        """Extract PEM files from PFX certificate."""
        if not self._settings.CERT_PFX_PATH or not os.path.exists(self._settings.CERT_PFX_PATH):
            raise ValueError(f"Certificate file not found: {self._settings.CERT_PFX_PATH}")
        
        self.logger.info("Processing PFX certificate...")
        
        try:
            with open(self._settings.CERT_PFX_PATH, 'rb') as pfx_file:
                pfx_data = pfx_file.read()
            
            private_key, certificate, _ = pkcs12.load_key_and_certificates(
                pfx_data,
                self._settings.CERT_PASSWORD.encode(),
                default_backend()
            )
            
            # Create temporary PEM files
            key_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pem', mode='w+b')
            self._settings.KEY_PEM_PATH = key_file.name
            key_file.write(
                private_key.private_bytes(
                    encoding=Encoding.PEM,
                    format=PrivateFormat.TraditionalOpenSSL,
                    encryption_algorithm=NoEncryption()
                )
            )
            key_file.close()
            
            cert_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pem', mode='w+b')
            self._settings.CERT_PEM_PATH = cert_file.name
            cert_file.write(certificate.public_bytes(Encoding.PEM))
            cert_file.close()
            
            self.logger.info("Certificate processed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to process certificate: {e}")
            raise

    def _log_configuration(self):
        """Log configuration without sensitive information."""
        safe_config = {
            "environment": self._settings.ENVIRONMENT,
            "api_base_url": self._settings.API_BASE_URL,
            "api_timeout": self._settings.API_TIMEOUT,
            "db_host": self._settings.DB_HOST,
            "db_port": self._settings.DB_PORT,
            "log_level": self._settings.LOG_LEVEL,
            "ssl_verification": self._settings.VERIFY_SSL,
            "feature_flags": {
                "detailed_logging": self._settings.ENABLE_DETAILED_LOGGING,
                "db_query_logging": self._settings.ENABLE_DB_QUERY_LOGGING
            }
        }
        self.logger.info(f"Configuration loaded: {safe_config}")

    def cleanup(self):
        """Clean up temporary files."""
        if self._settings:
            for path in [self._settings.CERT_PEM_PATH, self._settings.KEY_PEM_PATH]:
                if path and os.path.exists(path):
                    try:
                        os.remove(path)
                        self.logger.debug(f"Removed temporary file: {path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to remove temporary file {path}: {e}")
#Singleton instance
_config_loader = ConfigLoader()

def get_config() -> Settings:

    return _config_loader.load()

def cleanup_config():
    _config_loader.cleanup()

