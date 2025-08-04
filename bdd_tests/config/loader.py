import os
import json
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import (
    pkcs12, Encoding, PrivateFormat, NoEncryption
)
from cryptography.hazmat.backends import default_backend

from .settings import Settings
from .config_validator import ConfigValidator
from utils.logger_config import get_logger


class ConfigLoader:
    """Handles configuration loading and certificate processing."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._settings: Optional[Settings] = None
        self._temp_files: list[str] = []
    
    def load(self) -> Settings:
        """Load configuration with proper precedence."""
        if self._settings:
            return self._settings
        
        # Determine environment
        environment = os.getenv("ENVIRONMENT", "qa").lower()
        self.logger.info(f"Loading configuration for environment: {environment}")
        
        # Load environment-specific configuration
        self._load_env_file(environment)
        self._load_json_config(environment)
        
        # Validate required environment variables
        ConfigValidator.validate_required_vars()
        
        # Create settings instance
        self._settings = Settings()
        
        # Process certificate if needed
        self._process_certificate()
        
        # Log configuration (without sensitive data)
        self._log_configuration()
        
        return self._settings
    
    def _load_env_file(self, environment: str) -> None:
        """Load environment-specific .env file."""
        env_files = [
            Path(f".env.{environment}"),
            Path(".env"),
        ]
        
        for env_file in env_files:
            if env_file.exists():
                load_dotenv(dotenv_path=env_file, override=False)
                self.logger.info(f"Loaded environment file: {env_file}")
                return
        
        self.logger.warning("No environment file found. Using system environment variables.")
    
    def _load_json_config(self, environment: str) -> None:
        """Load JSON configuration if exists."""
        config_file = Path(f"config/{environment}.json")
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Set environment variables from JSON config
            for key, value in config_data.items():
                if not os.getenv(key):
                    os.environ[key] = str(value)
            
            self.logger.info(f"Loaded JSON config: {config_file}")
    
    def _process_certificate(self) -> None:
        """Extract PEM files from PFX certificate."""
        if not self._settings.CERT_PFX_PATH:
            raise ValueError("Certificate path not configured")
        
        if not Path(self._settings.CERT_PFX_PATH).exists():
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
            key_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='_key.pem', mode='w+b'
            )
            key_data = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=NoEncryption()
            )
            key_file.write(key_data)
            key_file.close()
            self._settings.KEY_PEM_PATH = key_file.name
            self._temp_files.append(key_file.name)
            
            cert_file = tempfile.NamedTemporaryFile(
                delete=False, suffix='_cert.pem', mode='w+b'
            )
            cert_file.write(certificate.public_bytes(Encoding.PEM))
            cert_file.close()
            self._settings.CERT_PEM_PATH = cert_file.name
            self._temp_files.append(cert_file.name)
            
            self.logger.info("Certificate processed successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to process certificate: {e}")
            raise
    
    def _log_configuration(self) -> None:
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
    
    def cleanup(self) -> None:
        """Clean up temporary files."""
        for temp_file in self._temp_files:
            if Path(temp_file).exists():
                try:
                    os.remove(temp_file)
                    self.logger.debug(f"Removed temporary file: {temp_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to remove temporary file {temp_file}: {e}")
        self._temp_files.clear()


# Singleton instance
_config_loader = ConfigLoader()


def get_config() -> Settings:
    """Get the configuration settings."""
    return _config_loader.load()


def cleanup_config() -> None:
    """Clean up configuration resources."""
    _config_loader.cleanup()