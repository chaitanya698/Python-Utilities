import os
import tempfile
from pathlib import Path
from typing import Optional
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.serialization import pkcs12, Encoding, PrivateFormat, NoEncryption

from .settings import Settings
from ..utils.logger_config import get_logger


class ConfigLoader:
    """Handles configuration loading and certificate processing."""
    
    _instance: Optional['ConfigLoader'] = None
    _settings: Optional[Settings] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            self.logger = get_logger(__name__)
            self._temp_files = []
            self.initialized = True
    
    def load(self) -> Settings:
        """Load configuration from environment-specific .env file."""
        if self._settings:
            return self._settings
        
        env = os.getenv('ENVIRONMENT', 'qa')
        self.logger.info(f"Loading configuration for environment: {env}")
        
        # Load settings with automatic .env file handling
        try:
            self._settings = Settings()
            self.logger.info(f"Successfully loaded settings from .env.{env}")
        except Exception as e:
            self.logger.error(f"Failed to load settings: {e}")
            raise
        
        # Process certificate if path is provided
        if self._settings.CERT_PFX_PATH:
            self._process_certificate()
        
        # Log configuration (without sensitive data)
        self._log_configuration()
        
        return self._settings
    
    def _process_certificate(self) -> None:
        """Extract PEM files from PFX certificate."""
        if not self._settings or not self._settings.CERT_PFX_PATH:
            return
        
        cert_path = Path(self._settings.CERT_PFX_PATH)
        
        # Try to resolve path relative to project root if not absolute
        if not cert_path.is_absolute():
            project_root = Path(__file__).resolve().parent.parent.parent
            cert_path = project_root / cert_path
        
        if not cert_path.exists():
            self.logger.warning(f"Certificate file not found at: {cert_path}")
            return
        
        self.logger.info("Processing PFX certificate...")
        try:
            with open(cert_path, 'rb') as pfx_file:
                pfx_data = pfx_file.read()
            
            private_key, certificate, _ = pkcs12.load_key_and_certificates(
                pfx_data,
                self._settings.CERT_PRD.encode() if self._settings.CERT_PRD else b'',
                default_backend()
            )
            
            # Create temporary PEM files
            key_file = tempfile.NamedTemporaryFile(delete=False, suffix='_key.pem', mode='w+b')
            key_data = private_key.private_bytes(
                encoding=Encoding.PEM,
                format=PrivateFormat.TraditionalOpenSSL,
                encryption_algorithm=NoEncryption()
            )
            key_file.write(key_data)
            key_file.close()
            self._settings.KEY_PEM_PATH = key_file.name
            self._temp_files.append(key_file.name)
            
            cert_file = tempfile.NamedTemporaryFile(delete=False, suffix='_cert.pem', mode='w+b')
            cert_file.write(certificate.public_bytes(Encoding.PEM))
            cert_file.close()
            self._settings.CERT_PEM_PATH = cert_file.name
            self._temp_files.append(cert_file.name)
            
            self.logger.info("Certificate processed successfully")
        except Exception as e:
            self.logger.error(f"Failed to process certificate: {e}")
    
    def _log_configuration(self) -> None:
        """Log configuration without sensitive information."""
        if self._settings:
            safe_config = {
                'ENVIRONMENT': self._settings.ENVIRONMENT,
                'API_BASE_URL': self._settings.API_BASE_URL,
                'API_TIMEOUT': self._settings.API_TIMEOUT,
                'DB_HOST': self._settings.DB_HOST,
                'DB_PORT': self._settings.DB_PORT,
                'DB_SERVICE_NAME': self._settings.DB_SERVICE_NAME,
                'LOG_LEVEL': self._settings.LOG_LEVEL,
                'VERIFY_SSL': self._settings.VERIFY_SSL
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


# Singleton accessor functions
def get_config() -> Settings:
    """Get the configuration settings."""
    return ConfigLoader().load()


def cleanup_config() -> None:
    """Clean up configuration resources."""
    ConfigLoader().cleanup()