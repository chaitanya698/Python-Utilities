import os
from pathlib import Path
import tempfile
from cryptography.hazmat.primitives.serialization import pkcs12, Encoding, PrivateFormat, NoEncryption
from cryptography.hazmat.backends import default_backend

from .settings import Settings
from utils.logger_config import get_logger

class ConfigLoader:
    """Handles configuration loading and certificate processing."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._settings: Optional[Settings] = None
        self._temp_files: list[str] = []
    
    def load(self) -> Settings:
        """Load configuration using Pydantic's built-in capabilities."""
        if self._settings:
            return self._settings
        
        # The environment should be set by pytest before this is called.
        self.logger.info(f"Loading configuration for environment: {os.getenv('ENVIRONMENT', 'qa')}")
        
        # Instantiate Settings. Pydantic now handles the .env file loading automatically.
        self._settings = Settings()
        
        # Process certificate if needed
        self._process_certificate()
        
        # Log configuration (without sensitive data)
        self._log_configuration()
        
        return self._settings
    
    # The _load_env_file and _load_json_config methods are no longer needed
    # and can be removed.
    
    def _process_certificate(self) -> None:
        """Extract PEM files from PFX certificate."""
        if not self._settings.CERT_PFX_PATH:
            raise ValueError("Certificate path not configured")
        
        if not Path(self._settings.CERT_PFX_PATH).exists():
            # Try to resolve path relative to project root
            project_root = Path(__file__).resolve().parent.parent.parent
            cert_path = project_root / self._settings.CERT_PFX_PATH
            if not cert_path.exists():
                raise ValueError(f"Certificate file not found at: {self._settings.CERT_PFX_PATH} or {cert_path}")
            self._settings.CERT_PFX_PATH = str(cert_path)

        self.logger.info("Processing PFX certificate...")
        
        try:
            with open(self._settings.CERT_PFX_PATH, 'rb') as pfx_file:
                pfx_data = pfx_file.read()
            
            private_key, certificate, _ = pkcs12.load_key_and_certificates(
                pfx_data,
                self._settings.CERT_PASSWORD.encode(),
                default_backend()
            )
            
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
            raise
    
    def _log_configuration(self) -> None:
        """Log configuration without sensitive information."""
        if self._settings:
            safe_config = self._settings.model_dump(exclude={'DB_PASSWORD', 'CERT_PASSWORD'})
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
