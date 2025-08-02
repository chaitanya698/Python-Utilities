# config/loader.py

import os
import tempfile
import logging
from pathlib import Path
from dotenv import load_dotenv
from cryptography.hazmat.primitives.serialization import pkcs12, Encoding, PrivateFormat, NoEncryption
from cryptography.hazmat.backends import default_backend

# Import the settings model (schema) from settings.py
from .settings import Settings

logger = logging.getLogger(__name__)

def _load_environment_variables():
    """
    Determines which .env file to load based on the TEST_ENV environment
    variable and loads it.
    """
    active_env = os.getenv("TEST_ENV", "qa").lower()
    # Assuming the .env files are in the root of the 'bdd_tests' directory
    env_file_path = Path(__file__).resolve().parent.parent / f".env.{active_env}"

    if env_file_path.exists():
        load_dotenv(dotenv_path=env_file_path, override=True)
        logger.info(f"Successfully loaded environment settings from: {env_file_path}")
    else:
        logger.warning(f"Environment file not found at {env_file_path}. Relying on OS environment variables.")

def _process_pfx_certificate(settings_instance: Settings):
    """
    Securely extracts key and cert from a PFX file into temporary PEM files.
    This function modifies the settings instance passed to it.
    """
    if not settings_instance.CERT_PFX_PATH or not os.path.exists(settings_instance.CERT_PFX_PATH):
        logger.warning(f"Certificate file not found at path: {settings_instance.CERT_PFX_PATH}. Skipping PFX processing.")
        return
    if not settings_instance.CERT_PASSWORD:
        raise ValueError("Certificate password (CERT_PASSWORD) is required but not set.")

    logger.info(f"Extracting certificate from: {settings_instance.CERT_PFX_PATH}")
    try:
        with open(settings_instance.CERT_PFX_PATH, 'rb') as pfx_file:
            pfx_data = pfx_file.read()
        
        private_key, certificate, _ = pkcs12.load_key_and_certificates(
            pfx_data,
            settings_instance.CERT_PASSWORD.encode(),
            default_backend()
        )

        # Create a temporary file for the PEM-encoded private key
        key_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pem', mode='w+b')
        settings_instance.KEY_PEM_PATH = key_file.name
        key_file.write(private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=NoEncryption()
        ))
        key_file.close()

        # Create a temporary file for the PEM-encoded certificate
        cert_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pem', mode='w+b')
        settings_instance.CERT_PEM_PATH = cert_file.name
        cert_file.write(certificate.public_bytes(Encoding.PEM))
        cert_file.close()

        logger.info("Certificate extracted successfully to temporary PEM files.")

    except ValueError:
        logger.error("Invalid certificate password. Please check CERT_PASSWORD in your .env file.")
        raise
    except Exception as e:
        logger.error(f"Failed to process PFX certificate: {e}")
        raise

def load_and_get_config() -> Settings:
    """
    The main function to load and process all configurations.
    This is the single entry point for the rest of the application.
    """
    # 1. Load variables from the correct .env file into the OS environment
    _load_environment_variables()
    
    # 2. Instantiate the Pydantic model, which automatically reads from the environment
    settings_instance = Settings()
    
    # 3. Perform any additional processing on the loaded settings, like certs
    if settings_instance.CERT_PFX_PATH:
        _process_pfx_certificate(settings_instance)
        
    return settings_instance

# Create a single, authoritative config instance to be imported by other modules.
# This ensures the loading and processing logic runs only once.
config = load_and_get_config()
