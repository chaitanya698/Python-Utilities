import os
import tempfile
import logging
from cryptography.hazmat.primitives.serialization import pkcs12, Encoding, PrivateFormat, NoEncryption
from cryptography.hazmat.backends import default_backend
from .settings import settings

logger = logging.getLogger(__name__)

def load_and_process_config():
    """
    Processes the loaded settings, specifically handling the secure PFX to PEM conversion.
    This function is called once when the application starts.
    """
    # Conditionally process the certificate only if the URL requires it
    if settings.API_BASE_URL and "wellsfargo.net" in settings.API_BASE_URL:
        _process_pfx_certificate()
    return settings

def _process_pfx_certificate():
    """
    Securely extracts key and cert from a PFX file into temporary, automatically
    cleaned-up PEM files.
    """
    if not settings.CERT_PFX_PATH or not os.path.exists(settings.CERT_PFX_PATH):
        raise FileNotFoundError(f"Certificate file not found at path: {settings.CERT_PFX_PATH}")
    if not settings.CERT_PASSWORD:
        raise ValueError("Certificate password (CERT_PASSWORD) is required but not set in .env file.")

    logger.info(f"Extracting certificate from: {settings.CERT_PFX_PATH}")
    try:
        with open(settings.CERT_PFX_PATH, 'rb') as pfx_file:
            pfx_data = pfx_file.read()
        
        private_key, certificate, _ = pkcs12.load_key_and_certificates(
            pfx_data,
            settings.CERT_PASSWORD.encode(),
            default_backend()
        )

        # Create a temporary file for the PEM-encoded private key.
        key_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pem', mode='w+b')
        settings.KEY_PEM_PATH = key_file.name
        key_file.write(private_key.private_bytes(
            encoding=Encoding.PEM,
            format=PrivateFormat.TraditionalOpenSSL, # Common format for requests lib
            encryption_algorithm=NoEncryption()
        ))
        key_file.close()

        # Create a temporary file for the PEM-encoded certificate.
        cert_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pem', mode='w+b')
        settings.CERT_PEM_PATH = cert_file.name
        cert_file.write(certificate.public_bytes(Encoding.PEM))
        cert_file.close()

        logger.info("Certificate extracted successfully to temporary files.")

    except ValueError:
        logger.error("Invalid certificate password. Please check CERT_PASSWORD in your .env file.")
        raise
    except Exception as e:
        logger.error(f"Failed to process PFX certificate: {e}")
        raise

# Load and process the configuration once when the module is imported
config = load_and_process_config()