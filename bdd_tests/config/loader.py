import os
from dotenv import load_dotenv
from pathlib import Path
import logging

# Configure logger for the loader
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_settings(env: str = 'qa'):
    
    # Set the environment variable so that other parts of the app, including
    # the Pydantic settings class, know which environment is active.
    
    os.environ['ENV'] = env
    logger.info(f"Environment set to: {env}")

    # Construct the path to the .env file.
    # We assume .env files are in the project root directory (bdd_tests).
    project_root = Path(__file__).parent.parent 
    dotenv_path = project_root / f".env.{env}"

    if dotenv_path.exists():
        # `load_dotenv` will load the variables from the specified file into the environment.
        # The `override=True` flag ensures that values from the .env file will
        # overwrite any existing environment variables.
        load_dotenv(dotenv_path=dotenv_path, override=True)
        logger.info(f"Successfully loaded settings from: {dotenv_path}")
    else:
        logger.error(f"Environment file not found at: {dotenv_path}")
        raise FileNotFoundError(f"Could not find the environment file for '{env}' at {dotenv_path}")

