import logging
import sys
from config.loader import config

def setup_logging():
    """Configures logging for the entire application, ensuring it only runs once."""
    if not logging.getLogger().handlers: # Check if handlers are already configured
        logging.basicConfig(
            level=config.LOG_LEVEL.upper(),
            stream=sys.stdout,
            format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    return logging.getLogger

# Initialize logging and make the get_logger function available for import
get_logger = setup_logging()