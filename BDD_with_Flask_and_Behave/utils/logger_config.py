import logging
import sys
from utils.config_loader import get_config

# Load configuration to get the log level
config = get_config()
LOG_LEVEL = config.get('LOG_LEVEL', 'INFO').upper()

# Prevent the root logger from being configured multiple times
if not logging.getLogger().handlers:
    # Configure the basic logging settings
    logging.basicConfig(
        level=LOG_LEVEL,
        stream=sys.stdout,  # Log messages to the console
        format="%(asctime)s [%(levelname)s] [%(name)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def get_logger(name: str):
    """
    Retrieves a logger instance with the specified name.

    Args:
        name (str): The name for the logger, typically __name__ of the calling module.

    Returns:
        logging.Logger: A configured logger instance.
    """
    return logging.getLogger(name)