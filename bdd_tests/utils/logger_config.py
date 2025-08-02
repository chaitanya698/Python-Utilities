# bdd_tests/utils/logger_config.py

import logging
import sys
from bdd_tests.config.settings import settings

def setup_logging():
    """
    Configures logging for the entire application based on the level in settings.
    Ensures it only runs once.
    """
    # Get the root logger
    root_logger = logging.getLogger()
    
    # Avoid adding handlers if they already exist
    if not root_logger.handlers:
        log_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
        root_logger.setLevel(log_level)
        
        # Create a handler to print to console
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        
        # Create a formatter and add it to the handler
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        
        # Add the handler to the root logger
        root_logger.addHandler(handler)
        
        logging.info(f"Logging configured with level: {settings.LOG_LEVEL}")

# Run setup when the module is imported
setup_logging()
