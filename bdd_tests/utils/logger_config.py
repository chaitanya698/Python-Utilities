import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Optional

class LoggerSetup:
    """Centralized logger configuration for the framework."""

    _initialized = False
    _log_dir = Path("logs")

    @classmethod
    def setup(cls, log_level: str = "INFO", log_file: Optional[str] = None):
        """Configure logging for the entire application."""
        if cls._initialized:
            return
            
        # Create logs directory
        cls._log_dir.mkdir(exist_ok=True)
        
        # Get root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] [%(name)-20s] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        if log_file is None:
            log_file = cls._log_dir / f"automation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)  # File gets all logs
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] [%(name)-30s:%(lineno)-4d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S.%f"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Suppress noisy loggers
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)
        
        cls._initialized = True
        root_logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")
    def get_logger(name: str) -> logging.Logger:
        """Get a logger instance for a module."""
        return logging.getLogger(name)

    # Initialize on import with defaults
    
LoggerSetup.setup(log_level=os.getenv("LOG_LEVEL", "INFO"))

