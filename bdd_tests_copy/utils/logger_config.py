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
    _loggers_configured = set()
    
    @classmethod
    def setup(cls, log_level: str = "INFO", log_file: Optional[str] = None) -> None:
        """Configure logging for the entire application."""
        if cls._initialized:
            return
        
        # Create logs directory
        cls._log_dir.mkdir(exist_ok=True)
        
        # Get root logger
        root_logger = logging.getLogger()
        
        # IMPORTANT: Clear all existing handlers to prevent double logging
        root_logger.handlers.clear()
        
        # Set level
        root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        
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
        else:
            log_file = Path(log_file)
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)-8s] [%(name)-30s:%(lineno)-4d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        
        # Suppress noisy loggers
        for logger_name in ["urllib3", "requests", "oracledb", "PIL", "matplotlib"]:
            logging.getLogger(logger_name).setLevel(logging.WARNING)
        
        # IMPORTANT: Set propagate to False for pytest loggers to avoid duplication
        logging.getLogger("pytest").propagate = False
        logging.getLogger("_pytest").propagate = False
        
        cls._initialized = True
        root_logger.info(f"Logging initialized - Level: {log_level}, File: {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a module."""
    logger = logging.getLogger(name)
    # Ensure logger doesn't duplicate by setting propagate correctly
    if name != "__main__" and "." in name:
        logger.propagate = True
    return logger
