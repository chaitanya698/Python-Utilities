import os
import sys
from typing import List, Dict, Optional


class ConfigValidator:
    """Validates required configuration parameters."""
    
    REQUIRED_ENV_VARS = [
        "DB_USER",
        "DB_PASSWORD",
        "DB_HOST",
        "DB_SERVICE_NAME",
        "API_BASE_URL",
        "CERT_PFX_PATH",
        "CERT_PASSWORD"
    ]
    
    OPTIONAL_WITH_DEFAULTS = {
        "DB_PORT": "1521",
        "ENVIRONMENT": "qa",
        "LOG_LEVEL": "INFO",
        "API_TIMEOUT": "45",
        "API_RETRY_COUNT": "3"
    }
    
    @classmethod
    def validate_required_vars(cls) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = cls.get_missing_vars()
        
        if missing_vars:
            error_msg = cls._build_error_message(missing_vars)
            print(error_msg, file=sys.stderr)
            sys.exit(1)
    
    @classmethod
    def get_missing_vars(cls) -> List[str]:
        """Get list of missing required variables."""
        return [var for var in cls.REQUIRED_ENV_VARS if not os.getenv(var)]
    
    @classmethod
    def _build_error_message(cls, missing_vars: List[str]) -> str:
        """Build descriptive error message for missing configuration."""
        return (
            "\n" + "=" * 60 + "\n"
            "CONFIGURATION ERROR\n"
            "=" * 60 + "\n"
            "The following required parameters are missing:\n" +
            "\n".join(f"  - {var}" for var in missing_vars) +
            "\n\nPlease set these as environment variables:\n"
            "  Example: export DB_USER=myuser DB_PASSWORD=mypass\n"
            "  Or use .env.qa file in the project root\n" +
            "=" * 60
        )