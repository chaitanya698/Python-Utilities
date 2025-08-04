import os
import sys
from typing import List, Dict

class ConfigValidator:
    """Validates required configuration parameters."""

    REQUIRED_ENV_VARS = [
        "DB_USER",
        "DB_PASSWORD",
        "CERT_PFX_PATH",
        "CERT_PASSWORD"
    ]

    REQUIRED_WITH_DEFAULTS = {
        "API_BASE_URL": None,
        "DB_HOST": None,
        "DB_PORT": "1521",
        "DB_SERVICE_NAME": None
    }

    @classmethod
    def validate_required_vars(cls) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = []
        
        # Check absolutely required variables
        for var in cls.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                missing_vars.append(var)
        
        # Check variables that need values (no defaults)
        for var, default in cls.REQUIRED_WITH_DEFAULTS.items():
            if default is None and not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars:
            error_msg = (
                "\n" + "="*50 + "\n"
                "CONFIGURATION ERROR\n"
                "="*50 + "\n"
                "The following required parameters are missing:\n"
            )
            for var in missing_vars:
                error_msg += f"  - {var}\n"
            
            error_msg += (
                "\nPlease set these as environment variables or VM arguments:\n"
                "  Example: -DDB_USER=myuser -DDB_PASSWORD=mypass\n"
                "  Or: export DB_USER=myuser\n"
                "="*50
            )
            
            print(error_msg, file=sys.stderr)
            sys.exit(1)

    @classmethod
    def get_missing_vars(cls) -> List[str]:
        """Get list of missing required variables."""
        missing = []
        
        for var in cls.REQUIRED_ENV_VARS:
            if not os.getenv(var):
                missing.append(var)
                
        for var, default in cls.REQUIRED_WITH_DEFAULTS.items():
            if default is None and not os.getenv(var):
                missing.append(var)
                
        return missing
