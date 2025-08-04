import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

from core.utils.logger import get_logger

logger = get_logger(name)

class TestHelpers:
    """Common helper functions for test automation."""

    @staticmethod
    def generate_correlation_id(prefix: str = "TEST") -> str:
        """Generate a unique correlation ID."""
        return f"{prefix}-{uuid.uuid4()}"

    @staticmethod
    def validate_conversation_id(conv_id: str) -> bool:
        """Validate conversation ID format."""
        pattern = r'^CVD-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
        return bool(re.match(pattern, conv_id))

    @staticmethod
    def validate_interaction_id(interaction_id: str) -> bool:
        """Validate interaction ID format."""
        pattern = r'^[EDL]\d{10}$'
        return bool(re.match(pattern, interaction_id))

    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.2f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.2f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.2f}h"

    @staticmethod
    def safe_dict_get(data: Dict[str, Any], path: str, default: Any = None) -> Any:
        """Safely get nested dictionary values."""
        keys = path.split('.')
        value = data
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value

    @staticmethod
    def mask_sensitive_data(text: str, patterns: Optional[list] = None) -> str:
        """Mask sensitive data in text."""
        if patterns is None:
            patterns = [
                (r'\b\d{4,}\b', lambda m: '*' * len(m.group())),  # Numbers > 4 digits
                (r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', lambda m: f'password: ***'),
                (r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)', lambda m: f'token: ***'),
            ]
        
        masked = text
        for pattern, replacement in patterns:
            masked = re.sub(pattern, replacement, masked, flags=re.IGNORECASE)
        
        return masked

    @staticmethod
    def calculate_success_rate(passed: int, total: int) -> float:
        """Calculate success rate percentage."""
        if total == 0:
            return 0.0
        return (passed / total) * 100

    @staticmethod
    def format_timestamp(dt: datetime = None, format: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Format timestamp for logging."""
        if dt is None:
            dt = datetime.now()
        return dt.strftime(format)
