import re
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, List, Tuple

from .logger_config import get_logger


class TestHelpers:
    """Common helper functions for test automation."""
    
    logger = get_logger(__name__)
    
    @staticmethod
    def generate_correlation_id(prefix: str = "TEST") -> str:
        """Generate a unique correlation ID."""
        return f"{prefix}-{uuid.uuid4()}"
    
    @staticmethod
    def validate_conversation_id(conv_id: str) -> bool:
        """Validate conversation ID format (CVD-UUID)."""
        pattern = r'^CVD[E,D,L]-[0-9]{6}-[0-9a-fA-F]{12}$'
        return bool(re.match(pattern, conv_id))
    
    @staticmethod
    def validate_interaction_id(interaction_id: str) -> bool:
        """Validate interaction ID format (E/D/L followed by 10 digits)."""
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
    def safe_dict_get(
        data: Dict[str, Any], 
        path: str, 
        default: Any = None
    ) -> Any:
        """
        Safely get nested dictionary values using dot notation.
        Example: safe_dict_get(data, 'user.profile.name')
        """
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
    def mask_sensitive_data(
        text: str, 
        patterns: Optional[List[Tuple[str, Any]]] = None
    ) -> str:
        """Mask sensitive data in text."""
        if patterns is None:
            patterns = [
                # Account numbers (4+ digits)
                (r'\b\d{4,}\b', lambda m: '*' * len(m.group())),
                # Passwords
                (r'password["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 
                 lambda m: 'password: ***'),
                # Tokens
                (r'token["\']?\s*[:=]\s*["\']?([^"\'\s]+)', 
                 lambda m: 'token: ***'),
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
        return round((passed / total) * 100, 2)
    
    @staticmethod
    def format_timestamp(
        dt: Optional[datetime] = None, 
        format_str: str = "%Y-%m-%d %H:%M:%S"
    ) -> str:
        """Format timestamp for logging."""
        if dt is None:
            dt = datetime.now()
        return dt.strftime(format_str)