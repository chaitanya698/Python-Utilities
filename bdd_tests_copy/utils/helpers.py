import uuid
import time
from datetime import datetime
from typing import Dict, Any, Optional
from .logger_config import get_logger


class TestHelpers:
    """Utility class for common test operations and helpers."""
    
    @staticmethod
    def generate_correlation_id(prefix: str = "TEST") -> str:
        """Generate a unique correlation ID for tracking requests."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        short_uuid = str(uuid.uuid4())[:8].upper()
        return f"{prefix}-{timestamp}-{short_uuid}"
    
    @staticmethod
    def generate_conversation_id() -> str:
        """Generate a unique conversation ID."""
        return f"CONV-{uuid.uuid4()}"
    
    @staticmethod
    def wait_for_condition(
        condition_func,
        timeout: float = 30.0,
        interval: float = 1.0,
        description: str = "condition"
    ) -> bool:
        """
        Wait for a condition to become true.
        
        Args:
            condition_func: Function that returns True when condition is met
            timeout: Maximum time to wait in seconds
            interval: Time between checks in seconds  
            description: Description of what we're waiting for
            
        Returns:
            True if condition was met, False if timeout
        """
        logger = get_logger(__name__)
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                if condition_func():
                    logger.info(f"✅ {description} - condition met")
                    return True
            except Exception as e:
                logger.debug(f"Condition check failed: {e}")
            
            time.sleep(interval)
        
        logger.warning(f"⚠️  {description} - timeout after {timeout}s")
        return False
    
    @staticmethod
    def clean_text_for_comparison(text: str) -> str:
        """Clean text for comparison by normalizing whitespace and case."""
        if not text:
            return ""
        return ' '.join(text.lower().split())
    
    @staticmethod
    def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
        """Try to extract JSON from text response."""
        import json
        import re
        
        # Look for JSON-like patterns
        json_pattern = r'\{.*\}'
        matches = re.findall(json_pattern, text, re.DOTALL)
        
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        
        return None
    
    @staticmethod
    def mask_sensitive_data(data: Dict[str, Any], sensitive_keys: list = None) -> Dict[str, Any]:
        """Mask sensitive data in dictionary for logging."""
        if sensitive_keys is None:
            sensitive_keys = [
                'password', 'pwd', 'secret', 'token', 'key', 'auth',
                'credential', 'api_key', 'access_token', 'refresh_token'
            ]
        
        masked_data = {}
        for key, value in data.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                masked_data[key] = "***MASKED***"
            elif isinstance(value, dict):
                masked_data[key] = TestHelpers.mask_sensitive_data(value, sensitive_keys)
            else:
                masked_data[key] = value
                
        return masked_data
    
    @staticmethod
    def validate_response_structure(response: Dict[str, Any], required_fields: list) -> bool:
        """Validate that response contains required fields."""
        logger = get_logger(__name__)
        
        missing_fields = []
        for field in required_fields:
            if field not in response:
                missing_fields.append(field)
        
        if missing_fields:
            logger.warning(f"Missing required fields in response: {missing_fields}")
            return False
        
        return True
    
    @staticmethod
    def get_nested_value(data: Dict[str, Any], path: str, default=None):
        """Get nested value from dictionary using dot notation path."""
        try:
            current = data
            for key in path.split('.'):
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    @staticmethod
    def format_duration(seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 1:
            return f"{seconds*1000:.0f}ms"
        elif seconds < 60:
            return f"{seconds:.1f}s"
        else:
            minutes = int(seconds // 60)
            remaining_seconds = seconds % 60
            return f"{minutes}m {remaining_seconds:.1f}s"
    
    @staticmethod
    def create_test_summary(results: Dict[str, Any]) -> str:
        """Create a formatted test summary."""
        total = results.get('total', 0)
        passed = results.get('passed', 0)
        failed = results.get('failed', 0)
        skipped = results.get('skipped', 0)
        
        summary = [
            "=" * 50,
            "TEST EXECUTION SUMMARY",
            "=" * 50,
            f"Total Tests: {total}",
            f"✅ Passed: {passed}",
            f"❌ Failed: {failed}",
            f"⚠️  Skipped: {skipped}",
            f"📊 Pass Rate: {(passed/total*100):.1f}%" if total > 0 else "📊 Pass Rate: N/A",
            "=" * 50
        ]
        
        return "\n".join(summary)
