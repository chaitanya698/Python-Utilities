import json
import uuid
import time
import asyncio
from typing import Dict, Any, Optional, Callable, Union
from playwright.sync_api import APIRequestContext, APIResponse, TimeoutError as PlaywrightTimeoutError

from ..config.settings import Settings
from ..utils.logger_config import get_logger
from ..utils.request_response_tracker import RequestResponseTracker

class PlaywrightAPIWaitStrategies:
“”“Waiting strategies for API testing with Playwright.”””

```
@staticmethod
def wait_for_status_code(
    response: APIResponse, 
    expected_status: Union[int, list] = 200,
    timeout: int = 30
) -> bool:
    """Wait and verify response has expected status code."""
    if isinstance(expected_status, int):
        expected_status = [expected_status]
    
    return response.status in expected_status

@staticmethod
def wait_for_json_response(response: APIResponse, timeout: int = 30) -> bool:
    """Wait and verify response contains valid JSON."""
    try:
        response.json()
        return True
    except:
        return False

@staticmethod
def wait_for_response_field(
    response: APIResponse, 
    field_path: str, 
    expected_value: Any = None,
    timeout: int = 30
) -> bool:
    """Wait for specific field in JSON response."""
    try:
        data = response.json()
        keys = field_path.split('.')
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return False
        
        if expected_value is not None:
            return current == expected_value
        return current is not None
    except:
        return False

@staticmethod
def wait_for_conversation_id(response: APIResponse, timeout: int = 30) -> bool:
    """Wait for valid conversation ID in response."""
    return PlaywrightAPIWaitStrategies.wait_for_response_field(
        response, 'conversationID', timeout=timeout
    ) or PlaywrightAPIWaitStrategies.wait_for_response_field(
        response, 'conversationId', timeout=timeout
    )
```

class ChatbotAPIClient:
“”“Playwright-based API client with comprehensive waiting mechanisms.”””

```
def __init__(
    self, 
    config: Settings, 
    tracker: Optional[RequestResponseTracker] = None, 
    request_context: Optional[APIRequestContext] = None
):
    self.config = config
    self.logger = get_logger(__name__)
    self.base_url = config.API_BASE_URL.rstrip('/')
    self.timeout = config.API_TIMEOUT * 1000  # Convert to milliseconds
    self.tracker = tracker
    self.request_context = request_context
    self.wait_strategies = PlaywrightAPIWaitStrategies()
    
    if not self.request_context:
        raise ValueError("Playwright APIRequestContext is required for API-only testing")

def _make_request(
    self, 
    method: str, 
    endpoint: str, 
    data: Optional[Dict] = None,
    headers: Optional[Dict] = None,
    correlation_id: Optional[str] = None,
    timeout: Optional[int] = None,
    wait_strategy: Optional[Callable[[APIResponse], bool]] = None,
    retry_count: int = 3,
    retry_delay: float = 1.0
) -> Dict[str, Any]:
    """
    Make HTTP request using Playwright with advanced waiting and retry mechanisms.
    
    Args:
        method: HTTP method
        endpoint: API endpoint
        data: Request payload
        headers: Additional headers
        correlation_id: Correlation ID for tracking
        timeout: Request timeout in milliseconds
        wait_strategy: Custom wait condition function
        retry_count: Number of retries
        retry_delay: Delay between retries in seconds
    """
    url = f"{self.base_url}/{endpoint.lstrip('/')}"
    correlation_id = correlation_id or str(uuid.uuid4())
    request_timeout = timeout or self.timeout
    
    # Prepare headers
    request_headers = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'User-Agent': 'ChatbotAutomation-Playwright/1.0',
        'CLIENT-CORRELATION-ID': correlation_id
    }
    
    if headers:
        request_headers.update(headers)
    
    # Track request
    if self.tracker:
        self.tracker.add_request(method, url, request_headers, data, correlation_id)
    
    self.logger.info(f"[{correlation_id}] {method} {url}")
    
    last_exception = None
    
    # Retry logic
    for attempt in range(retry_count + 1):
        start_time = time.time()
        
        try:
            # Make Playwright API request
            response = self._execute_playwright_request(
                method, url, request_headers, data, request_timeout
            )
            
            duration = time.time() - start_time
            
            # Apply wait strategy if provided
            if wait_strategy and not wait_strategy(response):
                raise Exception(f"Wait strategy failed for response")
            
            # Default wait strategies
            if not self.wait_strategies.wait_for_status_code(response, [200, 201, 202, 400, 401, 403, 404, 500]):
                raise Exception(f"Unexpected status code: {response.status}")
            
            # Track successful response
            if self.tracker:
                try:
                    response_data = response.json()
                except:
                    response_data = response.text()
                
                self.tracker.add_response(
                    response.status,
                    dict(response.headers),
                    response_data,
                    duration,
                    correlation_id
                )
            
            self.logger.info(f"[{correlation_id}] Response Status: {response.status} Duration: {duration:.2f}s")
            
            # Handle non-successful HTTP status codes
            if not response.ok:
                error_msg = f"HTTP {response.status}: {response.status_text}"
                try:
                    error_body = response.text()
                    if error_body:
                        error_msg += f" - {error_body}"
                except:
                    pass
                raise Exception(error_msg)
            
            # Return JSON response
            try:
                return response.json()
            except Exception as e:
                self.logger.warning(f"[{correlation_id}] Failed to parse JSON: {e}")
                return {"text": response.text(), "status": response.status}
            
        except PlaywrightTimeoutError as e:
            last_exception = e
            duration = time.time() - start_time
            self.logger.warning(f"[{correlation_id}] Attempt {attempt + 1}/{retry_count + 1} timed out after {duration:.2f}s")
            
        except Exception as e:
            last_exception = e
            duration = time.time() - start_time
            self.logger.warning(f"[{correlation_id}] Attempt {attempt + 1}/{retry_count + 1} failed after {duration:.2f}s: {e}")
        
        # Wait before retry (except on last attempt)
        if attempt < retry_count:
            time.sleep(retry_delay)
            retry_delay *= 1.5  # Exponential backoff
    
    # All retries failed
    duration = time.time() - start_time
    self.logger.error(f"[{correlation_id}] All {retry_count + 1} attempts failed after {duration:.2f}s")
    
    if self.tracker:
        self.tracker.add_error(type(last_exception).__name__, str(last_exception), correlation_id)
    
    raise Exception(f"API request failed after {retry_count + 1} attempts: {last_exception}")

def _execute_playwright_request(
    self, 
    method: str, 
    url: str, 
    headers: Dict[str, str], 
    data: Optional[Dict], 
    timeout: int
) -> APIResponse:
    """Execute the actual Playwright API request."""
    request_kwargs = {
        "headers": headers,
        "timeout": timeout
    }
    
    if data:
        request_kwargs["data"] = json.dumps(data)
    
    if method.upper() == 'GET':
        return self.request_context.get(url, **request_kwargs)
    elif method.upper() == 'POST':
        return self.request_context.post(url, **request_kwargs)
    elif method.upper() == 'PUT':
        return self.request_context.put(url, **request_kwargs)
    elif method.upper() == 'DELETE':
        return self.request_context.delete(url, **request_kwargs)
    elif method.upper() == 'PATCH':
        return self.request_context.patch(url, **request_kwargs)
    else:
        raise ValueError(f"Unsupported HTTP method: {method}")

def initiate_chat(
    self, 
    request_data: Dict[str, Any], 
    correlation_id: Optional[str] = None,
    wait_for_conversation_id: bool = True,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Initiate a new chat conversation with advanced waiting.
    
    Args:
        request_data: Request payload
        correlation_id: Correlation ID
        wait_for_conversation_id: Wait for valid conversation ID in response
        timeout: Custom timeout
    """
    correlation_id = correlation_id or f"INIT-{uuid.uuid4()}"
    
    # Ensure conversationId is set appropriately
    if 'conversationId' not in request_data and 'conversationID' not in request_data:
        request_data['conversationId'] = 'initial'
    
    self.logger.info(f"Initiating chat with correlation ID: {correlation_id}")
    
    # Define wait strategy for conversation initiation
    def wait_for_init_response(response: APIResponse) -> bool:
        if not self.wait_strategies.wait_for_status_code(response, [200, 201]):
            return False
        
        if wait_for_conversation_id:
            return self.wait_strategies.wait_for_conversation_id(response)
        
        return self.wait_strategies.wait_for_json_response(response)
    
    return self._make_request(
        'POST', 
        '/api/agentic-chat/v1', 
        data=request_data,
        correlation_id=correlation_id,
        timeout=timeout,
        wait_strategy=wait_for_init_response,
        retry_count=self.config.API_RETRY_COUNT
    )

def send_message(
    self, 
    conversation_id: str, 
    chat_text: str, 
    action: str = "proceed",
    headers: Optional[Dict] = None,
    correlation_id: Optional[str] = None,
    wait_for_response_text: bool = True,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """
    Send a message in an existing conversation with response validation.
    
    Args:
        conversation_id: Conversation ID
        chat_text: Message text
        action: Action type
        headers: Additional headers
        correlation_id: Correlation ID
        wait_for_response_text: Wait for chatResponseText in response
        timeout: Custom timeout
    """
    correlation_id = correlation_id or (headers.get('CLIENT-CORRELATION-ID') if headers else None)
    correlation_id = correlation_id or f"MSG-{uuid.uuid4()}"
    
    payload = {
        "channelID": "BBVA",
        "conversationID": conversation_id,
        "requestType": "ComplaintCapture",
        "chatText": chat_text,
        "action": action
    }
    
    self.logger.info(f"Sending message to conversation: {conversation_id}")
    
    # Define wait strategy for message response
    def wait_for_message_response(response: APIResponse) -> bool:
        if not self.wait_strategies.wait_for_status_code(response, [200, 201]):
            return False
        
        if wait_for_response_text:
            return self.wait_strategies.wait_for_response_field(response, 'chatResponseText')
        
        return self.wait_strategies.wait_for_json_response(response)
    
    return self._make_request(
        'POST',
        '/api/agentic-chat/v1',
        data=payload,
        headers=headers,
        correlation_id=correlation_id,
        timeout=timeout,
        wait_strategy=wait_for_message_response,
        retry_count=self.config.API_RETRY_COUNT
    )

def get_conversation_status(
    self,
    conversation_id: str,
    correlation_id: Optional[str] = None,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Get the status of a conversation."""
    correlation_id = correlation_id or f"STATUS-{uuid.uuid4()}"
    
    def wait_for_status_response(response: APIResponse) -> bool:
        return self.wait_strategies.wait_for_status_code(response, [200, 404])
    
    return self._make_request(
        'GET',
        f'/api/agentic-chat/v1/conversations/{conversation_id}/status',
        correlation_id=correlation_id,
        timeout=timeout,
        wait_strategy=wait_for_status_response
    )

def health_check(
    self, 
    correlation_id: Optional[str] = None,
    timeout: Optional[int] = None
) -> Dict[str, Any]:
    """Perform API health check with basic validation."""
    correlation_id = correlation_id or f"HEALTH-{uuid.uuid4()}"
    
    def wait_for_health_response(response: APIResponse) -> bool:
        return self.wait_strategies.wait_for_status_code(response, [200, 503])
    
    return self._make_request(
        'GET',
        '/api/health',
        correlation_id=correlation_id,
        timeout=timeout or 10000,  # Shorter timeout for health checks
        wait_strategy=wait_for_health_response,
        retry_count=1  # Less retries for health checks
    )

def wait_for_response_condition(
    self,
    response: APIResponse,
    condition: str,
    expected_value: Any = None,
    timeout: int = 30
) -> bool:
    """
    Wait for specific condition in API response.
    
    Args:
        response: API response
        condition: Condition type ('status', 'field', 'json', 'conversation_id')
        expected_value: Expected value for the condition
        timeout: Timeout in seconds
    """
    if condition == 'status':
        return self.wait_strategies.wait_for_status_code(response, expected_value, timeout)
    elif condition == 'field':
        field_path, expected = expected_value if isinstance(expected_value, tuple) else (expected_value, None)
        return self.wait_strategies.wait_for_response_field(response, field_path, expected, timeout)
    elif condition == 'json':
        return self.wait_strategies.wait_for_json_response(response, timeout)
    elif condition == 'conversation_id':
        return self.wait_strategies.wait_for_conversation_id(response, timeout)
    else:
        raise ValueError(f"Unknown condition: {condition}")

def close(self) -> None:
    """Close the API client and clean up resources."""
    # Note: APIRequestContext is managed by the test session
    # We don't dispose it here as it might be shared across tests
    self.logger.info("Playwright API client closed")
```

class PlaywrightAPIHelpers:
“”“Enhanced helper utilities for Playwright API testing.”””

```
@staticmethod
def validate_response_structure(
    response: APIResponse, 
    expected_fields: list,
    strict: bool = False
) -> bool:
    """
    Validate response has expected structure.
    
    Args:
        response: API response
        expected_fields: List of expected field paths
        strict: If True, response must have ONLY these fields
    """
    try:
        data = response.json()
        
        for field_path in expected_fields:
            keys = field_path.split('.')
            current = data
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return False
        
        return True
    except:
        return False

@staticmethod
def extract_conversation_id(response_data: Dict[str, Any]) -> str:
    """Extract conversation ID from response with fallback options."""
    conv_id = (
        response_data.get('conversationID') or 
        response_data.get('conversationId') or
        response_data.get('conversation_id')
    )
    
    if not conv_id:
        raise AssertionError(f"No conversation ID found in response: {response_data}")
    
    return conv_id

@staticmethod
def extract_interaction_id(response_text: str) -> Optional[str]:
    """Extract interaction ID from response text using regex."""
    import re
    pattern = r'INT[EOL]-\d{6}-\w{12}'
    match = re.search(pattern, response_text)
    return match.group(0) if match else None

@staticmethod
def validate_api_error_response(
    response: APIResponse,
    expected_error_code: Optional[str] = None
) -> bool:
    """Validate API error response format."""
    if response.ok:
        return False
    
    try:
        error_data = response.json()
        has_error_structure = any(key in error_data for key in ['error', 'message', 'detail'])
        
        if expected_error_code:
            error_code = error_data.get('error', {}).get('code') or error_data.get('code')
            return has_error_structure and error_code == expected_error_code
        
        return has_error_structure
    except:
        return False

@staticmethod
def measure_response_time(response: APIResponse) -> float:
    """Measure response time if available in headers."""
    try:
        # Some APIs include response time in headers
        server_timing = response.headers.get('server-timing', '')
        if 'total;dur=' in server_timing:
            import re
            match = re.search(r'total;dur=(\d+\.?\d*)', server_timing)
            if match:
                return float(match.group(1))
    except:
        pass
    
    return 0.0

@staticmethod
def create_wait_strategy(
    status_codes: list = None,
    required_fields: list = None,
    forbidden_fields: list = None,
    custom_validator: Callable[[Dict], bool] = None
) -> Callable[[APIResponse], bool]:
    """
    Create a custom wait strategy for API responses.
    
    Args:
        status_codes: Expected status codes
        required_fields: Fields that must be present
        forbidden_fields: Fields that must not be present
        custom_validator: Custom validation function
    """
    def wait_strategy(response: APIResponse) -> bool:
        # Check status codes
        if status_codes and response.status not in status_codes:
            return False
        
        try:
            data = response.json()
            
            # Check required fields
            if required_fields:
                for field in required_fields:
                    if field not in data:
                        return False
            
            # Check forbidden fields
            if forbidden_fields:
                for field in forbidden_fields:
                    if field in data:
                        return False
            
            # Apply custom validator
            if custom_validator and not custom_validator(data):
                return False
            
            return True
            
        except:
            return False
    
    return wait_strategy
```
