import json
import uuid
import time
from typing import Dict, Any, Optional
from playwright.sync_api import APIRequestContext, APIResponse

from ..config.settings import Settings
from ..utils.logger_config import get_logger
from ..utils.request_response_tracker import RequestResponseTracker


class ChatbotAPIClient:
    """Playwright-based API client with request/response tracking."""
    
    def __init__(self, config: Settings, tracker: Optional[RequestResponseTracker] = None, request_context: Optional[APIRequestContext] = None):
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.API_BASE_URL.rstrip('/')
        self.timeout = config.API_TIMEOUT * 1000  # Convert to milliseconds for Playwright
        self.tracker = tracker
        self.request_context = request_context
        
        if not self.request_context:
            raise ValueError("Playwright APIRequestContext is required")
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        correlation_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make HTTP request using Playwright with tracking and error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Use provided timeout or default from config
        request_timeout = timeout or self.timeout
        
        # Prepare headers
        request_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'ChatbotAutomation/1.0',
            'CLIENT-CORRELATION-ID': correlation_id
        }
        
        if headers:
            request_headers.update(headers)
        
        # Track request
        if self.tracker:
            self.tracker.add_request(method, url, request_headers, data, correlation_id)
        
        self.logger.info(f"[{correlation_id}] {method} {url}")
        
        start_time = time.time()
        response = None
        
        try:
            # Make Playwright API request based on method
            if method.upper() == 'GET':
                response = self.request_context.get(
                    url, 
                    headers=request_headers,
                    timeout=request_timeout
                )
            elif method.upper() == 'POST':
                response = self.request_context.post(
                    url,
                    headers=request_headers,
                    data=json.dumps(data) if data else None,
                    timeout=request_timeout
                )
            elif method.upper() == 'PUT':
                response = self.request_context.put(
                    url,
                    headers=request_headers, 
                    data=json.dumps(data) if data else None,
                    timeout=request_timeout
                )
            elif method.upper() == 'DELETE':
                response = self.request_context.delete(
                    url,
                    headers=request_headers,
                    timeout=request_timeout
                )
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            duration = time.time() - start_time
            
            # Track response
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
            
            # Handle non-successful responses
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
                self.logger.warning(f"[{correlation_id}] Failed to parse JSON response: {e}")
                return {"text": response.text(), "status": response.status}
            
        except Exception as e:
            duration = time.time() - start_time
            self.logger.error(f"[{correlation_id}] Request failed after {duration:.2f}s: {e}")
            
            if self.tracker:
                self.tracker.add_error(type(e).__name__, str(e), correlation_id)
            
            # Re-raise with context
            raise Exception(f"API request failed: {e}") from e
    
    def initiate_chat(
        self, 
        request_data: Dict[str, Any], 
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initiate a new chat conversation."""
        correlation_id = correlation_id or f"INIT-{uuid.uuid4()}"
        
        # Ensure conversationId is set appropriately
        if 'conversationId' not in request_data and 'conversationID' not in request_data:
            request_data['conversationId'] = 'initial'
        
        self.logger.info(f"Initiating chat with correlation ID: {correlation_id}")
        
        return self._make_request(
            'POST', 
            '/api/agentic-chat/v1', 
            data=request_data,
            correlation_id=correlation_id
        )
    
    def send_message(
        self, 
        conversation_id: str, 
        chat_text: str, 
        action: str = "proceed",
        headers: Optional[Dict] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a message in an existing conversation."""
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
        
        return self._make_request(
            'POST',
            '/api/agentic-chat/v1',
            data=payload,
            headers=headers,
            correlation_id=correlation_id
        )
    
    def get_conversation_status(
        self,
        conversation_id: str,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get the status of a conversation (if supported by API)."""
        correlation_id = correlation_id or f"STATUS-{uuid.uuid4()}"
        
        return self._make_request(
            'GET',
            f'/api/agentic-chat/v1/conversations/{conversation_id}/status',
            correlation_id=correlation_id
        )
    
    def health_check(self, correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Perform API health check."""
        correlation_id = correlation_id or f"HEALTH-{uuid.uuid4()}"
        
        return self._make_request(
            'GET',
            '/api/health',
            correlation_id=correlation_id
        )
    
    def close(self) -> None:
        """Close the API client and clean up resources."""
        # Note: The APIRequestContext is managed by the test session
        # We don't dispose it here as it might be shared across tests
        self.logger.info("Playwright API client closed")


class PlaywrightAPIHelpers:
    """Helper utilities for Playwright API testing."""
    
    @staticmethod
    def validate_response(response: APIResponse, expected_status: int = 200) -> Dict[str, Any]:
        """Validate API response and return JSON data."""
        if response.status != expected_status:
            error_body = ""
            try:
                error_body = response.text()
            except:
                pass
            raise AssertionError(
                f"Expected status {expected_status}, got {response.status}. "
                f"Response: {error_body}"
            )
        
        try:
            return response.json()
        except Exception as e:
            raise AssertionError(f"Failed to parse JSON response: {e}. Response text: {response.text()}")
    
    @staticmethod
    def extract_conversation_id(response_data: Dict[str, Any]) -> str:
        """Extract conversation ID from response."""
        conv_id = response_data.get('conversationID') or response_data.get('conversationId')
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
    def is_json_response(response: APIResponse) -> bool:
        """Check if response is JSON."""
        content_type = response.headers.get('content-type', '')
        return 'application/json' in content_type.lower()
