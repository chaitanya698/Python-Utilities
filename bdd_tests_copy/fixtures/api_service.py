import requests
import json
import uuid
import time
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException, HTTPError, Timeout

from ..config.settings import Settings
from ..utils.logger_config import get_logger
from ..utils.request_response_tracker import RequestResponseTracker


class ChatbotAPIClient:
    """Enterprise-grade API client with request/response tracking."""
    
    def __init__(self, config: Settings, tracker: Optional[RequestResponseTracker] = None):
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.API_BASE_URL.rstrip('/')
        self.timeout = config.API_TIMEOUT
        self.tracker = tracker
        self.session = self._create_session()
    
    def _create_session(self) -> requests.Session:
        """Create a configured session with retry logic and certificate handling."""
        session = requests.Session()
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=self.config.API_RETRY_COUNT,
            backoff_factor=1,
            status_forcelist=[502, 503, 504],
            allowed_methods=["GET", "POST", "PUT", "DELETE"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        
        # Configure SSL certificate if available
        if self.config.CERT_PEM_PATH and self.config.KEY_PEM_PATH:
            session.cert = (self.config.CERT_PEM_PATH, self.config.KEY_PEM_PATH)
            self.logger.info("Session configured with client certificate")
        
        # Set default headers
        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'ChatbotAutomation/1.0'
        })
        
        # Configure SSL verification
        session.verify = self.config.VERIFY_SSL
        
        return session
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        correlation_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make HTTP request with tracking and error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        correlation_id = correlation_id or str(uuid.uuid4())
        
        # Use provided timeout or default from config
        request_timeout = timeout or self.timeout
        
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
        
        request_headers['CLIENT-CORRELATION-ID'] = correlation_id
        
        # Track request
        if self.tracker:
            self.tracker.add_request(method, url, request_headers, data, correlation_id)
        
        self.logger.info(f"[{correlation_id}] {method} {url}")
        
        start_time = time.time()
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                headers=request_headers,
                timeout=request_timeout
            )
            
            duration = time.time() - start_time
            
            # Track response
            if self.tracker:
                try:
                    response_data = response.json()
                except:
                    response_data = response.text
                
                self.tracker.add_response(
                    response.status_code,
                    response.headers,
                    response_data,
                    duration,
                    correlation_id
                )
            
            self.logger.info(f"[{correlation_id}] Response Status: {response.status_code}")
            response.raise_for_status()
            
            return response.json()
            
        except Timeout as e:
            self.logger.error(f"[{correlation_id}] Request timeout after {request_timeout}s")
            if self.tracker:
                self.tracker.add_error("Timeout", str(e), correlation_id)
            raise
        except HTTPError as e:
            self.logger.error(f"[{correlation_id}] HTTP Error: {e}")
            if self.tracker:
                self.tracker.add_error("HTTPError", str(e), correlation_id)
            raise
        except Exception as e:
            self.logger.error(f"[{correlation_id}] Unexpected Error: {e}")
            if self.tracker:
                self.tracker.add_error(type(e).__name__, str(e), correlation_id)
            raise
    
    def initiate_chat(
        self, 
        request_data: Dict[str, Any], 
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Initiate a new chat conversation."""
        correlation_id = correlation_id or f"INIT-{uuid.uuid4()}"
        
        # Ensure conversationId is set appropriately
        if 'conversationId' not in request_data:
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
    
    def close(self) -> None:
        """Close the session and clean up resources."""
        if self.session:
            self.session.close()
            self.logger.info("API client session closed")
