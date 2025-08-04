import requests
import json
import uuid
from typing import Dict, Any, Optional
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from requests.exceptions import RequestException, HTTPError

from config.settings import Settings
from core.utils.logger import get_logger

class ChatbotAPIClient:
    """Enterprise-grade API client with retry logic and comprehensive error handling."""

    def __init__(self, config: Settings):
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.API_BASE_URL
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
        
        # Configure SSL certificate
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

    def _log_request(self, method: str, url: str, headers: Dict, data: Any):
        """Log API request details."""
        if self.config.ENABLE_DETAILED_LOGGING:
            self.logger.debug(f"API Request: {method} {url}")
            self.logger.debug(f"Headers: {self._sanitize_headers(headers)}")
            self.logger.debug(f"Payload: {json.dumps(data, indent=2)}")
        else:
            self.logger.info(f"API Request: {method} {url}")

    def _log_response(self, response: requests.Response, correlation_id: str):
        """Log API response details."""
        if self.config.ENABLE_DETAILED_LOGGING:
            self.logger.debug(f"[{correlation_id}] Response Status: {response.status_code}")
            self.logger.debug(f"[{correlation_id}] Response Body: {response.text}")
        else:
            self.logger.info(f"[{correlation_id}] Response Status: {response.status_code}")

    def _sanitize_headers(self, headers: Dict) -> Dict:
        """Remove sensitive information from headers for logging."""
        sanitized = headers.copy()
        sensitive_keys = ['authorization', 'x-api-key', 'cookie']
        for key in sensitive_keys:
            if key.lower() in {k.lower() for k in sanitized}:
                sanitized[key] = '***REDACTED***'
        return sanitized

    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None, 
                    headers: Optional[Dict] = None) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        correlation_id = headers.get('CLIENT-CORRELATION-ID', str(uuid.uuid4())) if headers else str(uuid.uuid4())
        
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
        
        self._log_request(method, url, request_headers, data)
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                json=data,
                headers=request_headers,
                timeout=self.config.API_TIMEOUT
            )
            
            self._log_response(response, correlation_id)
            response.raise_for_status()
            
            return response.json()
            
        except HTTPError as e:
            self.logger.error(f"[{correlation_id}] HTTP Error: {e}")
            self.logger.error(f"[{correlation_id}] Response: {e.response.text if e.response else 'No response'}")
            raise
        except RequestException as e:
            self.logger.error(f"[{correlation_id}] Request Error: {e}")
            raise
        except Exception as e:
            self.logger.error(f"[{correlation_id}] Unexpected Error: {e}")
            raise

    def initiate_chat(self, request_data: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Initiate a new chat conversation."""
        headers = {
            'CLIENT-CORRELATION-ID': correlation_id or f"INIT-{uuid.uuid4()}"
        }
        
        request_data['conversationId'] = 'initial'
        
        self.logger.info(f"Initiating chat with correlation ID: {headers['CLIENT-CORRELATION-ID']}")
        return self._make_request('POST', '/api/agentic-chat/v1', data=request_data, headers=headers)

    def send_message(self, conversation_id: str, message: str, action: str = "proceed",
                    correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Send a message in an existing conversation."""
        headers = {
            'CLIENT-CORRELATION-ID': correlation_id or f"MSG-{uuid.uuid4()}"
        }
        
        payload = {
            "channelID": "BBVA",
            "conversationID": conversation_id,
            "requestType": "ComplaintCapture",
            "chatText": message,
            "action": action
        }
        
        self.logger.info(f"Sending message to conversation: {conversation_id}")
        return self._make_request('POST', '/api/agentic-chat/v1', data=payload, headers=headers)

    def close(self):
        """Close the session and clean up resources."""
        if self.session:
            self.session.close()
            self.logger.info("API client session closed")
