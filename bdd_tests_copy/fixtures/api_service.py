from playwright.sync_api import sync_playwright, APIRequestContext, Playwright, Error as PlaywrightError
import uuid
import time
import os
from typing import Dict, Any, Optional
import json
from bdd_tests.config.settings import Settings
from bdd_tests.utils.logger_config import get_logger
from bdd_tests.utils.request_response_tracker import RequestResponseTracker


class ChatbotAPIClient:
    """Enterprise-grade API client with request/response tracking using Playwright."""
    
    def __init__(self, config: Settings, tracker: Optional[RequestResponseTracker] = None, 
                 playwright_context: Optional[APIRequestContext] = None):
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.API_BASE_URL.rstrip('/')
        self.timeout = config.API_TIMEOUT * 1000  # Playwright expects milliseconds
        self.tracker = tracker
        
        # Use provided context or create new one
        if playwright_context:
            self.api_request_context = playwright_context
            self.playwright = None  # Don't manage playwright lifecycle if context provided
            self.logger.info("Using provided Playwright API context")
        else:
            self.playwright = sync_playwright().start()
            self.api_request_context = self._create_api_context()

    def _create_api_context(self) -> APIRequestContext:
        """Create API request context with proper certificate handling."""
        extra_http_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'ChatbotAutomation-Playwright/1.0'
        }
        
        # Build context options
        context_options = {
            "base_url": self.base_url,
            "extra_http_headers": extra_http_headers,
            "ignore_https_errors": not self.config.VERIFY_SSL,
            "timeout": self.timeout
        }
        
        # Handle client certificates using processed PEM files from config loader
        client_certificates = self._setup_client_certificates()
        if client_certificates:
            context_options["client_certificates"] = client_certificates
            self.logger.info("Client certificates configured for Playwright API context")
        else:
            self.logger.info("No client certificates configured - using standard HTTPS")
            
        return self.playwright.request.new_context(**context_options)
    
    def _setup_client_certificates(self) -> Optional[list]:
        """Setup client certificates using processed PEM files from config loader."""
        try:
            # Check if we have processed PEM files from the config loader
            cert_pem_path = getattr(self.config, 'CERT_PEM_PATH', None)
            key_pem_path = getattr(self.config, 'KEY_PEM_PATH', None)
            
            if cert_pem_path and key_pem_path:
                # Verify files exist
                if os.path.exists(cert_pem_path) and os.path.exists(key_pem_path):
                    client_certificates = [{
                        "origin": self.base_url,
                        "certPath": cert_pem_path,
                        "keyPath": key_pem_path
                    }]
                    self.logger.info(f"Using processed PEM certificates: cert={cert_pem_path}, key={key_pem_path}")
                    return client_certificates
                else:
                    self.logger.warning(f"Processed PEM files not found: cert={cert_pem_path}, key={key_pem_path}")
            
            # Fallback: Try to handle PFX directly if no processed files available
            cert_pfx_path = getattr(self.config, 'CERT_PFX_PATH', None)
            cert_password = getattr(self.config, 'CERT_PRD', None)
            
            if cert_pfx_path and cert_password:
                # Build full path if relative
                if not os.path.isabs(cert_pfx_path):
                    # Try different possible locations
                    possible_paths = [
                        os.path.join("bdd_tests", "resources", "cert", cert_pfx_path),
                        os.path.join("bdd_tests_copy", "resources", "cert", cert_pfx_path),
                        cert_pfx_path
                    ]
                    
                    pfx_full_path = None
                    for path in possible_paths:
                        if os.path.exists(path):
                            pfx_full_path = path
                            break
                    
                    if not pfx_full_path:
                        self.logger.warning(f"PFX certificate file not found in any expected location: {possible_paths}")
                        return None
                else:
                    pfx_full_path = cert_pfx_path
                
                if os.path.exists(pfx_full_path):
                    # For PFX files, Playwright expects pfxPath and passphrase
                    client_certificates = [{
                        "origin": self.base_url,
                        "pfxPath": pfx_full_path,
                        "passphrase": cert_password
                    }]
                    self.logger.info(f"Using PFX certificate: {pfx_full_path}")
                    return client_certificates
                else:
                    self.logger.warning(f"PFX certificate file not found: {pfx_full_path}")
            
            self.logger.info("No valid certificate configuration found")
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to setup client certificates: {e}")
            return None

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        correlation_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """Make HTTP request using Playwright API context."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        correlation_id = correlation_id or f"REQ-{uuid.uuid4()}"
        request_timeout = timeout or self.timeout
        
        request_headers = {}
        if headers:
            request_headers.update(headers)
        request_headers['CLIENT-CORRELATION-ID'] = correlation_id

        # Track request if tracker is available
        if self.tracker:
            self.tracker.add_request(method, url, request_headers, data, correlation_id)

        self.logger.info(f"[{correlation_id}] {method}: {url}")
        start_time = time.time()

        try:
            if method.upper() == "POST":
                response = self.api_request_context.post(
                    endpoint,
                    data=json.dumps(data) if data else None,
                    headers=request_headers,
                    timeout=request_timeout
                )
            elif method.upper() == "GET":
                response = self.api_request_context.get(
                    endpoint,
                    headers=request_headers,
                    timeout=request_timeout
                )
            else:
                raise NotImplementedError(f"Method {method} not implemented")

            duration = time.time() - start_time
            
            # Handle response parsing
            try:
                response_text = response.text()
                if not response_text or not response_text.strip().startswith(('{', '[')):
                    self.logger.warning(f"[{correlation_id}] Response is not valid JSON: {response_text[:200]}...")
                    response_data = {"raw_response": response_text}
                else:
                    response_data = response.json()
            except Exception as e:
                self.logger.error(f"[{correlation_id}] Failed to parse response as JSON: {e}")
                response_data = {"raw_response": response.text()}

            # Track response if tracker is available
            if self.tracker:
                self.tracker.add_response(
                    response.status,
                    dict(response.headers),
                    response_data,
                    duration,
                    correlation_id
                )

            self.logger.info(f"[{correlation_id}] Response status: {response.status}, duration: {duration:.2f}s")
            return response_data

        except PlaywrightError as e:
            self.logger.error(f"[{correlation_id}] Playwright Error: {e}")
            if self.tracker:
                self.tracker.add_error("PlaywrightError", str(e), correlation_id)
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
        """Initiate chat conversation."""
        correlation_id = correlation_id or f"INIT-{uuid.uuid4()}"
        if 'conversationId' not in request_data:
            request_data['conversationId'] = 'initial'
        
        self.logger.info(f"Initiating chat with correlation ID: {correlation_id}")
        return self._make_request(
            method='POST',
            endpoint='/api/agentic-chat/v1',
            data=request_data,
            correlation_id=correlation_id
        )

    def initiate_chat_error(
        self,
        request_data: Dict[str, Any], 
        headers: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Initiate chat for error testing scenarios."""
        if 'conversationId' not in request_data:
            request_data['conversationId'] = 'initial'
        
        return self._make_request(
            method='POST',
            endpoint='/api/agentic-chat/v1',
            data=request_data,
            headers=headers
        )

    def send_message(
        self,
        conversation_id: str,
        chat_text: str,
        action: str = "proceed",
        headers: Optional[Dict] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send message to existing conversation."""
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
            method='POST',
            endpoint='/api/agentic-chat/v1',
            data=payload,
            headers=headers,
            correlation_id=correlation_id
        )
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on the API."""
        try:
            return self._make_request(
                method='GET',
                endpoint='/health',
                correlation_id=f"HEALTH-{uuid.uuid4()}"
            )
        except Exception as e:
            self.logger.warning(f"Health check failed: {e}")
            return {"status": "unavailable", "error": str(e)}

    def close(self) -> None:
        """Clean up resources."""
        if self.api_request_context:
            try:
                self.api_request_context.dispose()
                self.logger.info("Playwright API context disposed")
            except Exception as e:
                self.logger.warning(f"Error disposing API context: {e}")
        
        if self.playwright:
            try:
                self.playwright.stop()
                self.logger.info("Playwright instance stopped")
            except Exception as e:
                self.logger.warning(f"Error stopping Playwright: {e}")


class PlaywrightAPIHelpers:
    """Helper utilities for Playwright API testing."""
    
    @staticmethod
    def wait_for_response_condition(client: ChatbotAPIClient, condition_func, max_attempts: int = 10, delay: float = 1.0):
        """Wait for API response to meet specific condition."""
        logger = get_logger(__name__)
        
        for attempt in range(max_attempts):
            try:
                health = client.health_check()
                if condition_func(health):
                    logger.info(f"Condition met after {attempt + 1} attempts")
                    return True
                
                if attempt < max_attempts - 1:
                    logger.debug(f"Condition not met, waiting {delay}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
                    
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    time.sleep(delay)
        
        logger.error(f"Condition not met after {max_attempts} attempts")
        return False
    
    @staticmethod
    def extract_conversation_id(response: Dict[str, Any]) -> Optional[str]:
        """Extract conversation ID from API response."""
        return response.get('conversationId') or response.get('conversationID')
    
    @staticmethod
    def extract_interaction_id(response: Dict[str, Any]) -> Optional[str]:
        """Extract interaction ID from API response."""
        import re
        
        # Look in various response fields
        text_fields = [
            response.get('chatResponseText', ''),
            response.get('message', ''),
            str(response.get('interactionId', '')),
            str(response)
        ]
        
        # Pattern for interaction IDs
        pattern = r'INT[E0L]-\d{6}-\w{12}'
        
        for text in text_fields:
            if text:
                match = re.search(pattern, str(text))
                if match:
                    return match.group(0)
        
        return None
