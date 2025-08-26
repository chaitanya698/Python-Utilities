from playwright.sync_api import sync_playwright, APIRequestContext, Playwright, Error as PlaywrightError
import uuid
import time
from typing import Dict, Any, Optional
import json
from bdd_tests.config.settings import Settings
from bdd_tests.utils.logger_config import get_logger
from bdd_tests.utils.request_response_tracker import RequestResponseTracker


class ChatbotAPIClient:
    """Enterprise-grade API client with request/response tracking using Playwright."""
    
    def __init__(self, config: Settings, tracker: Optional[RequestResponseTracker] = None):
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.API_BASE_URL.rstrip('/')
        self.timeout = config.API_TIMEOUT * 1000  # Playwright expects milliseconds
        self.tracker = tracker
        self.playwright = sync_playwright().start()
        self.api_request_context = self._create_api_context()

    def _create_api_context(self) -> APIRequestContext:
        extra_http_headers = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'User-Agent': 'ChatbotAutomation/1.0'
        }
        
        # Handle client certificates properly
        client_certificates = None
        client_cert_path = getattr(self.config, "CERT_PFX_PATH", None)
        client_key_value = getattr(self.config, "CERT_PRD", None)
        
        # Only set up client certificates if BOTH cert path and key are provided
        if client_cert_path and client_key_value:
            try:
                with open(client_cert_path, "rb") as cert_file:
                    cert_data = cert_file.read()
                    client_certificates = [{
                        "origin": self.base_url,
                        "cert": cert_data,  # Use 'cert' instead of 'certificate'
                        "key": client_key_value.encode() if isinstance(client_key_value, str) else client_key_value
                    }]
                    self.logger.info("Client certificates loaded successfully")
            except Exception as e:
                self.logger.warning(f"Failed to load client certificates: {e}")
                client_certificates = None
        
        # Create context with or without certificates
        context_options = {
            "base_url": self.base_url,
            "extra_http_headers": extra_http_headers,
            "ignore_https_errors": not self.config.VERIFY_SSL,
            "timeout": self.timeout
        }
        
        # Only add client_certificates if they were successfully loaded
        if client_certificates:
            context_options["client_certificates"] = client_certificates
            
        return self.playwright.request.new_context(**context_options)

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        correlation_id: Optional[str] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        correlation_id = correlation_id or None
        request_timeout = timeout or self.timeout
        
        request_headers = {}
        if headers:
            request_headers.update(headers)
        request_headers['CLIENT-CORRELATION-ID'] = correlation_id

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
            
            try:
                response_text = response.text()
                if not response_text or not response_text.strip().startswith(('{', '[')):
                    self.logger.error(f"[{correlation_id}] Response is not valid JSON: {response_text}")
                    response_data = {"raw_response": response_text}
                else:
                    response_data = response.json()
            except Exception as e:
                self.logger.error(f"[{correlation_id}] Failed to parse response as JSON: {e}")
                response_data = {"raw_response": response.text()}

            if self.tracker:
                self.tracker.add_response(
                    response.status,
                    dict(response.headers),
                    response_data,
                    duration,
                    correlation_id
                )

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

    def close(self) -> None:
        if self.api_request_context:
            self.api_request_context.dispose()
        if self.playwright:
            self.playwright.stop()
        self.logger.info("API client session closed")
