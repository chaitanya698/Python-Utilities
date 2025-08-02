# utils/api_service.py

import requests
import logging
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any

# Import the Settings model for type hinting, assuming it's in config/settings.py
from config.settings import Settings

logger = logging.getLogger(__name__)

class ChatbotAPIService:
    """A robust service class to handle all interactions with the Chatbot API."""

    def __init__(self, config: Settings):
        """
        Initializes the service with a persistent HTTP session and certificate handling.
        
        Args:
            config (Settings): The fully loaded and processed configuration object,
                               which includes paths to the temporary PEM files if a PFX
                               was processed by the loader.
        """
        self.config = config
        self.base_url = self.config.API_BASE_URL
        self.session = self._create_session()
        logger.info(f"ChatbotAPIService initialized for base URL: {self.base_url}")

    def _create_session(self) -> requests.Session:
        """Configures and returns a requests Session object with certificate handling."""
        session = requests.Session()
        
        # Configure retry strategy for transient network errors (e.g., 502, 503, 504)
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # Configure SSL with the client-side certificate.
        # The loader.py has already processed the PFX into temporary PEM files.
        # We just need to use the paths that are now stored in the config object.
        if self.config.CERT_PEM_PATH and self.config.KEY_PEM_PATH:
            session.cert = (self.config.CERT_PEM_PATH, self.config.KEY_PEM_PATH)
            logger.info(f"Session configured with client certificate authentication using cert: {self.config.CERT_PEM_PATH}")
        else:
            logger.warning("No client certificate paths found in config. Proceeding without certificate authentication.")

        session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        return session

    def _send_request(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """A private, reusable method to send POST requests and handle responses."""
        endpoint = f"{self.base_url}/api/agentic-chat/v1"
        correlation_id = headers.get('CLIENT-CORRELATION-ID', 'N/A')
        
        logger.info(f"[{correlation_id}] Sending POST request to endpoint: {endpoint}")
        logger.debug(f"[{correlation_id}] Request Headers: {headers}")
        logger.debug(f"[{correlation_id}] Request Payload: {json.dumps(payload, indent=2)}")

        try:
            # For dev/test environments, it's common to disable SSL verification.
            # In a production-ready framework, this could be controlled by a config flag.
            response = self.session.post(endpoint, json=payload, headers=headers, timeout=45, verify=False)
            
            logger.info(f"[{correlation_id}] Received response with status: {response.status_code}")
            logger.debug(f"[{correlation_id}] Response Body: {response.text}")
            
            # This will raise an HTTPError for bad responses (4xx or 5xx)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"[{correlation_id}] HTTP error occurred: {http_err} - Response: {http_err.response.text}")
            raise
        except requests.exceptions.RequestException as req_err:
            logger.error(f"[{correlation_id}] A critical request error occurred: {req_err}")
            raise

    def initiate_chat(self, initial_request_data: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """Initiates a new chat conversation."""
        initial_request_data['conversationId'] = 'initial'
        return self._send_request(initial_request_data, headers)

    def send_message(self, conversation_id: str, chat_text: str, action: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """Sends a subsequent message in an existing conversation."""
        payload = {
            "channelID": "BBVA", # This could be parameterized if needed
            "conversationID": conversation_id,
            "requestType": "ComplaintCapture",
            "chatText": chat_text,
            "action": action
        }
        return self._send_request(payload, headers)
