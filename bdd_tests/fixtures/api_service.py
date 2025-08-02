# fixtures/api.py

import pytest
import requests
import logging
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any

from config.config_loader import config

logger = logging.getLogger(__name__)

class ChatbotAPIService:
    """A robust service class to handle all interactions with the Chatbot API."""

    def __init__(self, base_url: str):
        """Initializes the service with a persistent HTTP session and retry logic."""
        self.base_url = base_url
        self.session = self._create_session()
        logger.info(f"ChatbotAPIService initialized for base URL: {self.base_url}")

    def _create_session(self) -> requests.Session:
        """Configures and returns a requests Session object."""
        session = requests.Session()
        # Configure retry strategy for transient network errors
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        # The certificate handling logic can be added here if needed,
        # using the paths from the loaded config object.
        # For this example, we'll omit the complex PFX logic for clarity.
        # if config.app.cert_pem_path and config.app.key_pem_path:
        #     session.cert = (config.app.cert_pem_path, config.app.key_pem_path)
        #     logger.info("Session configured with client certificate authentication.")

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
            response = self.session.post(endpoint, json=payload, headers=headers, timeout=45)
            logger.info(f"[{correlation_id}] Received response with status: {response.status_code}")
            logger.debug(f"[{correlation_id}] Response Body: {response.text}")
            
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
            "channelID": "BBVA",
            "conversationID": conversation_id,
            "requestType": "ComplaintCapture",
            "chatText": chat_text,
            "action": action
        }
        return self._send_request(payload, headers)

@pytest.fixture(scope="session")
def api_service() -> ChatbotAPIService:
    """
    Provides a single, session-scoped instance of the ChatbotAPIService.
    This fixture ensures that the HTTP session is created only once per test run,
    improving efficiency.
    """
    return ChatbotAPIService(base_url=config.app.api_base_url)

