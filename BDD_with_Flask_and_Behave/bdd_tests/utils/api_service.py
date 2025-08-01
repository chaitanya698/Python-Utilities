# bdd_tests/utils/api_service.py

import requests
import logging
import json
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, Any

# Note: config is now imported from the loader, which ensures it's processed.
from bdd_tests.config.loader import config

logger = logging.getLogger(__name__)

class ChatbotAPIService:
    """A robust service class to handle all interactions with the Chatbot API."""

    def __init__(self):
        """Initializes the service with a persistent HTTP session and retry logic."""
        self.base_url = config.API_BASE_URL
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Configures and returns a requests Session object."""
        session = requests.Session()
        retry_strategy = Retry(total=3, backoff_factor=1, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("https://", adapter)

        # Configure SSL with the client-side certificate from the processed config
        session.verify = False  # Disable normal verification for custom cert
        if config.CERT_PEM_PATH and config.KEY_PEM_PATH:
            session.cert = (config.CERT_PEM_PATH, config.KEY_PEM_PATH)
            logger.info("Session created with client certificate authentication.")

        session.headers.update({'Content-Type': 'application/json', 'Accept': 'application/json'})
        return session

    def _send_request(self, payload: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """A private, reusable method to send POST requests and handle responses."""
        endpoint = f"{self.base_url}/api/agentic-chat/v1"
        correlation_id = headers.get('CLIENT-CORRELATION-ID', 'N/A')
        
        logger.info(f"[{correlation_id}] Sending request to endpoint: {endpoint}")
        logger.debug(f"[{correlation_id}] Request Headers: {headers}")
        logger.debug(f"[{correlation_id}] Request Payload: {json.dumps(payload, indent=2)}")

        try:
            response = self.session.post(endpoint, json=payload, headers=headers, timeout=30)
            logger.info(f"[{correlation_id}] Received response with status: {response.status_code}")
            logger.debug(f"[{correlation_id}] Response Body: {response.text}")
            
            response.raise_for_status()  # Raises an HTTPError for bad responses (4xx or 5xx)
            return response.json()
        except requests.exceptions.HTTPError as http_err:
            logger.error(f"[{correlation_id}] HTTP error occurred: {http_err} - Response: {http_err.response.text}")
            raise
        except requests.exceptions.RequestException as req_err:
            logger.error(f"[{correlation_id}] A critical request error occurred: {req_err}")
            raise

    def initiate_chat(self, initial_request_data: Dict[str, Any], headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Initiates a new chat conversation.
        This method constructs the initial payload and uses the generic send method.
        """
        if not initial_request_data.get('channelId') or not initial_request_data.get('requestType'):
            raise ValueError("channelId and requestType are required for initiating a chat.")
        
        # Ensure the conversationID is set to 'initial' for the first request
        initial_request_data['conversationID'] = 'initial'
        return self._send_request(initial_request_data, headers)

    def send_message(self, conversation_id: str, chat_text: str, action: str, headers: Dict[str, str]) -> Dict[str, Any]:
        """
        Sends a subsequent message in an existing conversation.
        """
        payload = {
            "channelID": "BBVA",  # Assuming this is constant for subsequent messages
            "conversationID": conversation_id,
            "requestType": "ComplaintCapture", # Assuming constant
            "chatText": chat_text,
            "action": action
        }
        return self._send_request(payload, headers)
