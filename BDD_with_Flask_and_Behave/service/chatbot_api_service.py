import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils.config_loader import get_config
from utils.logger_config import get_logger
import uuid
import json

logger = get_logger(__name__)

class ChatbotAPIService:
    def __init__(self):
        self.config = get_config()
        self.base_url = self.config['API_BASE_URL']
        self.session = self._create_session()

    def _create_session(self):
        """Creates a request session with retry logic."""
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    def _send_request(self, method, endpoint, headers, payload=None):
        """A generic method to send requests and handle responses."""
        url = f"{self.base_url}/{endpoint}"
        headers['Content-Type'] = 'application/json'
        
        logger.info(f"Request: {method} {url}")
        logger.debug(f"Headers: {headers}")
        logger.debug(f"Payload: {json.dumps(payload)}")

        response = self.session.request(method, url, headers=headers, json=payload)
        response.raise_for_status() # Raises an exception for bad status codes (4xx or 5xx)
        
        logger.info(f"Response Status: {response.status_code}")
        response_data = response.json()
        logger.debug(f"Response Body: {json.dumps(response_data)}")
        return response_data

    def initiate_chat(self, channel_id, data_elements, headers):
        """Handles the first call to initiate a chat with a set of data elements."""
        endpoint = "agentic-chat/v2"
        payload = {
            "channelID": channel_id,
            "conversationID": "initial",
            "dataElements": data_elements,
            "requestType": "ComplaintCapture",
            "chatText": "proceed",
            "action": "proceed"
        }
        return self._send_request("POST", endpoint, headers, payload)

    def send_message(self, conversation_id, chat_text, headers):
        """Sends a subsequent message in an ongoing conversation."""
        endpoint = "agentic-chat/v2"  # Corrected: Removed correlation ID from query parameter
        payload = {
            "conversationID": conversation_id,
            "requestType": "ComplaintCapture",
            "chatText": chat_text,
            "action": "proceed"
        }
        return self._send_request("POST", endpoint, headers, payload)
