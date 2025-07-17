import requests
import logging
import json
import os
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

class ChatbotAPIService:
    """A robust and reusable service to interact with the chatbot API."""

    def __init__(self, config):
        self.config = config
        self.base_url = self.config.API_BASE_URL
        self.session = self._create_session()
        self.initial_request_template = self._load_initial_request_template()
        self.logger = logging.getLogger(__name__)

    def _create_session(self):
        """Creates a requests session with retry logic and certificate handling."""
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Use the securely processed PEM certificate paths from the config
        if self.config.CERT_PEM_PATH and self.config.KEY_PEM_PATH:
            self.logger.info("Attaching client certificates to session.")
            session.cert = (self.config.CERT_PEM_PATH, self.config.KEY_PEM_PATH)
        return session

    def _load_initial_request_template(self):
        """Loads the initial request JSON from the resources folder."""
        # Correctly navigate up from utils, then into resources
        template_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'initial_request.json')
        try:
            with open(template_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Could not load 'initial_request.json' from '{template_path}': {e}")
            raise

    def initiate_chat(self, channel_id, complainant_name, headers):
        """Initiates a chat using the JSON template."""
        self.logger.info(f"Initiating chat for '{complainant_name}' on channel '{channel_id}'")
        payload = self.initial_request_template.copy()
        payload['channelId'] = channel_id
        
        # Find and update the complainant's name in the dataElements
        for element in payload.get("dataElements", []):
            if element.get("name") == "complainantFullName":
                element["value"] = complainant_name
                break
        
        self.logger.debug(f"API Request Payload: {json.dumps(payload, indent=2)}")
        
        # This is where you would make the actual API call
        # response = self.session.post(f"{self.base_url}/your_endpoint", json=payload, headers=headers)
        # response.raise_for_status()
        # return response.json()
        
        # For demonstration purposes, we'll return a mock response
        return {
            "conversationId": "mock-conv-12345",
            "chatResponseText": "When was the complaint received?"
        }

    def send_message(self, conversation_id, chat_text, headers):
        """Sends a follow-up message in an existing conversation."""
        self.logger.info(f"Sending message to conversation '{conversation_id}': '{chat_text}'")
        payload = {"chatText": chat_text} # Example payload
        
        # Make the actual API call here
        # response = self.session.post(f"{self.base_url}/endpoint/{conversation_id}", json=payload, headers=headers)
        # response.raise_for_status()
        # return response.json()
        
        # Mock response for demonstration
        if "10/07/2024" in chat_text:
            return {"chatResponseText": "Select the account or reference this complaint is regarding."}
        else:
            return {"chatResponseText": "I have received your message."}