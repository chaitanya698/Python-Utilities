import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import uuid
import logging

class APIClient:
    def __init__(self, config):
        self.base_url = config['base_url']
        self.session = self._create_session()
        self.conversation_id = None # Will be set after the first call

    def _create_session(self):
        """Creates a requests session with retry logic and default headers."""
        session = requests.Session()
        # Add a unique correlation ID for each test run
        session.headers.update({
            'CLIENT_CORRELATION_ID': f'test-run-{uuid.uuid4()}',
            'Content-Type': 'application/json'
        })
        # Add retry logic for robustness
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount('http://', HTTPAdapter(max_retries=retries))
        return session

    def start_complaint(self, initial_data):
        """Initiates the conversation."""
        response = self.session.post(f"{self.base_url}/agentic-chat", json=initial_data)
        response.raise_for_status()
        data = response.json()
        # Capture the conversationID for subsequent requests
        self.conversation_id = data.get('conversationID')
        logging.info(f"Started conversation with ID: {self.conversation_id}")
        return data

    def send_response(self, user_response):
        """Sends a follow-up response in the same conversation."""
        if not self.conversation_id:
            raise ValueError("Conversation has not been started.")
        
        payload = {
            "conversationID": self.conversation_id,
            "userInput": user_response
        }
        response = self.session.post(f"{self.base_url}/agentic-chat", json=payload)
        response.raise_for_status()
        return response.json()