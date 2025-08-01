# bdd_tests/features/steps/test_complaint_capture_steps.py

from pytest_bdd import scenario, given, when, then, parsers
import uuid
import logging
import json
import re
from pathlib import Path

from bdd_tests.utils.api_service import ChatbotAPIService
from bdd_tests.utils.db_utils import DBUtils

logger = logging.getLogger(__name__)

# --- Scenario Definition ---

@scenario('../complaint_capture.feature', 'Verify initial complaint request creates a conversation and is saved to the database')
def test_complaint_workflow():
    """Binds the Gherkin feature scenario to the test runner."""
    pass

# --- Step Definitions ---

@given('the chatbot API is available')
def setup_api_context(chatbot_context: dict, api_service: ChatbotAPIService):
    """Prepares the context for the test with the API service and request headers."""
    chatbot_context['api_service'] = api_service
    chatbot_context['headers'] = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'CLIENT-CORRELATION-ID': str(uuid.uuid4())
    }
    logger.info(f"[{chatbot_context['headers']['CLIENT-CORRELATION-ID']}] Scenario context initialized.")

@when(parsers.parse('I send the initial complaint request from "{json_file}"'))
def send_initial_request(chatbot_context: dict, json_file: str):
    """Loads a request payload from a JSON file and sends it to the API."""
    api = chatbot_context['api_service']
    headers = chatbot_context['headers']
    
    # Load request data from the specified file in the resources directory
    resources_path = Path(__file__).parent.parent.parent / "resources"
    request_path = resources_path / json_file
    
    logger.info(f"Loading initial request payload from: {request_path}")
    with open(request_path, 'r') as f:
        initial_request_data = json.load(f)

    response = api.initiate_chat(initial_request_data=initial_request_data, headers=headers)
    
    # Store the full response and the conversation ID for subsequent steps
    chatbot_context['response'] = response
    if response and 'conversationId' in response:
        chatbot_context['conversationId'] = response['conversationId']
        logger.info(f"Received conversationId: {response['conversationId']}")

@then('the API response should be successful')
def check_api_response(chatbot_context: dict):
    """Verifies that the API returned a response."""
    assert chatbot_context.get('response'), "Did not receive a response from the API."
    logger.info("API response was received successfully.")

@then('the response should contain a valid conversation ID')
def check_conversation_id(chatbot_context: dict):
    """Checks for the presence and validity of the conversation ID."""
    response = chatbot_context['response']
    assert 'conversationId' in response, "Response JSON does not contain 'conversationId'."
    conversation_id = response['conversationId']
    assert isinstance(conversation_id, str) and len(conversation_id) > 0, "conversationId is not a valid string."
    logger.info(f"Validated conversationId: {conversation_id}")

@then(parsers.parse('the response action should be to "{action}" with label "{label}"'))
def check_response_action(chatbot_context: dict, action: str, label: str):
    """Validates the action returned by the API."""
    response = chatbot_context['response']
    assert 'actions' in response and response['actions'], "Response does not contain 'actions'."
    expected_action = {"action": action, "type": "button", "label": label}
    assert expected_action in response['actions'], f"Expected action '{expected_action}' not found in response actions."
    logger.info(f"Validated API action: {expected_action}")

@then(parsers.parse('the chat response text should be "{expected_text}"'))
def check_chat_response_text(chatbot_context: dict, expected_text: str):
    """Validates the text message returned by the chatbot."""
    actual_text = chatbot_context['response'].get('chatResponseText')
    assert actual_text == expected_text, f"Expected chat text '{expected_text}', but got '{actual_text}'."
    logger.info("Validated chat response text.")

@then(parsers.parse('the conversation ID should follow the pattern "{pattern_desc}"'))
def check_conversation_id_pattern(chatbot_context: dict, pattern_desc: str):
    """Validates the conversation ID against a regex pattern."""
    conversation_id = chatbot_context['conversationId']
    # A more robust regex for the given pattern
    pattern = r'^CVD-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    assert re.match(pattern, conversation_id), f"Conversation ID '{conversation_id}' does not match the expected pattern."
    logger.info("Validated conversation ID pattern.")

@then('the initial chat interaction should be saved in the database')
def check_db_chat_history(chatbot_context: dict, db_utils: DBUtils):
    """Connects to the database and verifies that the chat history was persisted."""
    conversation_id = chatbot_context.get('conversationId')
    assert conversation_id, "Cannot verify database without a conversationId from a previous step."
    
    chat_history = db_utils.get_chat_history(conversation_id)
    assert len(chat_history) > 0, f"No chat history found in the database for conversation_id: {conversation_id}"
    
    # Example validation: check if the first message from the bot is correct
    initial_bot_message = chatbot_context['response']['chatResponseText']
    # Assuming the bot's response is the first entry
    first_db_entry = chat_history[0]
    assert first_db_entry['message_text'] == initial_bot_message, "The message saved in the database does not match the API response."
    
    logger.info(f"Successfully validated that chat history for conversation {conversation_id} was saved to the database.")
