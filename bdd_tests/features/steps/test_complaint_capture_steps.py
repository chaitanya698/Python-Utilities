# steps/test_complaint_capture_steps.py

from pytest_bdd import scenario, given, when, then, parsers
import uuid
import logging
import json
import re
from pathlib import Path

from fixtures.api import ChatbotAPIService
from fixtures.db import DBUtils

# The test scenario function that will be parameterized by the hook in conftest.py
@scenario('../features/complaint_capture.feature', 'Verify end-to-end complaint capture process using data from an external source')
def test_complaint_workflow():
    """This test function is a placeholder. The actual tests are generated dynamically."""
    pass

logger = logging.getLogger(__name__)

# --- Step Definitions ---

@given('the chatbot API is available and test data is loaded')
def setup_api_and_data_context(chatbot_context: dict, api_service: ChatbotAPIService, test_data_row: dict):
    """
    Initializes the context for the test run.
    - Sets up API service and headers.
    - Loads the test data for the current scenario run into the context.
    """
    chatbot_context['api_service'] = api_service
    chatbot_context['headers'] = {
        'Content-Type': 'application/json',
        'Accept': 'application/json',
        'CLIENT-CORRELATION-ID': f"{test_data_row.get('test_case_id', 'TC-UNKNOWN')}-{uuid.uuid4()}"
    }
    # Load the current row of test data into the context for later steps to use
    chatbot_context['current_test_data'] = test_data_row
    logger.info(f"[{chatbot_context['headers']['CLIENT-CORRELATION-ID']}] Scenario context initialized for test case: {test_data_row.get('test_case_id')}")

@when('I send the initial complaint request')
def send_initial_request(chatbot_context: dict):
    """Loads and sends the initial request payload."""
    api = chatbot_context['api_service']
    headers = chatbot_context['headers']
    test_data = chatbot_context['current_test_data']
    
    request_file = test_data['initial_request_file']
    request_path = Path(__file__).parent.parent / "data" / request_file
    
    logger.info(f"Loading initial request payload from: {request_path}")
    with open(request_path, 'r') as f:
        initial_request_data = json.load(f)

    response = api.initiate_chat(initial_request_data=initial_request_data, headers=headers)
    
    chatbot_context['response'] = response
    if response and 'conversationId' in response:
        chatbot_context['conversationId'] = response['conversationId']
        logger.info(f"Received conversationID: {response['conversationId']}")

@then('the API response should be successful and contain a valid conversation ID')
def check_api_response_and_conv_id(chatbot_context: dict):
    """Verifies a successful response and a valid conversation ID format."""
    response = chatbot_context.get('response')
    assert response, "Did not receive a response from the API."
    assert 'conversationId' in response, "Response JSON does not contain 'conversationId'."
    
    conversation_id = response['conversationId']
    pattern = r'^CVD-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
    assert re.match(pattern, conversation_id), f"Conversation ID '{conversation_id}' does not match the expected pattern."
    logger.info(f"Validated successful response with conversationId: {conversation_id}")

@then('the conversation ID must exist in the database')
def check_conversation_id_in_database(chatbot_context: dict, db_utils: DBUtils):
    """Verifies the conversation ID is present in the database."""
    conversation_id = chatbot_context.get('conversationId')
    assert conversation_id, "Cannot verify database without a conversationId."
    
    chat_history = db_utils.get_chat_history(conversation_id)
    assert len(chat_history) > 0, f"Conversation ID '{conversation_id}' not found in the database."
    logger.info(f"Successfully verified conversation ID '{conversation_id}' exists in the database.")

@then('the initial response action and text should be as expected')
def check_initial_response(chatbot_context: dict):
    """Validates the action and text of the initial API response."""
    response = chatbot_context['response']
    test_data = chatbot_context['current_test_data']
    
    # Validate action
    expected_action = {"action": "proceed", "type": "button", "label": test_data['expected_initial_action_label']}
    assert 'actions' in response and expected_action in response['actions'], \
        f"Expected action '{expected_action}' not found in response: {response.get('actions')}"

    # Validate text
    actual_text = response.get('chatResponseText')
    expected_text = test_data['expected_initial_response_text']
    assert actual_text == expected_text, f"Expected chat text '{expected_text}', but got '{actual_text}'."
    logger.info("Validated initial API response action and text.")

def _send_user_response(chatbot_context: dict, chat_text: str, action: str = "proceed"):
    """Helper function to send a user message to the API."""
    api = chatbot_context['api_service']
    headers = chatbot_context['headers']
    conversation_id = chatbot_context['conversationId']
    
    logger.info(f"User responding with text: '{chat_text}' and action: '{action}'")
    response = api.send_message(
        conversation_id=conversation_id,
        chat_text=chat_text,
        action=action,
        headers=headers
    )
    chatbot_context['response'] = response

@when('the user responds with the complaint date')
def user_responds_with_date(chatbot_context: dict):
    date_text = chatbot_context['current_test_data']['complaint_date']
    _send_user_response(chatbot_context, date_text)

@then('the API response should prompt for the method of complaint')
def check_prompt_for_method(chatbot_context: dict):
    response = chatbot_context['response']
    assert 'How the complaint received?' in response.get('chatResponseText', '')
    logger.info("Validated prompt for complaint method.")

@when('the user responds with the method of complaint')
def user_responds_with_method(chatbot_context: dict):
    method_text = chatbot_context['current_test_data']['complaint_method']
    _send_user_response(chatbot_context, method_text)

@then('the API response should prompt for the account number')
def check_prompt_for_account(chatbot_context: dict):
    response = chatbot_context['response']
    assert 'Select the account' in response.get('chatResponseText', '')
    logger.info("Validated prompt for account number.")

@when('the user responds with the account number')
def user_responds_with_account(chatbot_context: dict):
    account_text = chatbot_context['current_test_data']['account_number']
    _send_user_response(chatbot_context, account_text)

@then('the API response should prompt for complaint details')
def check_prompt_for_details(chatbot_context: dict):
    response = chatbot_context['response']
    assert 'provide more details' in response.get('chatResponseText', '')
    logger.info("Validated prompt for complaint details.")

@when('the user responds with the complaint details')
def user_responds_with_details(chatbot_context: dict):
    details_text = chatbot_context['current_test_data']['complaint_details']
    _send_user_response(chatbot_context, details_text)

@then('the API response should contain a valid chat text')
def check_for_any_chat_text(chatbot_context: dict):
    response = chatbot_context['response']
    assert 'chatResponseText' in response and response['chatResponseText']
    logger.info("Validated that response has a non-empty chatResponseText.")

@when('the user provides a final summary comment')
def user_provides_summary_comment(chatbot_context: dict):
    summary_text = chatbot_context['current_test_data']['final_summary_comment']
    _send_user_response(chatbot_context, summary_text)

@then('the API response should ask for clarification')
def check_ask_for_clarification(chatbot_context: dict):
    response = chatbot_context['response']
    assert 'Final summary' in response.get('chatResponseText', '')
    assert 'revise,clarify' in {act['action'] for act in response.get('actions', [])}
    logger.info("Validated prompt for clarification.")

@when('the user confirms the summary')
def user_confirms_summary(chatbot_context: dict):
    _send_user_response(chatbot_context, "Continue", action="proceed")

@then('the API response should ask for contact willingness')
def check_ask_for_contact(chatbot_context: dict):
    response = chatbot_context['response']
    assert 'willing to be contacted' in response.get('chatResponseText', '')
    logger.info("Validated prompt for contact willingness.")

@when('the user responds with their contact willingness')
def user_responds_with_contact_willingness(chatbot_context: dict):
    willingness = chatbot_context['current_test_data']['contact_willingness_response']
    _send_user_response(chatbot_context, willingness, action="proceed")

@then('the API response should prompt to submit the complaint')
def check_prompt_for_submission(chatbot_context: dict):
    response = chatbot_context['response']
    assert 'review the complaint summary' in response.get('chatResponseText', '')
    assert 'Submit complaint' in {act['label'] for act in response.get('actions', [])}
    logger.info("Validated prompt for submission.")

@when('the user submits the complaint')
def user_submits_complaint(chatbot_context: dict):
    _send_user_response(chatbot_context, "Submit", action="proceed")

@then('the final response should contain a confirmation and a valid Interaction ID')
def check_final_response(chatbot_context: dict):
    response = chatbot_context['response']
    actual_text = response.get('chatResponseText', '')
    
    assert 'Thanks for submitting the complaint' in actual_text
    
    search_pattern = r'Interaction ID:\s*([EDL]\d{10})'
    assert re.search(search_pattern, actual_text), f"No valid Interaction ID found in response: {actual_text}"
    
    logger.info("Validated final response with confirmation text and Interaction ID.")
