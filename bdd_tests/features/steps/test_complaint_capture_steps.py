import re
import uuid
from datetime import datetime
from typing import Dict, Any

from pytest_bdd import scenario, given, when, then, parsers

from utils.logger_config import get_logger
from utils.helpers import TestHelpers

# Initialize logger
logger = get_logger(__name__)


@scenario(
    '../complaint_capture.feature',
    'Verify end-to-end complaint capture process using data from an external source'
)
def test_complaint_workflow():
    """Parameterized test for complaint workflow."""
    pass


# --- Given Steps ---
@given('the chatbot API is available and test data is loaded')
def setup_test_context(
    given_api_is_available: Dict[str, Any], 
    given_test_data_loaded: Dict[str, Any]
) -> Dict[str, Any]:
    """Initialize test context with API and data."""
    context = given_api_is_available
    context.update(given_test_data_loaded)
    
    # Set correlation ID
    test_case_id = context['test_data'].get('test_case_id', 'UNKNOWN')
    context['correlation_id'] = TestHelpers.generate_correlation_id(test_case_id)
    
    logger.info(
        f"Test initialized - Case: {test_case_id}, "
        f"Correlation: {context['correlation_id']}"
    )
    
    return context


# --- When Steps ---
@when('I send the initial complaint request')
def send_initial_request(
    test_context: Dict[str, Any], 
    data_loader
) -> None:
    """Send initial complaint request."""
    api_client = test_context['api_client']
    test_data = test_context['test_data']
    
    # Load request template
    request_file = test_data.get('initial_request_file', 'initial_request.json')
    initial_request = data_loader.load_json(request_file)
    
    logger.info(f"Sending initial request for case: {test_data.get('test_case_id')}")
    
    try:
        # Send request
        response = api_client.initiate_chat(
            request_data=initial_request,
            correlation_id=test_context['correlation_id']
        )
        
        test_context['response'] = response
        test_context['conversation_id'] = response.get('conversationId')
        
    except Exception as e:
        logger.error(f"Failed to send initial request: {e}")
        raise


@when(parsers.parse('the user responds with the {field}'))
def user_responds_with_field(
    test_context: Dict[str, Any], 
    field: str
) -> None:
    """Handle user response for various fields."""
    field_mapping = {
        'complaint date': 'complaint_date',
        'method of complaint': 'complaint_method',
        'account number': 'account_number',
        'complaint details': 'complaint_details',
        'contact willingness': 'contact_willingness_response'
    }
    
    data_field = field_mapping.get(field)
    if not data_field:
        raise ValueError(f"Unknown field: {field}")
    
    response_text = test_context['test_data'].get(data_field, '')
    if not response_text:
        logger.warning(f"No data found for field: {field}")
    
    _send_message(test_context, response_text)


@when('the user provides a final summary comment')
def user_provides_summary(test_context: Dict[str, Any]) -> None:
    """User provides summary comment."""
    summary_text = test_context['test_data'].get(
        'final_summary_comment', 
        'This is my complaint summary'
    )
    _send_message(test_context, summary_text)


@when('the user confirms the summary')
def user_confirms_summary(test_context: Dict[str, Any]) -> None:
    """User confirms the summary."""
    _send_message(test_context, "Continue", action="proceed")


@when('the user submits the complaint')
def user_submits_complaint(test_context: Dict[str, Any]) -> None:
    """User submits the complaint."""
    _send_message(test_context, "Submit", action="proceed")


# --- Then Steps ---
@then('the API response should be successful and contain a valid conversation ID')
def verify_initial_response(test_context: Dict[str, Any]) -> None:
    """Verify initial API response."""
    response = test_context.get('response')
    
    assert response, "No response received from API"
    assert 'conversationId' in response, "Response missing conversationId"
    
    # Verify conversation ID format
    conv_id = response['conversationId']
    assert TestHelpers.validate_conversation_id(conv_id), \
        f"Invalid conversation ID format: {conv_id}"
    
    logger.info(f"Verified conversation ID: {conv_id}")


@then('the conversation ID must exist in the database')
def verify_conversation_in_db(
    test_context: Dict[str, Any], 
    db_utils
) -> None:
    """Verify conversation exists in database."""
    conv_id = test_context.get('conversation_id')
    
    if not conv_id:
        raise ValueError("No conversation ID found in test context")
    
    exists = db_utils.verify_conversation_exists(conv_id)
    assert exists, f"Conversation {conv_id} not found in database"
    
    logger.info(f"Verified conversation {conv_id} exists in database")


@then('the initial response action and text should be as expected')
def verify_initial_action_and_text(test_context: Dict[str, Any]) -> None:
    """Verify expected action and response text."""
    response = test_context.get('response', {})
    test_data = test_context.get('test_data', {})
    
    # Verify action
    expected_action_label = test_data.get('expected_initial_action_label')
    if expected_action_label:
        expected_action = {
            "action": "proceed",
            "type": "button",
            "label": expected_action_label
        }
        
        assert 'actions' in response, "Response missing actions"
        assert expected_action in response['actions'], \
            f"Expected action not found. Got: {response['actions']}"
    
    # Verify text
    expected_text = test_data.get('expected_initial_response_text')
    if expected_text:
        actual_text = response.get('chatResponseText', '')
        assert actual_text == expected_text, \
            f"Expected: '{expected_text}', Got: '{actual_text}'"
    
    logger.info("Verified initial response action and text")


@then(parsers.parse('the API response should prompt for {expected_prompt}'))
def verify_prompt(
    test_context: Dict[str, Any], 
    expected_prompt: str
) -> None:
    """Verify API prompts for expected information."""
    prompt_patterns = {
        'the method of complaint': 'How the complaint received?',
        'the account number': 'Select the account',
        'complaint details': 'provide more details',
        'clarification': 'Final summary',
        'contact willingness': 'willing to be contacted',
        'to submit the complaint': 'review the complaint summary'
    }
    
    pattern = prompt_patterns.get(expected_prompt)
    if not pattern:
        raise ValueError(f"Unknown prompt type: {expected_prompt}")
    
    response = test_context.get('response', {})
    response_text = response.get('chatResponseText', '')
    
    assert pattern in response_text, \
        f"Expected prompt '{pattern}' not found in: {response_text}"
    
    logger.info(f"Verified prompt for: {expected_prompt}")


@then('the API response should contain a valid chat text')
def verify_chat_text_exists(test_context: Dict[str, Any]) -> None:
    """Verify response contains chat text."""
    response = test_context.get('response', {})
    
    assert 'chatResponseText' in response, "Response missing chatResponseText"
    assert response['chatResponseText'], "Chat text is empty"
    
    logger.info("Verified chat text exists in response")


@then('the API response should ask for clarification')
def verify_clarification_request(test_context: Dict[str, Any]) -> None:
    """Verify API asks for clarification."""
    response = test_context.get('response', {})
    
    assert 'Final summary' in response.get('chatResponseText', ''), \
        "Clarification text not found in response"
    
    # Check for clarify action
    actions = {act['action'] for act in response.get('actions', [])}
    assert 'revise,clarify' in actions or 'clarify' in actions, \
        f"Clarify action not found. Got actions: {actions}"
    
    logger.info("Verified clarification request")


@then('the final response should contain a confirmation and a valid Interaction ID')
def verify_final_response(test_context: Dict[str, Any]) -> None:
    """Verify final submission response."""
    response = test_context.get('response', {})
    response_text = response.get('chatResponseText', '')
    
    # Verify confirmation message
    assert 'Thanks for submitting the complaint' in response_text, \
        "Confirmation message not found"
    
    # Verify interaction ID format
    pattern = r'Interaction ID:\s*([EDL]\d{10})'
    match = re.search(pattern, response_text)
    
    assert match, f"No valid Interaction ID found in: {response_text}"
    
    interaction_id = match.group(1)
    test_context['interaction_id'] = interaction_id
    
    assert TestHelpers.validate_interaction_id(interaction_id), \
        f"Invalid interaction ID format: {interaction_id}"
    
    logger.info(f"Verified final response with Interaction ID: {interaction_id}")


# --- Helper Functions ---
def _send_message(
    test_context: Dict[str, Any], 
    message: str, 
    action: str = "proceed"
) -> None:
    """Helper to send message to API."""
    api_client = test_context['api_client']
    conversation_id = test_context.get('conversation_id')
    
    if not conversation_id:
        raise ValueError("No conversation ID found in test context")
    
    logger.debug(f"Sending message: '{message}' with action: '{action}'")
    
    try:
        response = api_client.send_message(
            conversation_id=conversation_id,
            message=message,
            action=action,
            correlation_id=test_context['correlation_id']
        )
        
        test_context['response'] = response
        
    except Exception as e:
        logger.error(f"Failed to send message: {e}")
        raise