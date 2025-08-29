import json
import re
import logging
from typing import Dict, Any, Optional

from pytest_bdd import scenario, given, when, then, parsers

from bdd_tests.utils.helpers import TestHelpers
from bdd_tests.utils.data_loader import DataLoader

# Initialize the logger for this module
logger = logging.getLogger(__name__)


# --- Scenario Definition ---

@scenario(
    '../features/complaint_capture_end_to_end_test.feature',
    'Execute complaint capture workflow for test case "<test_case_id>"'
)
def test_complaint_workflow(test_data_row):
    """
    This function links the test scenario in the .feature file with the Python test runner.
    """
    pass


# --- Helper Functions ---

def clean_chat_text(raw_text: Any) -> Optional[str]:
    """
    Clean and validate chatText from CSV data.
    
    Handles various edge cases:
    - Removes surrounding quotes from CSV data
    - Strips whitespace
    - Handles None, empty strings, and placeholder values
    
    Returns:
        Cleaned text if valid, None if should be skipped
    """
    if raw_text is None:
        return None
    
    # Convert to string and strip whitespace
    text = str(raw_text).strip()
    
    # Handle empty or placeholder values
    if not text or text.lower() in ['', 'n/a', 'null', 'none', 'nan', '""', "''", '""', "''", 'empty']:
        return None
    
    # Remove surrounding quotes if they exist (common CSV issue)
    if len(text) >= 2:
        if (text.startswith('"') and text.endswith('"')) or (text.startswith("'") and text.endswith("'")):
            text = text[1:-1].strip()
    
    # Double-check after quote removal
    if not text or text.lower() in ['', 'n/a', 'null', 'none', 'nan']:
        return None
    
    return text


def send_message_if_text_available(test_context: Dict[str, Any], column_name: str):
    """
    Enhanced helper that checks if a value exists for a given step in the CSV data.
    If it exists and is valid, it sends the value to the API. Otherwise, it logs a skip.
    It also captures the request and response for reporting and sets execution flag.
    """
    # Clear previous step execution state
    test_context['step_executed'] = False
    test_context['step_skipped'] = False
    test_context['request'] = None
    test_context['response'] = None
    
    # Get and clean the chatText value
    raw_chat_text = test_context.get('workflow_data', {}).get(column_name)
    chat_text = clean_chat_text(raw_chat_text)

    if chat_text is None:
        logger.info(f"SKIPPED step for column '{column_name}' - no valid chatText data found (raw: '{raw_chat_text}')")
        test_context['step_skipped'] = True
        return

    api_client = test_context['api_client']
    conversation_id = test_context.get('conversation_id')
    correlation_id = test_context.get('correlation_id')
    
    if not conversation_id:
        logger.error(f"Cannot execute step '{column_name}' - no conversation ID available")
        test_context['step_skipped'] = True
        return
    
    request_payload = {
        "conversationID": conversation_id,
        "chatText": chat_text,
        "action": "proceed"
    }
    
    try:
        logger.info(f"EXECUTING step for column '{column_name}' with chatText: '{chat_text}'")
        
        response = api_client.send_message(
            conversation_id=conversation_id,
            chat_text=chat_text,
            action="proceed",
            correlation_id=correlation_id
        )
        
        # Mark step as executed and store results
        test_context['step_executed'] = True
        test_context['last_response'] = response
        test_context['request'] = request_payload
        test_context['response'] = response
        
        logger.info(f"SUCCESS: Step '{column_name}' executed successfully")
        
    except Exception as e:
        logger.error(f"FAILED: API call failed for step '{column_name}': {e}")
        test_context['step_executed'] = True  # Step was attempted
        test_context['request'] = request_payload
        test_context['response'] = {"error": str(e)}
        raise


def verify_response_if_executed(test_context: Dict[str, Any], expected_key: str):
    """
    Enhanced verification that checks if the step was actually executed before validating.
    Uses explicit execution flag rather than checking for request presence.
    """
    # Check if the preceding step was skipped
    if test_context.get('step_skipped', False):
        logger.info(f"SKIPPED verification for '{expected_key}' - preceding step was skipped due to missing data")
        return
    
    # Check if the preceding step was executed
    if not test_context.get('step_executed', False):
        logger.info(f"SKIPPED verification for '{expected_key}' - preceding step was not executed")
        return

    # Proceed with verification
    response = test_context.get('last_response', {})
    expected_responses = test_context.get('expected_responses', {})
    expected_text = expected_responses.get(expected_key)

    if not expected_text:
        logger.warning(f"No expected response found for key '{expected_key}' in the loaded JSON data - skipping verification")
        return

    # Verify response contains expected text
    actual_text = response.get('chatResponseText', '')
    
    if not actual_text:
        logger.error(f"API response does not contain 'chatResponseText' field. Full response: {response}")
        assert False, f"API response missing 'chatResponseText' field for verification of '{expected_key}'"
    
    # Perform case-insensitive partial match for more robust validation
    if expected_text.lower() not in actual_text.lower():
        logger.error(f"Expected text not found in response")
        logger.error(f"Expected (key='{expected_key}'): '{expected_text}'")
        logger.error(f"Actual response: '{actual_text}'")
        assert False, f"Expected response key '{expected_key}' with value '{expected_text}' not found in actual response: '{actual_text}'"
    
    logger.info(f"VERIFIED: Response contains expected content for key '{expected_key}'")


def validate_workflow_data(test_context: Dict[str, Any]):
    """Validate and log workflow data availability."""
    workflow_data = test_context.get('workflow_data', {})
    available_steps = []
    
    for i in range(1, 12):
        column_name = f'chatText{i}'
        raw_value = workflow_data.get(column_name)
        cleaned_value = clean_chat_text(raw_value)
        
        if cleaned_value is not None:
            available_steps.append(f"{column_name}: '{cleaned_value[:50]}{'...' if len(cleaned_value) > 50 else ''}'")
    
    logger.info(f"Available workflow steps for test case: {len(available_steps)}")
    for step in available_steps:
        logger.info(f"  - {step}")


# --- Given Steps ---

@given('the chatbot API is available and test data is loaded')
def setup_test_context(given_api_is_available: Dict[str, Any]) -> Dict[str, Any]:
    return given_api_is_available

@given(parsers.parse('the expected responses are loaded from "{json_file}"'))
def load_expected_responses(test_context: Dict[str, Any], json_file: str):
    data_loader = DataLoader()
    try:
        test_context['expected_responses'] = data_loader.load_json(json_file, from_resources=True)
        logger.info(f"Successfully loaded expected responses from '{json_file}'")
    except FileNotFoundError:
        logger.warning(f"Expected responses file not found: '{json_file}' - continuing without expected response validation")
        test_context['expected_responses'] = {}
    except Exception as e:
        logger.error(f"Failed to load expected responses from '{json_file}': {e}")
        test_context['expected_responses'] = {}

@given(parsers.parse('I have test case "{test_case_id}" with data from CSV'))
def load_test_case_data(test_context: Dict[str, Any], test_case_id: str, test_data_row: Dict[str, Any]):
    test_context['test_case_id'] = test_case_id
    test_context['csv_data'] = test_data_row
    test_context['correlation_id'] = TestHelpers.generate_correlation_id(test_case_id)
    
    # Build workflow data with cleaned chatText values
    workflow_data = {}
    for i in range(1, 12):
        column_name = f'chatText{i}'
        raw_value = test_data_row.get(column_name)
        workflow_data[column_name] = raw_value  # Keep raw value for processing in steps
    
    test_context['workflow_data'] = workflow_data
    
    # Log available data for debugging
    validate_workflow_data(test_context)
    logger.info(f"Loaded data for test case: {test_case_id} with correlation ID: {test_context['correlation_id']}")

# --- When Steps ---

@when('I send the initial complaint request')
def send_initial_request(test_context: Dict[str, Any]):
    api_client = test_context['api_client']
    csv_data = test_context['csv_data']
    initial_request_file = csv_data.get('initial_request_file', 'initial_request.json')
    
    data_loader = DataLoader()
    
    try:
        request_data = data_loader.load_json(initial_request_file, from_resources=False)
    except FileNotFoundError:
        # Fallback to resources directory if not found in data directory
        try:
            request_data = data_loader.load_json(initial_request_file, from_resources=True)
            logger.info(f"Using initial request from resources: {initial_request_file}")
        except FileNotFoundError:
            logger.error(f"Initial request file not found: {initial_request_file}")
            raise
    
    correlation_id = test_context['correlation_id']
    
    # Ensure conversationId is set correctly
    if 'conversationId' not in request_data:
        request_data['conversationId'] = 'initial'
    
    try:
        response = api_client.initiate_chat(request_data, correlation_id)
        
        test_context['initial_response'] = response
        test_context['last_response'] = response
        test_context['conversation_id'] = response.get('conversationID') or response.get('conversationId')
        test_context['request'] = request_data
        test_context['response'] = response
        test_context['step_executed'] = True
        
        logger.info(f"Initial request sent successfully. Conversation ID: {test_context['conversation_id']}")
        
    except Exception as e:
        logger.error(f"Initial request failed: {e}")
        test_context['request'] = request_data
        test_context['response'] = {"error": str(e)}
        test_context['step_executed'] = True
        raise

# Enhanced when steps with better chatText handling
@when(parsers.parse('I respond with complaint date from "{column_name}" if available'))
def respond_with_complaint_date(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I respond with complaint method from "{column_name}" if available'))
def respond_with_complaint_method(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I respond with account number option from "{column_name}" if available'))
def respond_with_account_number_option(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I respond with account number from "{column_name}" if available'))
def respond_with_account_number(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I provide complaint description from "{column_name}" if available'))
def provide_complaint_description(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I respond to followup question from "{column_name}" if available'))
def respond_to_followup_question(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I respond to risk indicator from "{column_name}" if available'))
def respond_to_risk_indicator(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I respond with proceed from "{column_name}" if available'))
def respond_with_proceed(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I respond with communication preference from "{column_name}" if available'))
def respond_with_communication_preference(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I respond with communication details from "{column_name}" if available'))
def respond_with_communication_details(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

@when(parsers.parse('I respond with final proceed from "{column_name}" if available'))
def respond_with_final_proceed(test_context: Dict[str, Any], column_name: str):
    send_message_if_text_available(test_context, column_name)

# --- Then Steps ---

@then('the API response should be successful and contain a valid conversation ID')
def verify_initial_response(test_context: Dict[str, Any]):
    response = test_context.get('initial_response', {})
    
    # Check for conversation ID in either possible field name
    conversation_id = response.get('conversationID') or response.get('conversationId')
    
    assert conversation_id, f"Initial response does not contain a valid conversation ID. Response: {response}"
    assert conversation_id != 'initial', f"Conversation ID should not remain 'initial'. Got: {conversation_id}"
    
    logger.info(f"VERIFIED: Initial response contains valid conversation ID: {conversation_id}")

@then('the initial response action and text should be as expected')
def verify_initial_response_content(test_context: Dict[str, Any]):
    """Verify the initial response contains expected action and text."""
    response = test_context.get('initial_response', {})
    csv_data = test_context.get('csv_data', {})
    
    # Check for expected initial action
    expected_action = csv_data.get('expected_initial_action_label')
    if expected_action:
        actual_action = response.get('nextAction', {}).get('label', '')
        assert expected_action.lower() in actual_action.lower(), \
            f"Expected initial action '{expected_action}' not found in '{actual_action}'"
        logger.info(f"VERIFIED: Initial action matches expected: {expected_action}")
    
    # Check for expected initial response text
    expected_text = csv_data.get('expected_initial_response_text')
    if expected_text:
        actual_text = response.get('chatResponseText', '')
        assert expected_text.lower() in actual_text.lower(), \
            f"Expected initial text '{expected_text}' not found in '{actual_text}'"
        logger.info(f"VERIFIED: Initial response text matches expected: {expected_text}")

@then(parsers.parse('the API response should match expected key "{expected_key}" if step was executed'))
def verify_response_conditionally(test_context: Dict[str, Any], expected_key: str):
    verify_response_if_executed(test_context, expected_key)

@then('verify the conversation details are stored properly in the Complaints AI database')
def verify_conversation_in_db(test_context: Dict[str, Any], db_utils):
    conversation_id = test_context.get('conversation_id')
    
    assert conversation_id, "Conversation ID not found in test context to verify in DB"
    
    # Wait a moment for database write to complete
    import time
    time.sleep(1)
    
    assert db_utils.verify_conversation_exists(conversation_id), \
        f"Conversation {conversation_id} was not found in the Complaints AI database"
    
    logger.info(f"VERIFIED: Conversation {conversation_id} exists in the Complaints AI database")

@then('verify the complaint details are stored properly in the Complaints database')
def verify_complaint_details_in_db(test_context: Dict[str, Any], db_utils):
    response_text = test_context.get('last_response', {}).get('chatResponseText', '')
    
    # Extract interaction ID using regex
    pattern = r'INT[E0L]-\d{6}-\w{12}'
    match = re.search(pattern, response_text)
    
    assert match, f"No valid Interaction ID found in the final response text: '{response_text}'"
    
    interaction_id = match.group(0)
    test_context['interaction_id'] = interaction_id
    
    # Wait a moment for database write to complete
    import time
    time.sleep(2)
    
    complaint_details = db_utils.get_complaint_details(interaction_id)
    assert complaint_details, f"Complaint details for interaction ID {interaction_id} not found in the Complaints database"
    
    logger.info(f"VERIFIED: Complaint details for interaction ID {interaction_id} exist in the Complaints database")

@then('the API response should contain a followup question from LLM if step was executed')
def verify_llm_followup_question(test_context: Dict[str, Any]):
    if test_context.get('step_skipped', False):
        logger.info("SKIPPED: LLM followup question verification - step was skipped")
        return
    
    if not test_context.get('step_executed', False):
        logger.info("SKIPPED: LLM followup question verification - step was not executed")
        return
    
    response_text = test_context.get('last_response', {}).get('chatResponseText', '')
    
    # Look for question indicators
    has_question = ('?' in response_text or 
                   'please' in response_text.lower() or 
                   'can you' in response_text.lower() or
                   'could you' in response_text.lower())
    
    assert has_question, f"Expected a followup question from LLM, but response doesn't appear to contain one: '{response_text}'"
    logger.info("VERIFIED: API response contains a followup question from LLM")

@then('the API response should contain a followup indicator question if step was executed')
def verify_llm_indicator_question(test_context: Dict[str, Any]):
    if test_context.get('step_skipped', False):
        logger.info("SKIPPED: LLM indicator question verification - step was skipped")
        return
    
    if not test_context.get('step_executed', False):
        logger.info("SKIPPED: LLM indicator question verification - step was not executed")
        return
    
    response_text = test_context.get('last_response', {}).get('chatResponseText', '')
    
    # Look for risk/indicator question patterns
    indicator_patterns = ['risk', 'indicator', 'concern', 'additional', 'anything else', '?']
    has_indicator_question = any(pattern in response_text.lower() for pattern in indicator_patterns)
    
    assert has_indicator_question, f"Expected a risk/indicator question from LLM, but not found in: '{response_text}'"
    logger.info("VERIFIED: API response contains a followup indicator question")

@then('the API response should return the clarification summary if step was executed')
def verify_clarification_summary(test_context: Dict[str, Any]):
    if test_context.get('step_skipped', False):
        logger.info("SKIPPED: Clarification summary verification - step was skipped")
        return
    
    if not test_context.get('step_executed', False):
        logger.info("SKIPPED: Clarification summary verification - step was not executed")
        return
    
    response = test_context.get('last_response', {})
    response_text = response.get('chatResponseText', '').lower()
    
    # Look for summary indicators
    summary_patterns = ['summary', 'understand', 'clarification', 'details']
    has_summary = any(pattern in response_text for pattern in summary_patterns)
    
    assert has_summary, f"Expected a clarification summary, but not found in response: '{response.get('chatResponseText', '')}'"
    logger.info("VERIFIED: API response returned the clarification summary")

@then('the API response should return the classification summary if step was executed')
def verify_classification_summary(test_context: Dict[str, Any]):
    if test_context.get('step_skipped', False):
        logger.info("SKIPPED: Classification summary verification - step was skipped")
        return
    
    if not test_context.get('step_executed', False):
        logger.info("SKIPPED: Classification summary verification - step was not executed")
        return
    
    response = test_context.get('last_response', {})
    response_text = response.get('chatResponseText', '').lower()
    
    # Look for classification indicators  
    classification_patterns = ['classification', 'summary', 'categorized', 'type', 'complaint']
    has_classification = any(pattern in response_text for pattern in classification_patterns)
    
    assert has_classification, f"Expected a classification summary, but not found in response: '{response.get('chatResponseText', '')}'"
    logger.info("VERIFIED: API response returned the classification summary")

# --- Debug Helper Steps ---

@then('debug the current test context')
def debug_test_context(test_context: Dict[str, Any]):
    """Debug helper to print current test context state."""
    logger.info("=== DEBUG TEST CONTEXT ===")
    logger.info(f"Test Case ID: {test_context.get('test_case_id', 'Not Set')}")
    logger.info(f"Conversation ID: {test_context.get('conversation_id', 'Not Set')}")
    logger.info(f"Step Executed: {test_context.get('step_executed', False)}")
    logger.info(f"Step Skipped: {test_context.get('step_skipped', False)}")
    
    workflow_data = test_context.get('workflow_data', {})
    logger.info(f"Available chatText columns:")
    for i in range(1, 12):
        column = f'chatText{i}'
        raw_value = workflow_data.get(column)
        cleaned_value = clean_chat_text(raw_value)
        logger.info(f"  {column}: raw='{raw_value}' -> cleaned='{cleaned_value}'")
    
    last_response = test_context.get('last_response', {})
    logger.info(f"Last Response Keys: {list(last_response.keys())}")
    if 'chatResponseText' in last_response:
        logger.info(f"Last Response Text: '{last_response['chatResponseText'][:200]}...'")
    
    logger.info("=== END DEBUG ===")
