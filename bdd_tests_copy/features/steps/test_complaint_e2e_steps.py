from pytest_bdd import scenario, given, when, then, parsers
from typing import Dict, Any
import logging
import re
import time

from utils.helpers import TestHelpers
from utils.error_injector import ErrorInjector

logger = logging.getLogger(__name__)

# --- Scenario Definition ---
@scenario(
    '../features/complaint_capture_e2e.feature',
    'Execute complaint capture workflow for test case "<test_case_id>"'
)
def test_complaint_workflow():
    """Execute the complaint capture workflow test."""
    pass

# --- Helper Functions ---
def is_valid_value(value: Any) -> bool:
    """Check if a value is valid (not empty or a placeholder)."""
    if value is None:
        return False
    text = str(value).strip().lower()
    return text not in ['', 'n/a', 'null', 'none', 'nan', '""', "''", 'empty', 'na', '-']

def execute_workflow_step(test_context: Dict[str, Any], field_name: str, expected_key: str):
    """Executes a single step in the workflow."""
    chat_text = test_context['csv_data'].get(field_name)
    if not is_valid_value(chat_text):
        logger.info(f"⏭️ Skipping step '{field_name}' - no data provided.")
        return

    api_client = test_context['api_client']
    conversation_id = test_context['conversation_id']
    
    response = api_client.send_message(conversation_id, chat_text)
    test_context['last_response'] = response
    
    # Immediate validation
    expected_response = test_context['expected_responses'].get(expected_key)
    if expected_response and expected_response not in response.get('chatResponseText', ''):
        pytest.fail(f"Validation failed for step '{field_name}'. Expected '{expected_response}', but got '{response.get('chatResponseText', '')}'")

# --- Given Steps ---
@given(parsers.parse('I have the test data for test case "{test_case_id}"'))
def get_test_data(test_context: Dict[str, Any], test_case_id: str, test_data_row: Dict[str, Any], data_loader):
    """Load test data and expected responses."""
    test_context['test_case_id'] = test_case_id
    test_context['csv_data'] = test_data_row
    test_context['expected_responses'] = data_loader.load_json("complaint_api_expected_response.json")
    return test_context

# --- When Steps ---
@when('I send the initial complaint request')
def send_initial_request(test_context: Dict[str, Any], data_loader):
    """Send the initial complaint request and capture the conversation ID."""
    api_client = test_context['api_client']
    csv_data = test_context['csv_data']
    
    request_data = data_loader.load_json(csv_data.get('initial_request_file', 'initial_request.json'))
    
    # Inject error if specified
    error_scenario = csv_data.get('initial_request_error_key')
    if is_valid_value(error_scenario) and error_scenario != 'no_change':
        error_injector = ErrorInjector()
        request_data = error_injector.inject_error(request_data, error_scenario)

    response = api_client.initiate_chat(request_data)
    test_context['initial_response'] = response
    test_context['conversation_id'] = response.get('conversationID')

@when('I execute the dynamic complaint capture workflow')
def execute_dynamic_workflow(test_context: Dict[str, Any]):
    """Execute the workflow steps defined in the CSV file."""
    workflow_steps = [
        ("complaint_date", "expected_complaint_method"),
        ("complaint_method", "expected_full_name"),
        # ... and so on for all steps
    ]
    for field, expected_key in workflow_steps:
        execute_workflow_step(test_context, field, expected_key)

# --- Then Steps ---
@then('the initial API response should be successful and contain a valid conversation ID')
def verify_initial_response(test_context: Dict[str, Any]):
    """Verify that the initial response is successful and contains a conversation ID."""
    assert test_context['conversation_id'], "Conversation ID not found in the initial response."

@then('all executed steps should be validated against their expected responses')
def verify_all_steps_validated(test_context: Dict[str, Any]):
    """This step serves as a placeholder to confirm the dynamic workflow has completed."""
    pass

@then('the conversation and complaint details should be verified in the database')
def verify_database_details(test_context: Dict[str, Any], db_utils):
    """Verify the captured data in the database."""
    conversation_id = test_context['conversation_id']
    interaction_id_match = re.search(r'INT[E0L]-\d{6}-\w{12}', test_context.get('last_response', {}).get('chatResponseText', ''))
    
    if conversation_id:
        assert db_utils.verify_conversation_exists(conversation_id), f"Conversation {conversation_id} not found in the database."

    if interaction_id_match:
        interaction_id = interaction_id_match.group(0)
        assert db_utils.get_complaint_details(interaction_id), f"Complaint {interaction_id} not found in the database."
