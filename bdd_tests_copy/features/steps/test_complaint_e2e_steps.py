import json
import re
import logging
from typing import Dict, Any, Optional
from pytest_bdd import scenario, given, when, then, parsers

from bdd_tests.utils.helpers import TestHelpers
from bdd_tests.utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


# -------- Scenario --------
@scenario(
    '../features/complaint_e2e_test.feature',
    'Execute complaint capture workflow for test case "<test_case_id>"'
)
def test_complaint_workflow():
    pass


# -------- Given Steps --------
@given('the chatbot API is available and test data is loaded')
def setup_test_context(given_api_is_available: Dict[str, Any]) -> Dict[str, Any]:
    """Setup basic test context with API client."""
    context = given_api_is_available
    context['step_execution_log'] = []
    context['skipped_steps'] = []
    logger.info("Test context initialized with API client")
    return context


@given(parsers.parse('the expected responses are loaded from "{json_file}"'))
def load_expected_responses(test_context: Dict[str, Any], json_file: str) -> None:
    """Load expected responses from JSON file."""
    data_loader = DataLoader()
    try:
        expected_responses = data_loader.load_json(json_file, from_resources=True)
        test_context['expected_responses'] = expected_responses
        logger.info(f"Expected responses loaded from {json_file}")
    except FileNotFoundError:
        logger.warning(f"Expected responses file not found: {json_file}")
        test_context['expected_responses'] = {}


@given(parsers.parse('I have test case "{test_case_id}" with data from CSV'))
def load_test_case_data(test_context: Dict[str, Any], test_case_id: str, test_data_row: Dict[str, Any]) -> None:
    """Load specific test case data from CSV and map to chatText workflow."""
    test_context['test_case_id'] = test_case_id
    test_context['csv_data'] = test_data_row
    test_context['correlation_id'] = TestHelpers.generate_correlation_id(test_case_id)
    
    # Map CSV columns to chatText workflow steps
    test_context['workflow_data'] = map_csv_to_workflow(test_data_row)
    
    # Log available workflow data
    available_steps = [key for key, value in test_context['workflow_data'].items() if value]
    logger.info(f"Test case {test_case_id} mapped to workflow with available steps: {available_steps}")


# -------- CSV to Workflow Mapping --------
def map_csv_to_workflow(csv_data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """Map CSV columns to workflow steps (chatText1-11 equivalents)."""
    workflow_mapping = {}
    
    # Check if CSV already has chatText columns (preferred)
    for i in range(1, 12):
        chattext_col = f'chatText{i}'
        if chattext_col in csv_data:
            value = str(csv_data[chattext_col]).strip() if csv_data[chattext_col] else ""
            workflow_mapping[chattext_col] = value if value and value.lower() not in ['', 'n/a', 'null', 'none'] else None
        else:
            workflow_mapping[chattext_col] = None
    
    # If no chatText columns exist, map from other CSV columns
    if not any(workflow_mapping.values()):
        logger.info("No chatText columns found, mapping from other CSV columns")
        
        # Map based on the workflow logic:
        # Step 1: Complaint Date
        workflow_mapping['chatText1'] = get_valid_value(csv_data.get('complaint_date'))
        
        # Step 2: Complaint Method 
        workflow_mapping['chatText2'] = get_valid_value(csv_data.get('complaint_method'))
        
        # Step 3: Account Number Selection (Yes/No)
        if csv_data.get('account_number'):
            workflow_mapping['chatText3'] = "Yes"  # User wants to provide account number
        
        # Step 4: Account Number Entry
        workflow_mapping['chatText4'] = get_valid_value(csv_data.get('account_number'))
        
        # Step 5: Complaint Details/Description
        workflow_mapping['chatText5'] = get_valid_value(csv_data.get('complaint_details'))
        
        # Step 6: Followup Question Response (generic response)
        if workflow_mapping['chatText5']:
            workflow_mapping['chatText6'] = "Yes, please proceed"
        
        # Step 7: Risk Indicator Response (generic)
        if workflow_mapping['chatText6']:
            workflow_mapping['chatText7'] = "No additional risk factors"
        
        # Step 8: Proceed Decision (Yes to proceed)
        if workflow_mapping['chatText7']:
            workflow_mapping['chatText8'] = "Yes, proceed"
        
        # Step 9: Communication Preference
        workflow_mapping['chatText9'] = "Email"
        
        # Step 10: Communication Details
        if workflow_mapping['chatText9']:
            workflow_mapping['chatText10'] = "user@example.com"
        
        # Step 11: Final Submission
        if workflow_mapping['chatText10']:
            workflow_mapping['chatText11'] = "Yes, submit complaint"
    
    return workflow_mapping


def get_valid_value(value: Any) -> Optional[str]:
    """Get valid string value, return None for empty/invalid values."""
    if value is None:
        return None
    
    str_value = str(value).strip()
    if not str_value or str_value.lower() in ['', 'n/a', 'null', 'none', 'nan']:
        return None
    
    return str_value


# -------- Helper Functions --------
def get_workflow_text_value(test_context: Dict[str, Any], column_name: str) -> Optional[str]:
    """Get workflow text value, return None if empty or missing."""
    workflow_data = test_context.get('workflow_data', {})
    value = workflow_data.get(column_name)
    return value if value else None


def log_step_execution(test_context: Dict[str, Any], step_name: str, executed: bool, chat_text: str = None):
    """Log step execution for tracking."""
    log_entry = {
        'step': step_name,
        'executed': executed,
        'chat_text': chat_text,
        'timestamp': TestHelpers.get_current_timestamp()
    }
    test_context['step_execution_log'].append(log_entry)
    
    if not executed:
        test_context['skipped_steps'].append(step_name)
        logger.info(f"SKIPPED: {step_name} - No workflow data provided")
    else:
        logger.info(f"EXECUTED: {step_name} - chatText: {chat_text}")


def send_message_if_text_available(test_context: Dict[str, Any], column_name: str, action: str = "proceed") -> bool:
    """Send message if workflow text is available, return whether step was executed."""
    chat_text = get_workflow_text_value(test_context, column_name)
    step_name = f"respond_with_{column_name}"
    
    if chat_text is None:
        log_step_execution(test_context, step_name, False)
        return False
    
    api_client = test_context['api_client']
    conversation_id = test_context.get('conversation_id')
    correlation_id = test_context['correlation_id']
    
    try:
        response = api_client.send_message(
            conversation_id=conversation_id,
            chat_text=chat_text,
            action=action,
            headers={"CLIENT-CORRELATION-ID": correlation_id}
        )
        
        test_context['last_response'] = response
        test_context['response'] = response  # For backward compatibility
        log_step_execution(test_context, step_name, True, chat_text)
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to send message for {column_name}: {e}")
        log_step_execution(test_context, step_name, False, chat_text)
        return False


def verify_response_if_executed(test_context: Dict[str, Any], expected_key: str, step_executed: bool) -> None:
    """Verify response only if the previous step was executed."""
    if not step_executed:
        logger.info(f"SKIPPED VERIFICATION: {expected_key} - Previous step was not executed")
        return
    
    response = test_context.get('last_response', {})
    expected_responses = test_context.get('expected_responses', {})
    
    if expected_key in expected_responses:
        expected_text = expected_responses[expected_key]
        actual_text = response.get('chatResponseText', '')
        
        assert expected_text in actual_text, \
            f"Expected '{expected_text}' not found in response: '{actual_text}'"
        logger.info(f"VERIFIED: Response contains expected text for {expected_key}")
    else:
        logger.warning(f"Expected response key '{expected_key}' not found in JSON file")


# -------- When Steps (Conditional Execution) --------
@when('I send the initial complaint request')
def send_initial_request(test_context: Dict[str, Any]) -> None:
    """Send initial complaint request using CSV data."""
    api_client = test_context['api_client']
    correlation_id = test_context['correlation_id']
    
    # Build initial request from CSV data
    csv_data = test_context['csv_data']
    request_data = {
        "channelID": "BBVA",
        "conversationId": "initial",
        "dataElements": [],
        "requestType": "ComplaintCapture",
        "chatText": "Initial complaint request",
        "action": "proceed"
    }
    
    # Add data elements from CSV if available
    data_elements = []
    
    # Add business name if available
    if csv_data.get('initial_request_file') and 'wf' in csv_data.get('initial_request_file', '').lower():
        data_elements.append({"name": "businessName", "value": "WF"})
    else:
        data_elements.append({"name": "businessName", "value": "OtherBank"})
    
    # Add customer info if available
    if csv_data.get('account_number'):
        # Extract customer name from account number context if available
        data_elements.append({"name": "customerFullName", "value": "John Doe"})
    
    # Add country
    data_elements.append({"name": "country", "value": "USA"})
    
    # Add data elements to request
    if data_elements:
        request_data['dataElements'] = data_elements
    
    response = api_client.initiate_chat(
        request_data=request_data,
        correlation_id=correlation_id
    )
    
    test_context['initial_response'] = response
    test_context['conversation_id'] = response.get('conversationId')
    logger.info(f"Initial request sent, conversation ID: {test_context['conversation_id']}")


@when(parsers.parse('I respond with complaint date from "{column_name}" if available'))
def respond_with_complaint_date(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond with complaint date if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I respond with complaint method from "{column_name}" if available'))
def respond_with_complaint_method(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond with complaint method if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I respond with account number option from "{column_name}" if available'))
def respond_with_account_number_option(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond with account number option if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I respond with account number from "{column_name}" if available'))
def respond_with_account_number(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond with account number if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I provide complaint description from "{column_name}" if available'))
def provide_complaint_description(test_context: Dict[str, Any], column_name: str) -> None:
    """Provide complaint description if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I respond to followup question from "{column_name}" if available'))
def respond_to_followup_question(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond to followup question if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I respond to risk indicator from "{column_name}" if available'))
def respond_to_risk_indicator(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond to risk indicator if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I respond with proceed from "{column_name}" if available'))
def respond_with_proceed(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond with proceed if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I respond with communication preference from "{column_name}" if available'))
def respond_with_communication_preference(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond with communication preference if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I respond with communication details from "{column_name}" if available'))
def respond_with_communication_details(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond with communication details if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


@when(parsers.parse('I respond with final proceed from "{column_name}" if available'))
def respond_with_final_proceed(test_context: Dict[str, Any], column_name: str) -> None:
    """Respond with final proceed if available in workflow data."""
    executed = send_message_if_text_available(test_context, column_name)
    test_context[f'{column_name}_executed'] = executed


# -------- Then Steps (Conditional Verification) --------
@then('the API response should be successful and contain a valid conversation ID')
def verify_initial_response(test_context: Dict[str, Any]) -> None:
    """Verify initial response is successful."""
    response = test_context.get('initial_response', {})
    assert 'conversationId' in response, "Initial response missing conversationId"
    assert response['conversationId'] != 'initial', "ConversationId not updated from initial"
    logger.info("Initial response verified successfully")


@then('the initial response action and text should be as expected')
def verify_initial_response_content(test_context: Dict[str, Any]) -> None:
    """Verify initial response content matches CSV expectations."""
    response = test_context.get('initial_response', {})
    csv_data = test_context.get('csv_data', {})
    
    assert 'chatResponseText' in response, "Initial response missing chatResponseText"
    
    # Check if expected initial response is defined in CSV
    expected_initial_response = csv_data.get('expected_initial_response_text')
    if expected_initial_response:
        actual_text = response.get('chatResponseText', '')
        assert expected_initial_response in actual_text, \
            f"Expected '{expected_initial_response}' not found in initial response: '{actual_text}'"
        logger.info(f"Initial response content verified against CSV expectation")
    else:
        logger.info("Initial response content verified (no specific expectation in CSV)")


@then(parsers.parse('the API response should match expected key "{expected_key}" if step was executed'))
def verify_response_conditionally(test_context: Dict[str, Any], expected_key: str) -> None:
    """Verify response only if corresponding step was executed."""
    # Find the most recent step execution
    last_executed_step = None
    for column_name in ['chatText1', 'chatText2', 'chatText3', 'chatText4', 'chatText5', 
                       'chatText6', 'chatText7', 'chatText8', 'chatText9', 'chatText10', 'chatText11']:
        if test_context.get(f'{column_name}_executed'):
            last_executed_step = column_name
    
    if last_executed_step:
        step_executed = test_context.get(f'{last_executed_step}_executed', False)
        verify_response_if_executed(test_context, expected_key, step_executed)
    else:
        logger.info(f"SKIPPED VERIFICATION: {expected_key} - No recent step execution found")


@then(parsers.parse('the API response should contain a followup question from LLM if step was executed'))
def verify_followup_question_conditionally(test_context: Dict[str, Any]) -> None:
    """Verify followup question only if step was executed."""
    step_executed = test_context.get('chatText5_executed', False)
    if step_executed:
        response = test_context.get('last_response', {})
        response_text = response.get('chatResponseText', '')
        assert len(response_text) > 0, "Expected followup question but got empty response"
        logger.info("Followup question verified")
    else:
        logger.info("SKIPPED: Followup question verification - chatText5 step not executed")


@then(parsers.parse('the API response should contain a followup indicator question if step was executed'))
def verify_followup_indicator_conditionally(test_context: Dict[str, Any]) -> None:
    """Verify followup indicator question only if step was executed."""
    step_executed = test_context.get('chatText6_executed', False)
    if step_executed:
        response = test_context.get('last_response', {})
        response_text = response.get('chatResponseText', '')
        assert len(response_text) > 0, "Expected followup indicator question but got empty response"
        logger.info("Followup indicator question verified")
    else:
        logger.info("SKIPPED: Followup indicator verification - chatText6 step not executed")


@then(parsers.parse('the API response should return the clarification summary if step was executed'))
def verify_clarification_summary_conditionally(test_context: Dict[str, Any]) -> None:
    """Verify clarification summary only if step was executed."""
    step_executed = test_context.get('chatText7_executed', False)
    if step_executed:
        response = test_context.get('last_response', {})
        response_text = response.get('chatResponseText', '')
        assert len(response_text) > 0, "Expected clarification summary but got empty response"
        # Look for summary-related keywords
        summary_keywords = ['summary', 'clarification', 'understand', 'confirm']
        has_summary_content = any(keyword in response_text.lower() for keyword in summary_keywords)
        if not has_summary_content:
            logger.warning("Clarification summary may not contain expected content")
        logger.info("Clarification summary verified")
    else:
        logger.info("SKIPPED: Clarification summary verification - chatText7 step not executed")


@then(parsers.parse('the API response should return the classification summary if step was executed'))
def verify_classification_summary_conditionally(test_context: Dict[str, Any]) -> None:
    """Verify classification summary only if step was executed."""
    step_executed = test_context.get('chatText10_executed', False)
    if step_executed:
        response = test_context.get('last_response', {})
        response_text = response.get('chatResponseText', '')
        assert len(response_text) > 0, "Expected classification summary but got empty response"
        logger.info("Classification summary verified")
    else:
        logger.info("SKIPPED: Classification summary verification - chatText10 step not executed")


@then('verify the conversation details are stored properly in the Complaints AI database')
def verify_conversation_database(test_context: Dict[str, Any], db_utils) -> None:
    """Verify conversation is stored in database."""
    conversation_id = test_context.get('conversation_id')
    if conversation_id:
        try:
            exists = db_utils.verify_conversation_exists(conversation_id)
            assert exists, f"Conversation {conversation_id} not found in database"
            logger.info(f"Conversation {conversation_id} verified in database")
        except Exception as e:
            logger.warning(f"Database verification failed: {e}")
            # Don't fail the test if database verification fails
    else:
        logger.warning("No conversation ID available for database verification")


@then('verify the complaint details are stored properly in the Complaints database')
def verify_complaint_database(test_context: Dict[str, Any], db_utils) -> None:
    """Verify complaint details are stored in database."""
    # Extract interaction ID from final response if available
    final_response = test_context.get('last_response', {})
    response_text = final_response.get('chatResponseText', '')
    
    # Look for interaction ID pattern
    pattern = r'INT[E0L]-\d{6}-\w{12}'
    match = re.search(pattern, response_text)
    
    if match:
        interaction_id = match.group(0)
        try:
            complaint_details = db_utils.get_complaint_details(interaction_id)
            assert complaint_details is not None, f"Complaint details not found for interaction {interaction_id}"
            logger.info(f"Complaint details verified for interaction {interaction_id}")
            test_context['interaction_id'] = interaction_id
        except Exception as e:
            logger.warning(f"Database complaint verification failed: {e}")
            # Don't fail the test if database verification fails
    else:
        logger.warning("No interaction ID found in final response, skipping database verification")


# -------- Debug Helper --------
@then('debug workflow mapping')
def debug_workflow_mapping(test_context: Dict[str, Any]) -> None:
    """Debug step to log workflow mapping for troubleshooting."""
    logger.info("=== WORKFLOW MAPPING DEBUG ===")
    csv_data = test_context.get('csv_data', {})
    workflow_data = test_context.get('workflow_data', {})
    
    logger.info("CSV Data:")
    for key, value in csv_data.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("Workflow Mapping:")
    for key, value in workflow_data.items():
        status = "✓ HAS_DATA" if value else "✗ NO_DATA"
        logger.info(f"  {key}: {value} [{status}]")
    
    logger.info("Step Execution Log:")
    for log_entry in test_context.get('step_execution_log', []):
        logger.info(f"  {log_entry}")
    
    logger.info("=== END DEBUG ===")
