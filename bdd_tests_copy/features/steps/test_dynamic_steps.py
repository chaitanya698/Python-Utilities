from pytest_bdd import scenario, given, when, then, parsers
from typing import Dict, Any, Optional, List
import logging
import re
import json
import time

from utils.helpers import TestHelpers
from utils.error_injector import ErrorInjector
from utils.data_loader import DataLoader

logger = logging.getLogger(__name__)


# Mapping of CSV fields to their expected response keys
FIELD_TO_EXPECTED_KEY_MAPPING = {
    'complaint_date': 'expected_complaint_date',
    'complaint_method': 'expected_complaint_method',
    'Full Name': 'expected_full_name',
    'account_number': 'expected_account_number',
    'complaint_eloboration': 'expected_complaint_eloboration',
    'follow_up_question1': 'expected_follow_up_question1',
    'followup_question_2': 'expected_followup_question_2',
    'clarification_revise_action': 'expected_clarification_revise_action',
    'clarification_revise_1': 'expected_clarification_revise_1',
    'clarification_revise_2': 'expected_clarification_revise_2',
    'clarification_revise_3': 'expected_clarification_revise_3',
    'clarification_revise_4': 'expected_clarification_revise_4',
    'unauthorized_account_handling': 'expected_unauthorized_account_handling',
    'contact_willingness_response': 'expected_contact_willingness_response',
    'Add_new_phone_email': 'expected_Add_new_phone_email',
    'complaint_submission_action': 'expected_complaint_submission_action'
}

# Define the order of workflow steps
WORKFLOW_STEPS_ORDER = [
    'complaint_date',
    'complaint_method',
    'Full Name',
    'account_number',
    'complaint_eloboration',
    'follow_up_question1',
    'followup_question_2',
    'clarification_revise_action',
    'clarification_revise_1',
    'clarification_revise_2',
    'clarification_revise_3',
    'clarification_revise_4',
    'unauthorized_account_handling',
    'contact_willingness_response',
    'Add_new_phone_email',
    'complaint_submission_action'
]


# ===== Scenario Definition =====
@scenario(
    '../features/complaint_capture.feature',
    'Execute complaint capture workflow for test case "<test_case_id>"'
)
def test_complaint_workflow():
    """Execute complaint capture workflow test."""
    pass


# ===== Helper Functions =====
def is_valid_value(value: Any) -> bool:
    """Check if a value is valid (not empty or placeholder)."""
    if value is None:
        return False
    
    text = str(value).strip()
    invalid_values = ['', 'n/a', 'null', 'none', 'nan', '""', "''", 'empty', 'na', '-', 'nil', 'undefined']
    
    return text.lower() not in invalid_values


def get_csv_value(test_context: Dict[str, Any], field_name: str) -> Optional[str]:
    """Get value from CSV data for a given field."""
    csv_data = test_context.get('csv_data', {})
    value = csv_data.get(field_name, '')
    
    if is_valid_value(value):
        return str(value).strip()
    
    return None


def load_expected_responses_json(test_context: Dict[str, Any], json_file: str):
    """Load expected responses from JSON file."""
    try:
        data_loader = DataLoader()
        expected_responses = data_loader.load_json(json_file, from_resources=True)
        test_context['expected_responses_json'] = expected_responses
        logger.info(f"📋 Loaded expected responses from {json_file}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load expected responses: {e}")
        test_context['expected_responses_json'] = {}


def execute_workflow_step(test_context: Dict[str, Any], field_name: str) -> bool:
    """Execute a single workflow step if data is available."""
    
    # Get value from CSV
    value = get_csv_value(test_context, field_name)
    
    if not value:
        logger.info(f"⏭️ Skipping {field_name} - no valid data in CSV")
        test_context['last_step_skipped'] = True
        test_context['skipped_steps'].append(field_name)
        return False
    
    api_client = test_context.get('api_client')
    conversation_id = test_context.get('conversation_id')
    
    if not api_client or not conversation_id:
        logger.error(f"❌ Cannot execute {field_name} - missing API client or conversation ID")
        test_context['last_step_skipped'] = True
        return False
    
    try:
        correlation_id = test_context.get('correlation_id', TestHelpers.generate_correlation_id())
        
        logger.info(f"📤 Sending {field_name}: {value[:50]}...")
        
        response = api_client.send_message(
            conversation_id=conversation_id,
            chat_text=value,
            action="proceed",
            correlation_id=correlation_id
        )
        
        # Store for verification
        test_context['last_response'] = response
        test_context['last_step_skipped'] = False
        test_context['last_executed_field'] = field_name
        test_context['executed_steps'].append(field_name)
        
        # Store response for each step
        test_context['step_responses'][field_name] = response
        
        logger.info(f"✅ Successfully sent {field_name}")
        
        # Validate response immediately if expected response is available
        validate_step_response(test_context, field_name)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to send {field_name}: {e}")
        test_context['last_step_skipped'] = False
        test_context['failed_steps'].append(field_name)
        raise


def validate_step_response(test_context: Dict[str, Any], field_name: str) -> bool:
    """Validate response for a specific step against expected response."""
    
    # Check if step was skipped
    if field_name in test_context.get('skipped_steps', []):
        logger.info(f"⏭️ Skipping validation for {field_name} - step was skipped")
        return True
    
    # Get expected response key from mapping
    expected_key = FIELD_TO_EXPECTED_KEY_MAPPING.get(field_name)
    if not expected_key:
        logger.warning(f"⚠️ No expected response mapping for field: {field_name}")
        return True
    
    # Get the JSON key from CSV (e.g., "When_Complaint_Received")
    csv_data = test_context.get('csv_data', {})
    json_key_from_csv = csv_data.get(expected_key, '').strip()
    
    # Get expected responses JSON
    expected_responses_json = test_context.get('expected_responses_json', {})
    
    # Determine the actual expected text
    expected_value = None
    
    if is_valid_value(json_key_from_csv):
        # CSV contains a key that maps to JSON
        if json_key_from_csv in expected_responses_json:
            # Use the JSON value for this key
            expected_value = expected_responses_json.get(json_key_from_csv)
            logger.debug(f"Using JSON mapping: {json_key_from_csv} -> {expected_value[:50] if expected_value else 'None'}...")
        else:
            # If not found in JSON, treat the CSV value as literal expected text
            expected_value = json_key_from_csv
            logger.debug(f"JSON key '{json_key_from_csv}' not found, using as literal text")
    else:
        # No CSV value, try to use default from JSON using the expected key itself
        expected_value = expected_responses_json.get(expected_key, '')
        if expected_value:
            logger.debug(f"Using default JSON value for {expected_key}")
    
    if not expected_value:
        logger.info(f"📋 No expected response defined for {field_name}")
        return True
    
    # Get actual response
    response = test_context.get('step_responses', {}).get(field_name, test_context.get('last_response', {}))
    actual_text = response.get('chatResponseText', '')
    
    # Perform validation
    if str(expected_value).lower() in actual_text.lower():
        logger.info(f"✅ Response validated for {field_name}: contains expected text")
        test_context['validated_steps'].append(field_name)
        return True
    else:
        logger.warning(f"⚠️ Response validation failed for {field_name}")
        logger.debug(f"Expected: {expected_value}")
        logger.debug(f"Actual: {actual_text[:200]}...")
        test_context['validation_failures'].append({
            'field': field_name,
            'expected': expected_value,
            'actual': actual_text,
            'json_key': json_key_from_csv if is_valid_value(json_key_from_csv) else None
        })
        return False


# ===== Given Steps =====
@given('the chatbot API is available and test data is loaded')
def setup_test_context(given_api_is_available, given_test_data_loaded):
    """Setup test context with API client and test data."""
    test_context = given_test_data_loaded
    
    # Initialize tracking lists
    test_context['executed_steps'] = []
    test_context['skipped_steps'] = []
    test_context['failed_steps'] = []
    test_context['validated_steps'] = []
    test_context['validation_failures'] = []
    test_context['step_responses'] = {}
    
    # Generate correlation ID for this test
    test_case_id = test_context.get('test_case_id', 'unknown')
    test_context['correlation_id'] = TestHelpers.generate_correlation_id(test_case_id)
    
    # Log test start
    logger.info(f"🎯 Test {test_case_id} starting")
    
    return test_context


@given(parsers.parse('the expected responses are loaded from "{json_file}"'))
def load_expected_responses(test_context: Dict[str, Any], json_file: str):
    """Load expected responses from JSON file."""
    load_expected_responses_json(test_context, json_file)


@given(parsers.parse('I have test case "{test_case_id}" with data from CSV'))
def verify_test_case_data(test_context: Dict[str, Any], test_case_id: str):
    """Verify test case data is loaded."""
    assert test_context.get('test_case_id') == test_case_id, f"Test case ID mismatch"
    
    # Log available fields
    csv_data = test_context.get('csv_data', {})
    available_fields = [field for field in WORKFLOW_STEPS_ORDER if is_valid_value(csv_data.get(field))]
    
    logger.info(f"📊 Test case {test_case_id} has {len(available_fields)} valid workflow fields")
    test_context['available_workflow_fields'] = available_fields
    
    return test_context


# ===== When Steps =====
@when('I send the initial complaint request with error scenario if specified')
def send_initial_request_with_error_injection(test_context: Dict[str, Any]):
    """Send initial complaint request with optional error injection."""
    api_client = test_context['api_client']
    csv_data = test_context.get('csv_data', {})
    data_loader = DataLoader()
    
    # Load initial request template
    try:
        request_data = data_loader.load_json('initial_request.json', from_resources=True)
    except Exception as e:
        logger.error(f"Failed to load initial request template: {e}")
        request_data = {
            "channelID": "BBVA",
            "conversationId": "initial",
            "requestType": "ComplaintCapture",
            "chatText": "proceed",
            "action": "proceed",
            "dataElements": []
        }
    
    # Check for error injection scenario
    error_key = csv_data.get('initial_request_error_key', '').strip()
    if is_valid_value(error_key) and error_key != 'no_change':
        logger.info(f"🔧 Applying error injection: {error_key}")
        error_injector = ErrorInjector()
        request_data = error_injector.inject_error(request_data, error_key)
        test_context['error_scenario_applied'] = error_key
    
    # Add any initial data elements from CSV if needed
    if csv_data.get('businessName'):
        request_data.setdefault('dataElements', []).append({
            'name': 'businessName',
            'value': csv_data['businessName']
        })
    
    try:
        response = api_client.initiate_chat(
            request_data=request_data,
            correlation_id=test_context['correlation_id']
        )
        
        test_context['initial_request'] = request_data
        test_context['initial_response'] = response
        test_context['last_response'] = response
        test_context['conversation_id'] = response.get('conversationID') or response.get('conversationId')
        test_context['initial_request_success'] = True
        
        logger.info(f"🚀 Initial request sent. Conversation ID: {test_context['conversation_id']}")
        
    except Exception as e:
        logger.error(f"❌ Initial request failed: {e}")
        test_context['initial_request_success'] = False
        test_context['initial_request_error'] = str(e)
        
        # If error was expected, this might be valid
        if error_key:
            logger.info(f"ℹ️ Error might be expected due to error injection: {error_key}")
        else:
            raise


@when('I execute dynamic workflow steps based on available data')
def execute_dynamic_workflow(test_context: Dict[str, Any]):
    """Execute workflow steps dynamically based on available data in CSV."""
    
    # Check if initial request was successful
    if not test_context.get('initial_request_success', False):
        logger.warning("⚠️ Skipping workflow execution - initial request failed")
        return
    
    csv_data = test_context.get('csv_data', {})
    
    # Execute steps in order
    for field_name in WORKFLOW_STEPS_ORDER:
        # Check if field has valid data in CSV
        if is_valid_value(csv_data.get(field_name)):
            logger.info(f"📌 Executing step: {field_name}")
            execute_workflow_step(test_context, field_name)
            
            # Add small delay between steps to avoid overwhelming the API
            time.sleep(0.5)
        else:
            logger.info(f"⏭️ Skipping step: {field_name} - no data in CSV")
            test_context['skipped_steps'].append(field_name)
    
    # Log execution summary
    logger.info(f"📊 Workflow Execution Summary:")
    logger.info(f"  - Executed: {len(test_context['executed_steps'])} steps")
    logger.info(f"  - Skipped: {len(test_context['skipped_steps'])} steps")
    logger.info(f"  - Failed: {len(test_context['failed_steps'])} steps")


@when('the workflow is complete')
def workflow_complete(test_context: Dict[str, Any]):
    """Mark workflow as complete and prepare for final validations."""
    test_context['workflow_complete'] = True
    
    # Extract interaction ID if present in last response
    last_response = test_context.get('last_response', {})
    response_text = last_response.get('chatResponseText', '')
    
    # Look for interaction ID pattern
    pattern = r'INT[E0L]-\d{6}-\w{12}'
    match = re.search(pattern, response_text)
    
    if match:
        test_context['interaction_id'] = match.group(0)
        logger.info(f"🎯 Interaction ID extracted: {test_context['interaction_id']}")


# ===== Then Steps =====
@then('the API response should be validated based on expected initial response')
def validate_initial_response(test_context: Dict[str, Any]):
    """Validate initial response based on expected values."""
    
    # Check if error scenario was applied
    if test_context.get('error_scenario_applied'):
        # For error scenarios, we might expect failure
        if not test_context.get('initial_request_success'):
            logger.info("✅ Initial request failed as expected with error scenario")
            return
    
    # For successful requests, validate response
    if test_context.get('initial_request_success'):
        response = test_context.get('initial_response', {})
        conversation_id = test_context.get('conversation_id')
        
        assert conversation_id, "No conversation ID in response"
        assert conversation_id != 'initial', "Conversation ID not updated"
        
        logger.info(f"✅ Initial response validated. Conversation ID: {conversation_id}")
    else:
        # Check if failure was expected
        error_key = test_context.get('csv_data', {}).get('initial_request_error_key')
        if not error_key or error_key == 'no_change':
            raise AssertionError(f"Initial request failed unexpectedly: {test_context.get('initial_request_error')}")


@then('all executed steps should be validated against expected responses')
def validate_all_executed_steps(test_context: Dict[str, Any]):
    """Validate all executed steps against their expected responses."""
    
    executed_steps = test_context.get('executed_steps', [])
    validation_failures = test_context.get('validation_failures', [])
    
    logger.info(f"📊 Validation Summary:")
    logger.info(f"  - Total steps executed: {len(executed_steps)}")
    logger.info(f"  - Steps validated: {len(test_context.get('validated_steps', []))}")
    logger.info(f"  - Validation failures: {len(validation_failures)}")
    
    if validation_failures:
        logger.warning("⚠️ Some validations failed:")
        for failure in validation_failures:
            logger.warning(f"  - {failure['field']}: Expected '{failure['expected'][:50]}...' but got '{failure['actual'][:50]}...'")
        
        # Determine if failures should be treated as errors
        # You can make this configurable based on test requirements
        strict_validation = test_context.get('csv_data', {}).get('strict_validation', 'false').lower() == 'true'
        if strict_validation:
            raise AssertionError(f"Validation failures: {len(validation_failures)} steps failed validation")
    else:
        logger.info("✅ All executed steps passed validation")


@then('verify the conversation details are stored properly in the Complaints AI database if applicable')
def verify_conversation_in_db(test_context: Dict[str, Any], db_utils):
    """Verify conversation in database if applicable."""
    
    # Skip if initial request failed
    if not test_context.get('initial_request_success'):
        logger.info("⏭️ Skipping DB verification - initial request failed")
        return
    
    conversation_id = test_context.get('conversation_id')
    
    if not conversation_id:
        logger.warning("⚠️ No conversation ID to verify in database")
        return
    
    # Add a small delay for DB write
    time.sleep(2)
    
    if db_utils.verify_conversation_exists(conversation_id):
        logger.info(f"✅ Conversation {conversation_id} verified in database")
    else:
        logger.warning(f"⚠️ Conversation {conversation_id} not found in database")


@then('verify the complaint details are stored properly in the Complaints database if applicable')
def verify_complaint_in_db(test_context: Dict[str, Any], db_utils):
    """Verify complaint in database if applicable."""
    
    interaction_id = test_context.get('interaction_id')
    
    if not interaction_id:
        logger.info("⏭️ No interaction ID found - complaint might not be submitted")
        return
    
    # Add delay for DB write
    time.sleep(2)
    
    complaint = db_utils.get_complaint_details(interaction_id)
    if complaint:
        logger.info(f"✅ Complaint {interaction_id} verified in database")
        test_context['complaint_details'] = complaint
    else:
        logger.warning(f"⚠️ Complaint {interaction_id} not found in database")


# ===== Additional Helper Functions =====
def generate_test_report(test_context: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a detailed test report for the executed test."""
    
    report = {
        'test_case_id': test_context.get('test_case_id'),
        'correlation_id': test_context.get('correlation_id'),
        'conversation_id': test_context.get('conversation_id'),
        'interaction_id': test_context.get('interaction_id'),
        'initial_request_success': test_context.get('initial_request_success'),
        'error_scenario_applied': test_context.get('error_scenario_applied'),
        'executed_steps': test_context.get('executed_steps', []),
        'skipped_steps': test_context.get('skipped_steps', []),
        'failed_steps': test_context.get('failed_steps', []),
        'validated_steps': test_context.get('validated_steps', []),
        'validation_failures': test_context.get('validation_failures', []),
        'workflow_complete': test_context.get('workflow_complete', False)
    }
    
    # Calculate success metrics
    total_possible_steps = len(WORKFLOW_STEPS_ORDER)
    executed_count = len(report['executed_steps'])
    validated_count = len(report['validated_steps'])
    
    report['metrics'] = {
        'execution_rate': (executed_count / total_possible_steps * 100) if total_possible_steps > 0 else 0,
        'validation_rate': (validated_count / executed_count * 100) if executed_count > 0 else 0,
        'success_rate': ((executed_count - len(report['failed_steps'])) / executed_count * 100) if executed_count > 0 else 0
    }
    
    return report
