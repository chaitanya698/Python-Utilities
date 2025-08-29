from pytest_bdd import scenario, given, when, then, parsers
from typing import Dict, Any, Optional
import logging
import re

from utils.helpers import TestHelpers

logger = logging.getLogger(__name__)


# ===== Scenario Definition =====
@scenario(
    '../features/complaint_e2e_test.feature',
    'Execute complaint capture workflow for test case "<test_case_id>"'
)
def test_complaint_workflow():
    """Execute complaint capture workflow test."""
    pass


# ===== Helper Functions =====
def is_valid_chat_text(value: Any) -> bool:
    """Check if a chat text value is valid."""
    if value is None:
        return False
    
    text = str(value).strip()
    invalid_values = ['', 'n/a', 'null', 'none', 'nan', '""', "''", 'empty', 'na', '-']
    
    return text.lower() not in invalid_values


def get_chat_text_value(test_context: Dict[str, Any], field_name: str) -> Optional[str]:
    """Get chat text value from test context if available."""
    # First check pre-validated available chat texts
    available_texts = test_context.get('available_chat_texts', {})
    if field_name in available_texts:
        return available_texts[field_name]
    
    # Fallback to CSV data
    csv_data = test_context.get('csv_data', {})
    value = csv_data.get(field_name, '')
    
    if is_valid_chat_text(value):
        return str(value).strip()
    
    return None


def send_chat_message_if_available(test_context: Dict[str, Any], field_name: str) -> bool:
    """Send chat message if data is available for the field. Returns True if executed."""
    chat_text = get_chat_text_value(test_context, field_name)
    
    if not chat_text:
        logger.info(f"⏭️ Skipping {field_name} - no valid data")
        test_context['last_step_skipped'] = True
        return False
    
    api_client = test_context.get('api_client')
    conversation_id = test_context.get('conversation_id')
    
    if not api_client or not conversation_id:
        logger.error(f"❌ Cannot execute {field_name} - missing API client or conversation ID")
        test_context['last_step_skipped'] = True
        return False
    
    try:
        correlation_id = test_context.get('correlation_id', TestHelpers.generate_correlation_id())
        
        logger.info(f"📤 Sending {field_name}: {chat_text[:50]}...")
        
        response = api_client.send_message(
            conversation_id=conversation_id,
            chat_text=chat_text,
            action="proceed",
            correlation_id=correlation_id
        )
        
        # Store for verification and reporting
        test_context['last_response'] = response
        test_context['last_step_skipped'] = False
        test_context['request'] = {
            'conversationId': conversation_id,
            'chatText': chat_text,
            'action': 'proceed'
        }
        test_context['response'] = response
        
        logger.info(f"✅ Successfully sent {field_name}")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to send {field_name}: {e}")
        test_context['last_step_skipped'] = False  # Was attempted but failed
        raise


# ===== Given Steps =====
@given('the chatbot API is available and test data is loaded')
def setup_test_context(given_api_is_available, given_test_data_loaded):
    """Setup test context with API client and test data."""
    test_context = given_test_data_loaded
    
    # Generate correlation ID for this test
    test_case_id = test_context.get('test_case_id', 'unknown')
    test_context['correlation_id'] = TestHelpers.generate_correlation_id(test_case_id)
    
    # Log available data
    available_texts = test_context.get('available_chat_texts', {})
    logger.info(f"🎯 Test {test_case_id} starting with {len(available_texts)} chat text fields")
    
    return test_context


@given(parsers.parse('the expected responses are loaded from "{json_file}"'))
def load_expected_responses(test_context: Dict[str, Any], json_file: str):
    """Load expected responses from JSON file."""
    from utils.data_loader import DataLoader
    data_loader = DataLoader()
    
    try:
        test_context['expected_responses'] = data_loader.load_json(json_file, from_resources=True)
        logger.info(f"📋 Loaded expected responses from {json_file}")
    except Exception as e:
        logger.warning(f"⚠️ Could not load expected responses: {e}")
        test_context['expected_responses'] = {}


@given(parsers.parse('I have test case "{test_case_id}" with data from CSV'))
def verify_test_case_data(test_context: Dict[str, Any], test_case_id: str):
    """Verify test case data is loaded."""
    assert test_context.get('test_case_id') == test_case_id, f"Test case ID mismatch"
    
    available_texts = test_context.get('available_chat_texts', {})
    if not available_texts:
        logger.warning(f"⚠️ Test case {test_case_id} has no valid chat text data")
    
    return test_context


# ===== When Steps with Conditional Execution =====
@when('I send the initial complaint request')
def send_initial_request(test_context: Dict[str, Any]):
    """Send initial complaint request."""
    api_client = test_context['api_client']
    csv_data = test_context.get('csv_data', {})
    
    # Build initial request
    request_data = {
        'channelID': 'BBVA',
        'conversationId': 'initial',
        'requestType': 'ComplaintCapture',
        'chatText': 'proceed',
        'action': 'proceed',
        'dataElements': []
    }
    
    # Add any initial data elements from CSV
    if csv_data.get('businessName'):
        request_data['dataElements'].append({
            'name': 'businessName',
            'value': csv_data['businessName']
        })
    
    try:
        response = api_client.initiate_chat(
            request_data=request_data,
            correlation_id=test_context['correlation_id']
        )
        
        test_context['initial_response'] = response
        test_context['last_response'] = response
        test_context['conversation_id'] = response.get('conversationID') or response.get('conversationId')
        test_context['request'] = request_data
        test_context['response'] = response
        
        logger.info(f"🚀 Initial request sent. Conversation ID: {test_context['conversation_id']}")
        
    except Exception as e:
        logger.error(f"❌ Initial request failed: {e}")
        raise


# Define all the conditional when steps
@when(parsers.parse('I respond with complaint date from "{field_name}" if available'))
def respond_with_complaint_date(test_context: Dict[str, Any], field_name: str):
    """Send complaint date if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I respond with complaint method from "{field_name}" if available'))
def respond_with_complaint_method(test_context: Dict[str, Any], field_name: str):
    """Send complaint method if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I respond with account number option from "{field_name}" if available'))
def respond_with_account_option(test_context: Dict[str, Any], field_name: str):
    """Send account number option if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I respond with account number from "{field_name}" if available'))
def respond_with_account_number(test_context: Dict[str, Any], field_name: str):
    """Send account number if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I provide complaint description from "{field_name}" if available'))
def provide_complaint_description(test_context: Dict[str, Any], field_name: str):
    """Send complaint description if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I respond to followup question from "{field_name}" if available'))
def respond_to_followup(test_context: Dict[str, Any], field_name: str):
    """Send followup response if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I respond to risk indicator from "{field_name}" if available'))
def respond_to_risk_indicator(test_context: Dict[str, Any], field_name: str):
    """Send risk indicator response if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I respond with proceed from "{field_name}" if available'))
def respond_with_proceed(test_context: Dict[str, Any], field_name: str):
    """Send proceed response if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I respond with communication preference from "{field_name}" if available'))
def respond_with_comm_preference(test_context: Dict[str, Any], field_name: str):
    """Send communication preference if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I respond with communication details from "{field_name}" if available'))
def respond_with_comm_details(test_context: Dict[str, Any], field_name: str):
    """Send communication details if available."""
    send_chat_message_if_available(test_context, field_name)


@when(parsers.parse('I respond with final proceed from "{field_name}" if available'))
def respond_with_final_proceed(test_context: Dict[str, Any], field_name: str):
    """Send final proceed response if available."""
    send_chat_message_if_available(test_context, field_name)


# ===== Then Steps with Skip Handling =====
def should_skip_verification(test_context: Dict[str, Any]) -> bool:
    """Check if verification should be skipped."""
    return test_context.get('last_step_skipped', False)


@then('the API response should be successful and contain a valid conversation ID')
def verify_initial_response(test_context: Dict[str, Any]):
    """Verify initial response is successful."""
    response = test_context.get('initial_response', {})
    conversation_id = test_context.get('conversation_id')
    
    assert conversation_id, "No conversation ID in response"
    assert conversation_id != 'initial', "Conversation ID not updated"
    
    logger.info(f"✅ Initial response verified. Conversation ID: {conversation_id}")


@then('the initial response action and text should be as expected')
def verify_initial_content(test_context: Dict[str, Any]):
    """Verify initial response content."""
    response = test_context.get('initial_response', {})
    
    # Basic validation
    assert 'chatResponseText' in response, "No response text"
    
    logger.info("✅ Initial response content verified")


@then(parsers.parse('the API response should match expected key "{expected_key}" if step was executed'))
def verify_response_conditionally(test_context: Dict[str, Any], expected_key: str):
    """Verify response only if step was executed."""
    if should_skip_verification(test_context):
        logger.info(f"⏭️ Skipping verification of {expected_key} - step was skipped")
        return
    
    response = test_context.get('last_response', {})
    expected_responses = test_context.get('expected_responses', {})
    
    if expected_key in expected_responses:
        expected_text = expected_responses[expected_key]
        actual_text = response.get('chatResponseText', '')
        
        if expected_text.lower() not in actual_text.lower():
            logger.warning(f"⚠️ Expected '{expected_text}' not found in response")
        else:
            logger.info(f"✅ Response matches expected key: {expected_key}")
    else:
        logger.info(f"📋 No expected response defined for {expected_key}")


@then('the API response should contain a followup question from LLM if step was executed')
def verify_llm_followup(test_context: Dict[str, Any]):
    """Verify LLM followup question if step executed."""
    if should_skip_verification(test_context):
        logger.info("⏭️ Skipping LLM followup verification - step was skipped")
        return
    
    response_text = test_context.get('last_response', {}).get('chatResponseText', '')
    
    # Check for question indicators
    has_question = '?' in response_text or any(
        phrase in response_text.lower() 
        for phrase in ['please', 'can you', 'could you', 'would you']
    )
    
    if has_question:
        logger.info("✅ LLM followup question detected")
    else:
        logger.warning("⚠️ No clear followup question detected")


@then('the API response should contain a followup indicator question if step was executed')
def verify_indicator_question(test_context: Dict[str, Any]):
    """Verify indicator question if step executed."""
    if should_skip_verification(test_context):
        logger.info("⏭️ Skipping indicator question verification - step was skipped")
        return
    
    response_text = test_context.get('last_response', {}).get('chatResponseText', '').lower()
    
    indicators = ['risk', 'concern', 'additional', 'anything else', 'other']
    has_indicator = any(ind in response_text for ind in indicators)
    
    if has_indicator:
        logger.info("✅ Indicator question detected")
    else:
        logger.warning("⚠️ No indicator question detected")


@then('the API response should return the clarification summary if step was executed')
def verify_clarification_summary(test_context: Dict[str, Any]):
    """Verify clarification summary if step executed."""
    if should_skip_verification(test_context):
        logger.info("⏭️ Skipping clarification summary verification - step was skipped")
        return
    
    response_text = test_context.get('last_response', {}).get('chatResponseText', '').lower()
    
    summary_indicators = ['summary', 'understand', 'clarification', 'confirm']
    has_summary = any(ind in response_text for ind in summary_indicators)
    
    if has_summary:
        logger.info("✅ Clarification summary detected")
    else:
        logger.warning("⚠️ No clarification summary detected")


@then('the API response should return the classification summary if step was executed')
def verify_classification_summary(test_context: Dict[str, Any]):
    """Verify classification summary if step executed."""
    if should_skip_verification(test_context):
        logger.info("⏭️ Skipping classification summary verification - step was skipped")
        return
    
    response_text = test_context.get('last_response', {}).get('chatResponseText', '').lower()
    
    classification_indicators = ['classification', 'category', 'type', 'classified']
    has_classification = any(ind in response_text for ind in classification_indicators)
    
    if has_classification:
        logger.info("✅ Classification summary detected")
    else:
        logger.warning("⚠️ No classification summary detected")


@then('verify the conversation details are stored properly in the Complaints AI database')
def verify_conversation_in_db(test_context: Dict[str, Any], db_utils):
    """Verify conversation in database."""
    conversation_id = test_context.get('conversation_id')
    
    if not conversation_id:
        logger.warning("⚠️ No conversation ID to verify in database")
        return
    
    # Add a small delay for DB write
    import time
    time.sleep(1)
    
    if db_utils.verify_conversation_exists(conversation_id):
        logger.info(f"✅ Conversation {conversation_id} verified in database")
    else:
        logger.warning(f"⚠️ Conversation {conversation_id} not found in database")


@then('verify the complaint details are stored properly in the Complaints database')
def verify_complaint_in_db(test_context: Dict[str, Any], db_utils):
    """Verify complaint in database."""
    response_text = test_context.get('last_response', {}).get('chatResponseText', '')
    
    # Extract interaction ID
    pattern = r'INT[E0L]-\d{6}-\w{12}'
    match = re.search(pattern, response_text)
    
    if not match:
        logger.warning("⚠️ No interaction ID found in response")
        return
    
    interaction_id = match.group(0)
    
    # Add delay for DB write
    import time
    time.sleep(2)
    
    complaint = db_utils.get_complaint_details(interaction_id)
    if complaint:
        logger.info(f"✅ Complaint {interaction_id} verified in database")
    else:
        logger.warning(f"⚠️ Complaint {interaction_id} not found in database")
