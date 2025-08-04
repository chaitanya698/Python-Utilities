import re
import uuid
from datetime import datetime
from pytest_bdd import scenario, given, when, then, parsers

from core.utils.logger import get_logger

#Initialize logger
logger = get_logger(name)

@scenario('../features/complaint_capture.feature',
'Verify end-to-end complaint capture process using data from an external source')
def test_complaint_workflow():
    """Parameterized test for complaint workflow."""
    pass

#--- Given Steps ---
@given('the chatbot API is available and test data is loaded')
def setup_test_context(given_api_is_available, given_test_data_loaded):
    """Initialize test context with API and data."""
    context = given_api_is_available
    context.update(given_test_data_loaded)

    # Set correlation ID
    context['correlation_id'] = f"{context['test_data']['test_case_id']}-{uuid.uuid4()}"

    logger.info(f"Test initialized - Case: {context['test_data']['test_case_id']}, "
            f"Correlation: {context['correlation_id']}")

    return context
#--- When Steps ---
@when('I send the initial complaint request')
def send_initial_request(test_context, data_loader):
"""Send initial complaint request."""
api_client = test_context['api_client']
test_data = test_context['test_data']

# Load request template
request_file = test_data['initial_request_file']
initial_request = data_loader.load_json(request_file)

logger.info(f"Sending initial request for case: {test_data['test_case_id']}")

# Send request
response = api_client.initiate_chat(
    request_data=initial_request,
    correlation_id=test_context['correlation_id']
)

test_context['response'] = response
test_context['conversation_id'] = response.get('conversationId')
@when(parsers.parse('the user responds with the {field}'))
def user_responds_with_field(test_context, field):
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

response_text = test_context['test_data'][data_field]
_send_message(test_context, response_text)
@when('the user provides a final summary comment')
def user_provides_summary(test_context):
"""User provides summary comment."""
summary_text = test_context['test_data']['final_summary_comment']
_send_message(test_context, summary_text)

@when('the user confirms the summary')
def user_confirms_summary(test_context):
"""User confirms the summary."""
_send_message(test_context, "Continue", action="proceed")

@when('the user submits the complaint')
def user_submits_complaint(test_context):
"""User submits the complaint."""
_send_message(test_context, "Submit", action="proceed")

--- Then Steps ---
@then('the API response should be successful and contain a valid conversation ID')
def verify_initial_response(test_context):
"""Verify initial API response."""
response = test_context['response']

assert response, "No response received from API"
assert 'conversationId' in response, "Response missing conversationId"

# Verify conversation ID format
conv_id = response['conversationId']
pattern = r'^CVD-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$'
assert re.match(pattern, conv_id), f"Invalid conversation ID format: {conv_id}"

logger.info(f"Verified conversation ID: {conv_id}")
@then('the conversation ID must exist in the database')
def verify_conversation_in_db(test_context, db_operations):
"""Verify conversation exists in database."""
conv_id = test_context['conversation_id']

exists = db_operations.verify_conversation_exists(conv_id)
assert exists, f"Conversation {conv_id} not found in database"

logger.info(f"Verified conversation {conv_id} exists in database")
@then('the initial response action and text should be as expected')
def verify_initial_action_and_text(test_context):
"""Verify expected action and response text."""
response = test_context['response']
test_data = test_context['test_data']

# Verify action
expected_action = {
    "action": "proceed",
    "type": "button",
    "label": test_data['expected_initial_action_label']
}

assert 'actions' in response, "Response missing actions"
assert expected_action in response['actions'], \
    f"Expected action not found. Got: {response['actions']}"

# Verify text
actual_text = response.get('chatResponseText', '')
expected_text = test_data['expected_initial_response_text']

assert actual_text == expected_text, \
    f"Expected: '{expected_text}', Got: '{actual_text}'"

logger.info("Verified initial response action and text")
@then(parsers.parse('the API response should prompt for {expected_prompt}'))
def verify_prompt(test_context, expected_prompt):
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

response_text = test_context['response'].get('chatResponseText', '')
assert pattern in response_text, \
    f"Expected prompt '{pattern}' not found in: {response_text}"

logger.info(f"Verified prompt for: {expected_prompt}")
@then('the API response should contain a valid chat text')
def verify_chat_text_exists(test_context):
"""Verify response contains chat text."""
response = test_context['response']

assert 'chatResponseText' in response, "Response missing chatResponseText"
assert response['chatResponseText'], "Chat text is empty"

logger.info("Verified chat text exists in response")
@then('the API response should ask for clarification')
def verify_clarification_request(test_context):
"""Verify API asks for clarification."""
response = test_context['response']

assert 'Final summary' in response.get('chatResponseText', '')

# Check for clarify action
actions = {act['action'] for act in response.get('actions', [])}
assert 'revise,clarify' in actions or 'clarify' in actions, \
    f"Clarify action not found. Got actions: {actions}"

logger.info("Verified clarification request")
@then('the final response should contain a confirmation and a valid Interaction ID')
def verify_final_response(test_context):
"""Verify final submission response."""
response = test_context['response']
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

logger.info(f"Verified final response with Interaction ID: {interaction_id}")
--- Helper Functions ---
def _send_message(test_context, message: str, action: str = "proceed"):
"""Helper to send message to API."""
api_client = test_context['api_client']
conversation_id = test_context['conversation_id']

logger.debug(f"Sending message: '{message}' with action: '{action}'")

response = api_client.send_message(
    conversation_id=conversation_id,
    message=message,
    action=action,
    correlation_id=test_context['correlation_id']
)

test_context['response'] = response
