import csv
import re
import logging
from typing import Dict, Any
from pytest_bdd import scenario, given, when, then, parsers
from bdd_tests.utils.helpers import TestHelpers

logger = logging.getLogger(__name__)


# -------- Load test data from CSV --------
def load_test_data_from_csv(filepath: str) -> Dict[str, Dict[str, str]]:
    data = {}
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            field_key = row['field_key']
            data[field_key] = {
                "chatText": row['chatText'],
                "expected_response": row['expected_response']
            }
    return data


# -------- Scenario --------

@scenario(
    feature_name='./complaint_capture_end_to_end.feature',
    scenario_name='Verify complaint workflow from user input to database storage'
)
def test_complaint_workflow():
    pass


# -------- Given --------

@given('the chatbot API is available and test data is loaded')
def setup_test_context(given_api_is_available: Dict[str, Any]) -> Dict[str, Any]:
    context = given_api_is_available

    # Load CSV test data into context
    context['test_data'] = load_test_data_from_csv("bdd_tests/test_data/chat_test_data.csv")

    # Generate correlation ID
    context['correlation_id'] = TestHelpers.generate_correlation_id("ComplaintWorkflow")
    logger.info("Test initialized with correlation=%s", context['correlation_id'])
    return context


# -------- When --------

@when(parsers.parse('the user responds with "{field_key}"'))
def user_responds_with_chattext(test_context: Dict[str, Any], field_key: str) -> None:
    api_client = test_context['api_client']
    conversation_id = test_context.get('conversation_id')
    chat_text = test_context['test_data'][field_key]['chatText']

    response = api_client.send_message(
        conversation_id=conversation_id,
        chat_text=chat_text,
        action="proceed",
        headers={"CLIENT-CORRELATION-ID": test_context['correlation_id']}
    )

    test_context['response'] = response
    logger.info("User responded with %s = %s", field_key, chat_text)


# -------- Then --------

@then(parsers.parse('the API response should prompt with "{expected_key}"'))
def verify_api_response_prompt(test_context: Dict[str, Any], expected_key: str) -> None:
    response = test_context.get('response', {})
    expected_text = test_context['test_data'][expected_key]['expected_response']
    actual_text = response.get('chatResponseText', '')

    assert expected_text in actual_text, \
        f"Expected '{expected_text}' but got '{actual_text}'"
    logger.info("Verified response for %s: %s", expected_key, expected_text)


# -------- Final confirmation --------

@then('the final response should contain a confirmation and a valid interaction ID')
def verify_final_response(test_context: Dict[str, Any]) -> None:
    response_text = test_context.get('response', {}).get('chatResponseText', '')

    assert "Thanks for submitting the Complaint" in response_text, \
        "Final confirmation not found"

    pattern = r'INT[E0L]-\d{6}-\w{12}'
    match = re.search(pattern, response_text)
    assert match, f"No valid Interaction ID in: {response_text}"

    test_context['interaction_id'] = match.group(0)
    logger.info("Final response verified | interactionID=%s", match.group(0))
