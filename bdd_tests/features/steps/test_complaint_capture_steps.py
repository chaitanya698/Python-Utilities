import re
from typing import Dict, Any
import logging
from pytest_bdd import scenario, given, when, then
from bdd_tests_copy.utils.helpers import TestHelpers

# Initialize logger
logger = logging.getLogger(__name__)


@scenario(
    feature_name='./complaint_capture.feature',
    scenario_name='Verify end-to-end complaint capture process using data from an external source'
)
def test_complaint_workflow():
    """Parameterized test for complaint workflow."""
    pass


# Given Steps
@given('the chatbot API is available and test data is loaded')
def setup_test_context(
        given_api_is_available: Dict[str, Any],
        given_test_data_loaded: Dict[str, Any]
) -> Dict[str, Any]:
    """Initialize test context with API and data."""
    context = given_api_is_available
    context.update(given_test_data_loaded)
    # Set correlation ID
    test_case_id = context['test_data'].get('test_case_id', 'TEST')
    context['correlation_id'] = TestHelpers.generate_correlation_id(test_case_id)
    logger.info(
        f"Test initialized. Case: {test_case_id}, Correlation: {context['correlation_id']}"
    )
    return context


# When Steps
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
        test_context['conversation_id'] = response.get('conversationID')
    except Exception as e:
        logger.error(f"Failed to send initial request: {e}")
        raise


@when('the user responds with the complaint date')
def user_responds_with_complaint_date(test_context: Dict[str, Any]) -> None:
    complaint_date = test_context['test_data'].get('complaint_date', '10/01/2025')
    api_client = test_context['api_client']
    conversation_id = test_context['conversation_id']
    # Call send_message with the correct parameters
    response = api_client.send_message(
        conversation_id=conversation_id,
        chat_text=complaint_date,
        action="proceed",
        headers={"CLIENT-CORRELATION-ID": test_context.get('correlation_id')}
    )
    # Save response for validation in then steps
    test_context['response'] = response
    logger.info(f"User responded with complaint date: {complaint_date}")


@when('the user responds with the method of complaint')
def user_responds_with_complaint_method(test_context: Dict[str, Any]) -> None:
    complaint_method = test_context['test_data'].get('complaint_method', 'Phone')
    api_client = test_context['api_client']
    conversation_id = test_context['conversation_id']
    response = api_client.send_message(
        conversation_id=conversation_id,
        chat_text=complaint_method,
        action="proceed",
        headers={"CLIENT-CORRELATION-ID": test_context.get('correlation_id')}
    )
    # Save response for validation in then steps
    test_context['response'] = response
    logger.info(f"User responded with complaint method: {complaint_method}")


@when('the user responds with the account number')
def user_responds_with_account_number(test_context: Dict[str, Any]) -> None:
    account_number = test_context['test_data'].get('account_number', 'DDA...3970')
    api_client = test_context['api_client']
    conversation_id = test_context['conversation_id']
    response = api_client.send_message(
        conversation_id=conversation_id,
        chat_text=account_number,
        action="proceed",
        headers={"CLIENT-CORRELATION-ID": test_context.get('correlation_id')}
    )
    # Save response for validation in then steps
    test_context['response'] = response
    logger.info(f"User responded with account number: {account_number}")


@when('the user responds with the complaint details')
def user_responds_with_complaint_details(test_context: Dict[str, Any]) -> None:
    complaint_details = test_context['test_data'].get(
        'complaint_details',
        'Customer stated that their loan application was rejected despite having a good credit score'
    )
    api_client = test_context['api_client']
    conversation_id = test_context['conversation_id']
    response = api_client.send_message(
        conversation_id=conversation_id,
        chat_text=complaint_details,
        action="proceed",
        headers={"CLIENT-CORRELATION-ID": test_context.get('correlation_id')}
    )
    # Save response for validation in then steps
    test_context['response'] = response
    logger.info(f"User responded with complaint details: {complaint_details}")


@when('the user provides a final summary comment')
def user_provides_summary(test_context: Dict[str, Any]) -> None:
    """User provides summary comment."""
    summary_text = test_context['test_data'].get(
        'final_summary_comment',
        'Its a sales misconduct'
    )
    api_client = test_context['api_client']
    conversation_id = test_context['conversation_id']
    response = api_client.send_message(
        conversation_id=conversation_id,
        chat_text=summary_text,
        action="proceed",
        headers={"CLIENT-CORRELATION-ID": test_context.get('correlation_id')}
    )
    # Save response for validation in then steps
    test_context['response'] = response
    logger.info(f"User provided summary comment: {summary_text}")


@when('the user responds with the contact willingness')
def user_responds_with_contact_willingness(test_context: Dict[str, Any]) -> None:
    contact_response = test_context['test_data'].get('contact_willingness_response', 'Continue')
    api_client = test_context['api_client']
    conversation_id = test_context['conversation_id']
    response = api_client.send_message(
        conversation_id=conversation_id,
        chat_text=contact_response,
        action="proceed",
        headers={"CLIENT-CORRELATION-ID": test_context.get('correlation_id')}
    )
    # Save response for validation in then steps
    test_context['response'] = response
    logger.info(f"User responded with contact willingness: {contact_response}")


@when('the user confirms the summary')
def user_confirms_summary(test_context: Dict[str, Any]) -> None:
    api_client = test_context['api_client']
    conversation_id = test_context['conversation_id']
    response = api_client.send_message(
        conversation_id=conversation_id,
        chat_text="proceed",
        action="proceed",
        headers={"CLIENT-CORRELATION-ID": test_context.get('correlation_id')}
    )
    # Save response for validation in then steps
    test_context['response'] = response
    logger.info(msg="User confirmed summary by proceeding", args=test_context['response'])


@when('the user submits the complaint')
def user_submits_complaint(test_context: Dict[str, Any]) -> None:
    api_client = test_context['api_client']
    conversation_id = test_context['conversation_id']
    response = api_client.send_message(
        conversation_id=conversation_id,
        chat_text="proceed",
        action="proceed",
        headers={"CLIENT-CORRELATION-ID": test_context.get('correlation_id')}
    )
    # Save response for validation in then steps
    test_context['response'] = response
    logger.info(msg="User successfully submitted the complaint with conversation ID: ", args=test_context)


# Then Steps
@then('the API response should be successful and contain a valid conversation ID')
def verify_initial_response(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response')
    assert response, "No response received from API"
    assert 'conversationID' in response, "Response missing conversationID"
    conv_id = response['conversationID']
    assert TestHelpers.validate_conversation_id(conv_id), \
        f"Invalid conversation ID format: {conv_id}"
    logger.info(f"Verified conversation ID: {conv_id}")


@then('the initial response action and text should be as expected')
def verify_initial_action_and_text(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response', {})
    test_data = test_context.get('test_data', {})
    expected_action_label = test_data.get('expected_initial_action_label', 'Confirm date')
    if expected_action_label:
        expected_action = {
            "action": "proceed",
            "actionType": "Button",
            "label": expected_action_label
        }
        assert 'actions' in response, "Response missing actions"
        assert expected_action in response['actions'], \
            f"Expected action not found. Got: {response['actions']}"

    expected_text = test_data.get('expected_initial_response_text', 'When was the complaint received?')
    if expected_text:
        actual_text = response.get('chatResponseText', '')
        assert expected_text in actual_text, \
            f"Expected text '{expected_text}' not found in: '{actual_text}'"
    logger.info("Verified initial response action and text")


@then('the API response should prompt for the method of complaint')
def verify_api_responds_complaint_method(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response', {})
    expected_text = "How the complaint received?"  # Common prompt for method of complaint
    actual_text = response.get('chatResponseText', '')
    assert expected_text in actual_text, \
        f"Expected prompt '{expected_text}' not found in: '{actual_text}'"
    logger.info("Verified API is prompting for method of complaint")


@then('the API response should prompt for the account number')
def verify_api_responds_account_number(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response', {})
    expected_text = "Select the account"  # Common prompt for account selection
    actual_text = response.get('chatResponseText', '')
    assert expected_text in actual_text, \
        f"Expected prompt '{expected_text}' not found in: '{actual_text}'"
    logger.info("Verified API is prompting for account number")


@then('the API response should prompt for complaint details')
def verify_api_responds_complaint_details(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response', {})
    expected_text = "can you tell me about what happened"  # Common prompt for complaint details
    actual_text = response.get('chatResponseText', '')
    assert expected_text in actual_text, \
        f"Expected prompt '{expected_text}' not found in: '{actual_text}'"
    logger.info("Verified API is prompting for complaint details")


@then('the API response should ask for clarification')
def verify_clarification_request(test_context: Dict[str, Any]) -> None:
    """Verify API asks for clarification."""
    response = test_context.get('response', {})
    response_text = response.get('chatResponseText', '')
    assert response_text, "Response text is empty"
    logger.info("Verified clarification request")


@then('the API response should prompt for contact willingness')
def verify_contact_willingness_prompt(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response', {})
    response_text = response.get('chatResponseText', '')
    assert response_text, "Response text is empty"
    logger.info("Verified API is prompting for contact willingness")


@then('the API response should prompt to submit the complaint')
def verify_prompt_to_submit(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response', {})
    response_text = response.get('chatResponseText', '')
    expected_phrases = ["Final classification", "summary", "submit"]
    found_phrase = any(phrase in response_text for phrase in expected_phrases)
    assert found_phrase, f"Expected prompt to submit complaint not found in: {response_text}"
    logger.info("Verified API is prompting to submit the complaint")


@then('the API response should contain a valid chat text')
def verify_chat_text_exists(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response', {})
    assert 'chatResponseText' in response, "Response missing chatResponseText"
    assert response['chatResponseText'], "Chat text is empty"
    logger.info("Verified chat text exists in response")


@then('the API response should prompt for classification summary')
def verify_classification_summary_exists(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response', {})
    assert 'chatResponseText' in response, "Response missing chatResponseText"
    assert response['chatResponseText'], "Chat text is empty"
    expected_phrases = ["Final classification summary", "Complaint type", "Resolution note"]
    found_phrase = any(phrase in response['chatResponseText'] for phrase in expected_phrases)
    assert found_phrase, f"Expected phrases not found in chat text: {response['chatResponseText']}"
    logger.info("Verified chat text exists in response, verify_classification_summary_exists")


@then('the API response should prompt for complaint classification')
def verify_complaint_classification(test_context: Dict[str, Any]) -> None:
    verify_classification_summary_exists(test_context)


@then('the final response should contain a confirmation and a valid Interaction ID')
def verify_final_response(test_context: Dict[str, Any]) -> None:
    response = test_context.get('response', {})
    response_text = response.get('chatResponseText', '')
    # Verify confirmation message
    assert 'Thanks for submitting the Complaint' in response_text, \
        "Confirmation message not found"
    # Verify interaction ID format
    pattern = r'INT[EDL]-\d{6}-\w{12}'
    match = re.search(pattern, response_text)
    assert match, f"No valid Interaction Reference Number found in: {response_text}"
    interaction_id = match.group(0)
    test_context['interaction_id'] = interaction_id
    logger.info(f"Verified final response with Interaction ID: {interaction_id}")


@then('the chat history should be correctly stored in the database')
def verify_chat_history_in_db(test_context: Dict[str, Any], db_utils) -> None:
    """Verify that the chat history has been stored in the database."""
    conversation_id = test_context.get('conversation_id')
    assert conversation_id, "Conversation ID not found in test context"

    chat_history = db_utils.get_chat_history(conversation_id)
    assert chat_history, f"No chat history found for conversation ID: {conversation_id}"

    # Example validation: Check if the user's complaint details are in the chat history
    complaint_details = test_context['test_data'].get('complaint_details')
    found_details = False
    for message in chat_history:
        if complaint_details in message.get('user_message', ''):
            found_details = True
            break
    
    assert found_details, f"Complaint details not found in chat history for conversation: {conversation_id}"
    logger.info(f"Verified chat history in the database for conversation: {conversation_id}")
