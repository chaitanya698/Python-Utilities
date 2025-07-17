from pytest_bdd import scenario, given, when, then, parsers
import uuid

# The scenario is now bound to the feature file
@scenario('../features/complaint_capture.feature', 'Full complaint capture flow for various complaint types')
def test_complaint_capture():
    """This test function binds the scenario to the steps."""
    pass

# --- Step Definitions ---

@given(parsers.parse('the chatbot API is available for "{channel}"'))
def setup_api(channel, chatbot_context, api_service):
    """Sets up the context for the test."""
    chatbot_context['api_service'] = api_service
    chatbot_context['channel_id'] = channel
    chatbot_context['headers'] = {'CLIENT_CORRELATION_ID': f'test-run-{uuid.uuid4()}'}
    return chatbot_context

@when(parsers.parse('I start a new complaint conversation for "{complainant_name}"'))
def start_complaint(chatbot_context, complainant_name):
    """Starts the complaint by calling the API service."""
    api = chatbot_context['api_service']
    response = api.initiate_chat(
        channel_id=chatbot_context['channel_id'],
        complainant_name=complainant_name,
        headers=chatbot_context['headers']
    )
    assert response is not None, "API did not return a response"
    assert response.get('chatResponseText') == "When was the complaint received?"
    chatbot_context['conversation_id'] = response['conversationId']
    chatbot_context['last_response'] = response

@when(parsers.parse('I provide the complaint received date as "{date}"'))
def provide_date(chatbot_context, date):
    """Sends the date to the chatbot."""
    api = chatbot_context['api_service']
    response = api.send_message(
        conversation_id=chatbot_context['conversation_id'],
        chat_text=date,
        headers=chatbot_context['headers']
    )
    assert response is not None
    chatbot_context['last_response'] = response

@when(parsers.parse('I provide the complaint received method as "{method}"'))
def provide_method(chatbot_context, method):
    """Sends the method to the chatbot."""
    # This is a placeholder for the next step in your flow
    pass

@when('I provide the account number')
def provide_account_number(chatbot_context):
    """Sends the account number to the chatbot."""
    # This is a placeholder for the next step in your flow
    pass

@then('the final summary should be generated correctly')
def check_summary(chatbot_context):
    """Checks the final summary from the chatbot."""
    # This is a placeholder for the final assertion
    assert 'last_response' in chatbot_context