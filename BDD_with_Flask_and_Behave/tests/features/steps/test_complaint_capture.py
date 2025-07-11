import pytest
import uuid
from pytest_bdd import scenario, given, when, then, parsers

# Assuming the service and util modules are in the project's python path
from services.chatbot_api_service import ChatbotAPIService
from utils.logger_config import get_logger

# Initialize logger
logger = get_logger(__name__)

# --- Dynamic Answer Logic ---
# This map allows the test to respond intelligently to LLM-generated questions
# without being brittle. We match keywords in the AI's question to a predefined answer.
DYNAMIC_RESPONSE_MAP = {
    "reason was the customer given for the loan denial": "No reason was provided by the bank.",
    "discrimination or unfair lending practices": "The customer alleges the loan denial is due to Misconduct Sales.",
    "unauthorised transaction": "I have already blocked the card, but the charge remains.",
    "what happened": "The customer's loan application was denied despite having a good credit score.",
    "resolve this": "No resolution was offered by the initial agent."
}

def get_dynamic_answer(question_text):
    """
    Parses the AI's question and finds the best-matching answer from the map.
    This is the core of handling the dynamic nature of the chatbot.
    """
    question_text_lower = question_text.lower()
    for keyword, answer in DYNAMIC_RESPONSE_MAP.items():
        if keyword in question_text_lower:
            logger.info(f"Matched keyword '{keyword}' in AI question. Providing answer.")
            return answer
    
    logger.warning("No specific keyword matched in the AI question. Providing a fallback answer.")
    return "The customer did not provide further details on that topic."


# --- State Management Fixture ---
@pytest.fixture
def chatbot_context():
    """
    This fixture acts as a shared dictionary to pass state between steps.
    It replaces the 'context' object from behave.
    """
    return {}


# --- Scenario Definition ---
# This line links the Gherkin feature file and the specific scenario to this test function.
@scenario('../features/complaint_capture.feature', 'Full complaint capture flow for various complaint types')
def test_complaint_capture():
    """This function binds the scenario to the steps below."""
    pass


# --- Step Definitions ---

@given(parsers.parse('the chatbot API is available for "{channel}"'))
def api_is_available(chatbot_context, channel):
    """Initializes the API service and sets up headers for the scenario."""
    logger.info(f"Setting up test for channel: {channel}")
    chatbot_context['api_service'] = ChatbotAPIService()
    chatbot_context['channel_id'] = channel
    chatbot_context['headers'] = {'CLIENT_CORRELATION_ID': f'test-run-{uuid.uuid4()}'}
    assert chatbot_context['api_service'] is not None

@when(parsers.parse('I start a new complaint conversation for "{complainant_name}"'))
def start_complaint(chatbot_context, complainant_name):
    """Initiates the chat and captures the conversation ID."""
    api = chatbot_context['api_service']
    response = api.initiate_chat(
        channel_id=chatbot_context['channel_id'],
        complainant_name=complainant_name,
        headers=chatbot_context['headers']
    )
    # Persist the conversationID and the response for the next steps
    chatbot_context['conversation_id'] = response.get('conversationID')
    chatbot_context['last_response'] = response
    
    assert chatbot_context['conversation_id'] is not None
    assert 'When was the complaint received?' in response.get('chatResponseText', '')

@when(parsers.parse('I provide the complaint received date as "{date}"'))
def provide_date(chatbot_context, date):
    """Sends the complaint date."""
    api = chatbot_context['api_service']
    response = api.send_message(
        conversation_id=chatbot_context['conversation_id'],
        chat_text=date,
        headers=chatbot_context['headers']
    )
    chatbot_context['last_response'] = response
    assert 'How was the complaint received?' in response.get('chatResponseText', '')

@when(parsers.parse('I provide the complaint received method as "{method}"'))
def provide_method(chatbot_context, method):
    """Sends the complaint reception method."""
    api = chatbot_context['api_service']
    response = api.send_message(
        conversation_id=chatbot_context['conversation_id'],
        chat_text=method,
        headers=chatbot_context['headers']
    )
    chatbot_context['last_response'] = response
    assert 'account or reference' in response.get('chatResponseText', '')

@when('I provide the account number')
def provide_account_number(chatbot_context):
    """Sends a placeholder account number."""
    api = chatbot_context['api_service']
    # This value is static for this example, but could be parameterized
    account_number = "9876543210"
    response = api.send_message(
        conversation_id=chatbot_context['conversation_id'],
        chat_text=account_number,
        headers=chatbot_context['headers']
    )
    chatbot_context['last_response'] = response
    assert 'classify the complaint correctly' in response.get('chatResponseText', '')

@when(parsers.parse('I describe the initial complaint about "{initial_complaint}"'))
def describe_complaint(chatbot_context, initial_complaint):
    """Sends the main complaint text, which triggers the first dynamic question."""
    api = chatbot_context['api_service']
    response = api.send_message(
        conversation_id=chatbot_context['conversation_id'],
        chat_text=initial_complaint,
        headers=chatbot_context['headers']
    )
    chatbot_context['last_response'] = response
    logger.info(f"AI Step 1 Question: {response.get('chatResponseText')}")
    assert response.get('valueType') == 'text'

@when('I answer the AI\'s first clarifying question regarding the complaint')
def answer_step1(chatbot_context):
    """Answers the first dynamic question from the AI."""
    api = chatbot_context['api_service']
    ai_question = chatbot_context['last_response']['chatResponseText']
    answer = get_dynamic_answer(ai_question)
    
    response = api.send_message(
        conversation_id=chatbot_context['conversation_id'],
        chat_text=answer,
        headers=chatbot_context['headers']
    )
    chatbot_context['last_response'] = response
    logger.info(f"AI Step 2 Question: {response.get('chatResponseText')}")
    assert response.get('valueType') == 'text'

@when('I answer the AI\'s second escalation question')
def answer_step2(chatbot_context):
    """Answers the second dynamic question, which should lead to the final summary."""
    api = chatbot_context['api_service']
    ai_question = chatbot_context['last_response']['chatResponseText']
    answer = get_dynamic_answer(ai_question)

    response = api.send_message(
        conversation_id=chatbot_context['conversation_id'],
        chat_text=answer,
        headers=chatbot_context['headers']
    )
    chatbot_context['final_summary'] = response.get('chatResponseText')
    logger.info(f"Final Summary Received: {chatbot_context['final_summary']}")

@then('the chatbot should generate a final summary containing the key details')
def validate_summary(chatbot_context):
    """Validates the content of the final summary."""
    final_summary = chatbot_context.get('final_summary', '')
    assert "Final summary" in final_summary
    
    # Check for key details from the conversation flow
    assert "denied" in final_summary.lower()
    assert "misconduct" in final_summary.lower()
    assert "no reason was provided" in final_summary.lower()
    logger.info("Final summary validation successful.")