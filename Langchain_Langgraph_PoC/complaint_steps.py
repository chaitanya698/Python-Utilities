from behave import given, when, then
from utils.config_manager import load_config
from services.api_client import APIClient
from utils.dynamic_handler import get_dynamic_response

# This runs before each scenario
def before_scenario(context, scenario):
    context.config = load_config()
    context.api = APIClient(context.config)

@given('the user is a bank teller starting a new complaint')
def step_impl(context):
    # Setup is handled in before_scenario
    pass

@when('the teller starts a "{request_type}" request')
def step_impl(context, request_type):
    initial_data = {
        "channelId": "BBVA",
        "conversationID": None, # Placeholder
        "requestType": request_type
    }
    response = context.api.start_complaint(initial_data)
    context.last_response = response
    # The conversationID is now stored in context.api.conversation_id

@when('the teller submits the initial complaint summary: "{summary}"')
def step_impl(context, summary):
    response = context.api.send_response(summary)
    context.last_response = response

@then('the bot should ask a {step_name} question')
def step_impl(context, step_name):
    question = context.last_response.get('chatResponseText')
    assert question is not None, f"{step_name} question was not returned."
    # Basic validation that a question was asked. More specific checks can be added.
    assert '?' in question

@when('the teller answers the Step 1 question with "{answer}"')
def step_impl(context, answer):
    # In a real scenario, this might be more dynamic
    response = context.api.send_response(answer)
    context.last_response = response

@when('the teller answers the Step 2 question with "{answer}"')
def step_impl(context, answer):
    # This step demonstrates dynamic handling based on the question asked
    question = context.last_response.get('chatResponseText')
    # Use a utility to decide the answer based on keywords in the question
    # For example, if the question contains "misconduct" or "unfair", the answer might be "Yes"
    dynamic_answer = get_dynamic_response(question, context.text) # context.text is the Gherkin step text
    response = context.api.send_response(dynamic_answer)
    context.last_response = response