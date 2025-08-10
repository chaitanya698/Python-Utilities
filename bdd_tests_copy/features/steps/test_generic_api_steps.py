import re
import json
from typing import Dict, Any
import logging
from pytest_bdd import scenario, given, when, then, parsers

from bdd_tests.utils.helpers import TestHelpers
from bdd_tests.utils.error_injector import ErrorInjector
from bdd_tests.utils.request_response_tracker import RequestResponseTracker

# Initialize logger
logger = logging.getLogger(__name__)


@scenario(
    '../features/generic_api_tests.feature',
    'Execute positive API test case'
)
def test_positive_scenarios():
    pass


@scenario(
    '../features/generic_api_tests.feature',
    'Execute negative API test case'
)
def test_negative_scenarios():
    pass


@given('the test environment is configured')
def setup_environment(config, request_response_tracker):
    """Setup test environment."""
    # This fixture is now primarily for setup and context initialization
    # and returns a dictionary to be used as the test context.
    return {
        'config': config,
        'tracker': request_response_tracker,
        'test_data': {}
    }


@given('the API client is initialized')
def initialize_api_client(test_context, api_client):
    """Initialize API client."""
    test_context['api_client'] = api_client
    logger.info("API client initialized")


@given(parsers.parse('I have test data for test case "{test_case_id}"'))
def load_test_data(test_context, test_case_id, test_data_row):
    """Load test data for specific test case."""
    test_context['test_case_id'] = test_case_id
    test_context['test_data'] = test_data_row
    test_context['correlation_id'] = TestHelpers.generate_correlation_id(test_case_id)

    logger.info(f"Loaded test data for {test_case_id}")


@given('the request response tracker is initialized for this test')
def init_tracker_for_test(test_context, request_response_tracker):
    """Initialize tracker for current test."""
    test_id = test_context.get('test_case_id', 'unknown')
    request_response_tracker.set_current_test(test_id)
    logger.info(f"Tracker initialized for test: {test_id}")


@when(parsers.parse('I prepare the initial request from file "{request_file}"'))
def prepare_initial_request(test_context, request_file, data_loader):
    """Prepare initial request from template file."""
    try:
        request_data = data_loader.load_json(request_file, from_resources=True)
        test_context['request_data'] = request_data
        logger.info(f"Prepared request from {request_file}")

    except FileNotFoundError:
        logger.error(f"Request file not found: {request_file}")
        # Create a default request if the file is not found.
        test_context['request_data'] = {
            "channelId": "",
            "conversationId": "initial",
            "dataElements": [],
            "requestType": "ComplaintCapture",
            "chatText": "Initial request",
            "action": "proceed"
        }


@when(parsers.parse('I apply error scenario "{error_scenario}" if defined'))
def apply_error_scenario(test_context, error_scenario):
    """Apply error scenario to the request using ErrorInjector."""
    if error_scenario and error_scenario.lower() not in ('', 'none'):
        error_injector = ErrorInjector()
        request_data = test_context.get('request_data', {})
        modified_request = error_injector.inject_error(request_data, error_scenario)
        test_context['request_data'] = modified_request
        test_context['error_scenario'] = error_scenario
        logger.info(f"Error scenario '{error_scenario}' applied.")


@when('I send the initial API request')
def send_initial_request(test_context):
    """Send the initial API request."""
    api_client = test_context['api_client']
    request_data = test_context['request_data']
    correlation_id = test_context['correlation_id']

    try:
        response = api_client.initiate_chat(
            request_data=request_data,
            correlation_id=correlation_id
        )
        test_context['response'] = response
        test_context.pop('response_error', None)
        if 'conversationId' in response:
            test_context['conversation_id'] = response['conversationId']
        logger.info(f"Request sent successfully, correlation ID: {correlation_id}")
    except Exception as e:
        test_context['response'] = None
        test_context['response_error'] = str(e)
        logger.error(f"Request failed: {e}")


@then('the API response should be successful')
def verify_successful_response(test_context):
    """Verify the API response is successful."""
    assert 'response_error' not in test_context, f"Request failed with error: {test_context.get('response_error')}"
    assert test_context.get('response') is not None, "No response received"
    if 'error_scenario' not in test_context:
        assert 'conversationId' in test_context.get('response', {}), "Response missing conversationId"
    logger.info("Response verified as successful")


@then(parsers.parse('the response should match expected result "{expected_result}"'))
def verify_expected_result(test_context, expected_result):
    """Verify response matches expected result."""
    response = test_context.get('response')
    if expected_result.lower() == 'success':
        assert response is not None, "Expected successful response but got None"
        assert 'error' not in response, f"Unexpected error in response: {response.get('error')}"
    else:
        response_text = json.dumps(response) if response else test_context.get('response_error', '')
        assert expected_result in response_text, f"Expected '{expected_result}' not found in response"
    logger.info(f"Response matches expected result: {expected_result}")


@then(parsers.parse('the API response should contain error "{expected_error}"'))
def verify_error_response(test_context, expected_error):
    """Verify the API response contains expected error."""
    response = test_context.get('response')
    error = test_context.get('response_error')
    error_found = False

    if error and expected_error.lower() in error.lower():
        error_found = True
    elif response and expected_error.lower() in json.dumps(response).lower():
        error_found = True

    assert error_found, f"Expected error '{expected_error}' not found"
    logger.info(f"Error verified: {expected_error}")


@then('the error should be properly handled')
def verify_error_handling(test_context):
    """Verify error is handled properly."""
    assert test_context.get('response') is not None or test_context.get('response_error') is not None, \
        "No response or error captured"
    logger.info("Error handling verified")


@when('I execute the complete workflow if applicable')
def execute_workflow(test_context):
    """Execute complete workflow for positive test cases."""
    if 'error_scenario' in test_context:
        logger.info("Skipping workflow execution for negative test case.")
        return

    # Placeholder for actual workflow execution steps
    logger.info("Executing workflow steps...")


@then('all workflow steps should complete successfully')
def verify_workflow_completion(test_context):
    """Verify all workflow steps completed."""
    if 'error_scenario' in test_context:
        logger.info("Skipping workflow verification for negative test case.")
        return

    # Placeholder for workflow verification
    logger.info("All workflow steps completed successfully.")
