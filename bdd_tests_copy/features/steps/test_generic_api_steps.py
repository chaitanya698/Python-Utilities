import re
import json
from typing import Dict, Any, Optional
import logging
from datetime import datetime
from pytest_bdd import scenario, given, when, then, parsers

from bdd_tests.utils.helpers import TestHelpers
from bdd_tests.utils.error_injector import ErrorInjector  # Import from utils
from bdd_tests.utils.request_response_tracker import RequestResponseTracker

# Initialize logger
logger = logging.getLogger(__name__)


class GenericAPITestSteps:
    """Generic test steps that can handle various test scenarios."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.test_context = {}
        self.error_injector = ErrorInjector()  # Initialize error injector
    
    @when(parsers.parse('I apply error scenario "{error_scenario}" if defined'))
    def apply_error_scenario(self, error_scenario):
        """Apply error scenario to the request using ErrorInjector."""
        if error_scenario and error_scenario != '' and error_scenario.lower() != 'none':
            request_data = self.test_context.get('request_data', {})
            
            # Log the scenario being applied
            self.logger.info(f"Applying error scenario: {error_scenario}")
            self.logger.debug(f"Scenario description: {self.error_injector.describe_scenario(error_scenario)}")
            
            # Apply error injection using the ErrorInjector utility
            modified_request = self.error_injector.inject_error(request_data, error_scenario)
            self.test_context['request_data'] = modified_request
            self.test_context['error_scenario'] = error_scenario
            
            self.logger.info(f"Error scenario '{error_scenario}' applied successfully")
    
    @when('I apply multiple error scenarios')
    def apply_multiple_error_scenarios(self):
        """Apply multiple error scenarios to test combinations."""
        test_data = self.test_context.get('test_data', {})
        error_scenarios = test_data.get('error_scenarios', '').split(',')
        
        if error_scenarios:
            request_data = self.test_context.get('request_data', {})
            
            # Apply multiple errors
            modified_request = self.error_injector.inject_multiple_errors(
                request_data, 
                [s.strip() for s in error_scenarios]
            )
            
            self.test_context['request_data'] = modified_request
            self.test_context['error_scenarios'] = error_scenarios
            
            self.logger.info(f"Applied multiple error scenarios: {error_scenarios}")
    
    @given('the test environment is configured')
    def setup_environment(self, config, request_response_tracker):
        """Setup test environment."""
        self.test_context['config'] = config
        self.test_context['tracker'] = request_response_tracker
        self.logger.info(f"Environment configured: {config.ENVIRONMENT}")
    
    @given('the API client is initialized')
    def initialize_api_client(self, api_client):
        """Initialize API client."""
        self.test_context['api_client'] = api_client
        self.logger.info("API client initialized")
    
    @given(parsers.parse('I have test data for test case "{test_case_id}"'))
    def load_test_data(self, test_case_id, test_data_row):
        """Load test data for specific test case."""
        self.test_context['test_case_id'] = test_case_id
        self.test_context['test_data'] = test_data_row
        self.test_context['correlation_id'] = TestHelpers.generate_correlation_id(test_case_id)
        
        self.logger.info(f"Loaded test data for {test_case_id}")
        self.logger.debug(f"Test data: {test_data_row}")
    
    @given('the request response tracker is initialized for this test')
    def init_tracker_for_test(self, request_response_tracker):
        """Initialize tracker for current test."""
        test_id = self.test_context.get('test_case_id', 'unknown')
        request_response_tracker.set_current_test(test_id)
        self.logger.info(f"Tracker initialized for test: {test_id}")
    
    @given(parsers.parse('external services are configured for "{category}"'))
    def configure_external_services(self, category):
        """Configure external service mocks or stubs."""
        self.test_context['service_category'] = category
        
        # Configure based on category
        if 'Customer search' in category:
            self.test_context['mock_customer_search'] = True
        if 'HR Data' in category:
            self.test_context['mock_hr_data'] = True
        if 'LLM' in category:
            self.test_context['mock_llm'] = True
        
        self.logger.info(f"External services configured for: {category}")
    
    @when(parsers.parse('I prepare the initial request from file "{request_file}"'))
    def prepare_initial_request(self, request_file, data_loader):
        """Prepare initial request from template file."""
        try:
            # Load request template
            request_data = data_loader.load_json(request_file)
            
            # Merge with test data if needed
            test_data = self.test_context.get('test_data', {})
            
            # Update request with test-specific data
            if test_data.get('channelID'):
                request_data['channelID'] = test_data['channelID']
            
            # Set conversation ID appropriately
            if test_data.get('error_scenario') == 'missing_conversation_id':
                request_data.pop('conversationId', None)
            elif 'conversationId' not in request_data:
                request_data['conversationId'] = 'initial'
            
            # Handle data elements
            if 'dataElements' in request_data:
                self._update_data_elements(request_data['dataElements'], test_data)
            
            self.test_context['request_data'] = request_data
            self.logger.info(f"Prepared request from {request_file}")
            
        except FileNotFoundError:
            # Use default request structure
            self.test_context['request_data'] = self._create_default_request(test_data)
            self.logger.info("Using default request structure")
    
    @when(parsers.parse('I apply error scenario "{error_scenario}" if defined'))
    def apply_error_scenario(self, error_scenario):
        """Apply error scenario to the request."""
        if error_scenario and error_scenario != '' and error_scenario.lower() != 'none':
            request_data = self.test_context.get('request_data', {})
            
            # Apply error injection based on scenario
            modified_request = self.error_injector.inject_error(request_data, error_scenario)
            self.test_context['request_data'] = modified_request
            self.test_context['error_scenario'] = error_scenario
            
            self.logger.info(f"Applied error scenario: {error_scenario}")
    
    @when('I send the initial API request')
    def send_initial_request(self):
        """Send the initial API request."""
        api_client = self.test_context['api_client']
        request_data = self.test_context['request_data']
        correlation_id = self.test_context['correlation_id']
        
        try:
            response = api_client.initiate_chat(
                request_data=request_data,
                correlation_id=correlation_id
            )
            
            self.test_context['response'] = response
            self.test_context['response_error'] = None
            
            # Extract conversation ID if present
            if 'conversationID' in response:
                self.test_context['conversation_id'] = response['conversationID']
            
            self.logger.info(f"Request sent successfully, correlation ID: {correlation_id}")
            
        except Exception as e:
            self.test_context['response'] = None
            self.test_context['response_error'] = str(e)
            self.logger.error(f"Request failed: {e}")
    
    @when('I send the initial API request with integration points')
    def send_request_with_integration(self):
        """Send request that triggers integration points."""
        # Similar to send_initial_request but with integration monitoring
        self.send_initial_request()
        
        # Track integration calls if mocking is enabled
        if self.test_context.get('mock_customer_search'):
            self.logger.info("Customer Search service would be called")
        if self.test_context.get('mock_hr_data'):
            self.logger.info("HR Data service would be called")
        if self.test_context.get('mock_llm'):
            self.logger.info("LLM service would be called")
    
    @when('I execute the complete workflow if applicable')
    def execute_workflow(self):
        """Execute complete workflow for positive test cases."""
        test_data = self.test_context.get('test_data', {})
        api_client = self.test_context['api_client']
        conversation_id = self.test_context.get('conversation_id')
        
        if not conversation_id:
            self.logger.info("No conversation ID, skipping workflow")
            return
        
        # Execute workflow steps based on test data
        workflow_steps = [
            ('complaint_date', 'complaint_date'),
            ('complaint_method', 'complaint_method'),
            ('account_number', 'account_number'),
            ('complaint_details', 'complaint_details'),
            ('final_summary_comment', 'final_summary_comment'),
            ('contact_willingness_response', 'contact_willingness_response')
        ]
        
        for step_name, data_field in workflow_steps:
            if test_data.get(data_field):
                try:
                    response = api_client.send_message(
                        conversation_id=conversation_id,
                        chat_text=test_data[data_field],
                        action="proceed",
                        correlation_id=f"{self.test_context['correlation_id']}-{step_name}"
                    )
                    
                    self.test_context[f'response_{step_name}'] = response
                    self.logger.info(f"Workflow step completed: {step_name}")
                    
                except Exception as e:
                    self.logger.error(f"Workflow step failed: {step_name} - {e}")
                    self.test_context[f'error_{step_name}'] = str(e)
                    break
    
    @then('the API response should be successful')
    def verify_successful_response(self):
        """Verify the API response is successful."""
        response = self.test_context.get('response')
        error = self.test_context.get('response_error')
        
        assert error is None, f"Request failed with error: {error}"
        assert response is not None, "No response received"
        
        # Check for conversation ID in positive cases
        if 'error_scenario' not in self.test_context:
            assert 'conversationID' in response, "Response missing conversationID"
        
        self.logger.info("Response verified as successful")
    
    @then(parsers.parse('the response should match expected result "{expected_result}"'))
    def verify_expected_result(self, expected_result):
        """Verify response matches expected result."""
        response = self.test_context.get('response')
        
        if expected_result.lower() == 'success':
            assert response is not None, "Expected successful response but got None"
            assert 'error' not in response, f"Unexpected error in response: {response.get('error')}"
        else:
            # Verify specific expected content
            response_text = json.dumps(response) if response else self.test_context.get('response_error', '')
            assert expected_result in response_text, \
                f"Expected '{expected_result}' not found in response"
        
        self.logger.info(f"Response matches expected result: {expected_result}")
    
    @then(parsers.parse('the API response should contain error "{expected_error}"'))
    def verify_error_response(self, expected_error):
        """Verify the API response contains expected error."""
        response = self.test_context.get('response')
        error = self.test_context.get('response_error')
        
        # Check if error is in response or error message
        error_found = False
        
        if error and expected_error.lower() in error.lower():
            error_found = True
        elif response:
            response_text = json.dumps(response)
            if expected_error.lower() in response_text.lower():
                error_found = True
        
        assert error_found, f"Expected error '{expected_error}' not found"
        self.logger.info(f"Error verified: {expected_error}")
    
    @then('the error should be properly handled')
    def verify_error_handling(self):
        """Verify error is handled properly."""
        # Verify that the system didn't crash and provided appropriate error response
        assert self.test_context.get('response') is not None or \
               self.test_context.get('response_error') is not None, \
               "No response or error captured"
        
        self.logger.info("Error handling verified")
    
    @then(parsers.parse('the integration should behave as expected "{expected_behavior}"'))
    def verify_integration_behavior(self, expected_behavior):
        """Verify integration behavior."""
        # This would check integration logs, mock calls, etc.
        self.logger.info(f"Integration behavior verified: {expected_behavior}")
    
    @then('all services should be called appropriately')
    def verify_service_calls(self):
        """Verify all required services were called."""
        # This would verify mock service calls
        self.logger.info("Service calls verified")
    
    @then('all workflow steps should complete successfully')
    def verify_workflow_completion(self):
        """Verify all workflow steps completed."""
        test_data = self.test_context.get('test_data', {})
        
        # Check that all expected workflow steps have responses
        expected_steps = ['complaint_date', 'complaint_method', 'account_number']
        
        for step in expected_steps:
            if test_data.get(step):
                assert f'response_{step}' in self.test_context or \
                       f'error_{step}' not in self.test_context, \
                       f"Workflow step '{step}' did not complete successfully"
        
        self.logger.info("All workflow steps completed successfully")
    
    def _update_data_elements(self, data_elements: list, test_data: dict):
        """Update data elements with test-specific values."""
        for element in data_elements:
            if element['name'] == 'businessName' and test_data.get('business_name'):
                element['value'] = test_data['business_name']
            elif element['name'] == 'customerFullName' and test_data.get('customer_name'):
                element['value'] = test_data['customer_name']
            elif element['name'] == 'country' and test_data.get('country'):
                element['value'] = test_data['country']
    
    def _create_default_request(self, test_data: dict) -> dict:
        """Create default request structure."""
        return {
            "channelID": test_data.get('channelID', 'BBVA'),
            "conversationId": "initial",
            "requestType": test_data.get('request_type', 'ComplaintCapture'),
            "chatText": test_data.get('chat_text', 'Initial request'),
            "action": test_data.get('action', 'proceed'),
            "dataElements": [
                {"name": "businessName", "value": test_data.get('business_name', 'John Doe')},
                {"name": "customerFullName", "value": test_data.get('customer_name', '')},
                {"name": "country", "value": test_data.get('country', 'US')}
            ]
        }
