from typing import Dict, Any, Optional, Callable
import copy
from datetime import datetime, timedelta

from .logger_config import get_logger


class ErrorInjector:
    """
    Helper class to inject errors into requests for negative testing.
    
    This class provides methods to modify valid requests to create
    various error scenarios for testing error handling and validation.
    """
    
    def __init__(self):
        """Initialize the ErrorInjector with logger and scenario mappings."""
        self.logger = get_logger(__name__)
        self._setup_error_scenarios()
    
    def _setup_error_scenarios(self):
        """Setup the mapping of error scenarios to injection functions."""
        self.error_scenarios: Dict[str, Callable] = {
            # Request structure errors
            'missing_conversation_id': self._remove_conversation_id,
            'missing_customer_name': self._remove_customer_name,
            'empty_payload': self._create_empty_payload,
            'invalid_request_type': self._set_invalid_request_type,
            'invalid_action': self._set_invalid_action,
            
            # Data validation errors
            'invalid_date': self._set_invalid_date_format,
            'future_date': self._set_future_date,
            'past_date': self._set_past_date,
            'null_chat_text': self._set_null_chat_text,
            'empty_chat_text': self._set_empty_chat_text,
            'junk_input': self._set_junk_characters,
            'special_characters': self._set_special_characters,
            
            # Missing required fields
            'missing_channel_id': self._remove_channel_id,
            'missing_request_type': self._remove_request_type,
            'missing_action': self._remove_action,
            'missing_data_elements': self._remove_data_elements,
            
            # Integration errors
            'no_ecn': self._remove_ecn,
            'no_employee_id': self._remove_employee_id,
            'invalid_ecn': self._set_invalid_ecn,
            'invalid_employee_id': self._set_invalid_employee_id,
            
            # Service simulation errors
            'service_unavailable': self._simulate_service_unavailable,
            'hr_service_unavailable': self._simulate_hr_unavailable,
            'customer_search_unavailable': self._simulate_customer_search_unavailable,
            'llm_unavailable': self._simulate_llm_unavailable,
            'llm_empty_response': self._simulate_llm_empty_response,
            'llm_malformed_response': self._simulate_llm_malformed_response,
            
            # Header-level errors (to be handled separately)
            'missing_correlation_id': self._no_modification,
            'invalid_correlation_id': self._no_modification,
            'missing_auth_header': self._no_modification,
        }
    
    def inject_error(self, request_data: Dict[str, Any], error_scenario: str) -> Dict[str, Any]:
        """
        Inject specific error based on scenario.
        
        Args:
            request_data: Original request data
            error_scenario: Name of the error scenario to inject
            
        Returns:
            Modified request data with error injected
        """
        if not error_scenario or error_scenario.lower() in ['none', '']:
            return request_data
        
        # Create a deep copy to avoid modifying original
        modified_request = copy.deepcopy(request_data)
        
        if error_scenario in self.error_scenarios:
            self.logger.info(f"Injecting error scenario: {error_scenario}")
            result = self.error_scenarios[error_scenario](modified_request)
            
            # Log the modification for debugging
            self.logger.debug(f"Original request: {request_data}")
            self.logger.debug(f"Modified request: {result}")
            
            return result if result is not None else modified_request
        else:
            self.logger.warning(f"Unknown error scenario: {error_scenario}")
            return modified_request
    
    def inject_multiple_errors(self, request_data: Dict[str, Any], error_scenarios: list) -> Dict[str, Any]:
        """
        Inject multiple error scenarios into a single request.
        
        Args:
            request_data: Original request data
            error_scenarios: List of error scenarios to inject
            
        Returns:
            Modified request data with all errors injected
        """
        modified_request = copy.deepcopy(request_data)
        
        for scenario in error_scenarios:
            modified_request = self.inject_error(modified_request, scenario)
        
        return modified_request
    
    # === Request Structure Errors ===
    
    def _remove_conversation_id(self, request_data: Dict) -> Dict:
        """Remove conversationId from request."""
        request_data.pop('conversationId', None)
        request_data.pop('conversationID', None)  # Handle both cases
        return request_data
    
    def _remove_customer_name(self, request_data: Dict) -> Dict:
        """Remove customerFullName from data elements."""
        return self._remove_data_element(request_data, 'customerFullName')
    
    def _create_empty_payload(self, request_data: Dict) -> Dict:
        """Return completely empty payload."""
        return {}
    
    def _set_invalid_request_type(self, request_data: Dict) -> Dict:
        """Set invalid request type."""
        request_data['requestType'] = 'InvalidType_' + str(datetime.now().timestamp())
        return request_data
    
    def _set_invalid_action(self, request_data: Dict) -> Dict:
        """Set invalid action."""
        request_data['action'] = 'invalid_action_test'
        return request_data
    
    # === Data Validation Errors ===
    
    def _set_invalid_date_format(self, request_data: Dict) -> Dict:
        """Set invalid date format in chat text."""
        request_data['chatText'] = 'MM/DD/YYYY'  # Invalid format
        return request_data
    
    def _set_future_date(self, request_data: Dict) -> Dict:
        """Set a future date in chat text."""
        future_date = datetime.now() + timedelta(days=365)
        request_data['chatText'] = future_date.strftime('%m/%d/%Y')
        return request_data
    
    def _set_past_date(self, request_data: Dict) -> Dict:
        """Set a very old date in chat text."""
        request_data['chatText'] = '01/01/1900'
        return request_data
    
    def _set_null_chat_text(self, request_data: Dict) -> Dict:
        """Set chat text to null."""
        request_data['chatText'] = None
        return request_data
    
    def _set_empty_chat_text(self, request_data: Dict) -> Dict:
        """Set chat text to empty string."""
        request_data['chatText'] = ''
        return request_data
    
    def _set_junk_characters(self, request_data: Dict) -> Dict:
        """Set junk characters in chat text."""
        request_data['chatText'] = '@#$%^&*()!~`[]{}|\\:;"\'<>?,./'
        return request_data
    
    def _set_special_characters(self, request_data: Dict) -> Dict:
        """Set special unicode characters in chat text."""
        request_data['chatText'] = '™€£¥©®§¶†‡•◊∆∑∏∫≈≠±∞'
        return request_data
    
    # === Missing Required Fields ===
    
    def _remove_channel_id(self, request_data: Dict) -> Dict:
        """Remove channelID from request."""
        request_data.pop('channelID', None)
        request_data.pop('channelId', None)  # Handle both cases
        return request_data
    
    def _remove_request_type(self, request_data: Dict) -> Dict:
        """Remove requestType from request."""
        request_data.pop('requestType', None)
        return request_data
    
    def _remove_action(self, request_data: Dict) -> Dict:
        """Remove action from request."""
        request_data.pop('action', None)
        return request_data
    
    def _remove_data_elements(self, request_data: Dict) -> Dict:
        """Remove all data elements from request."""
        request_data.pop('dataElements', None)
        return request_data
    
    # === Integration Errors ===
    
    def _remove_ecn(self, request_data: Dict) -> Dict:
        """Remove ECN from data elements."""
        return self._remove_data_element(request_data, 'ECN')
    
    def _remove_employee_id(self, request_data: Dict) -> Dict:
        """Remove submitterEmployeeID from data elements."""
        return self._remove_data_element(request_data, 'submitterEmployeeID')
    
    def _set_invalid_ecn(self, request_data: Dict) -> Dict:
        """Set invalid ECN format."""
        return self._update_data_element(request_data, 'ECN', 'INVALID_ECN_123')
    
    def _set_invalid_employee_id(self, request_data: Dict) -> Dict:
        """Set invalid employee ID format."""
        return self._update_data_element(request_data, 'submitterEmployeeID', 'INVALID_EMP')
    
    # === Service Simulation Errors ===
    
    def _simulate_service_unavailable(self, request_data: Dict) -> Dict:
        """Add flag to simulate general service unavailability."""
        request_data['simulateError'] = 'service_unavailable'
        return request_data
    
    def _simulate_hr_unavailable(self, request_data: Dict) -> Dict:
        """Add flag to simulate HR service unavailability."""
        request_data['simulateError'] = 'hr_service_unavailable'
        return request_data
    
    def _simulate_customer_search_unavailable(self, request_data: Dict) -> Dict:
        """Add flag to simulate Customer Search service unavailability."""
        request_data['simulateError'] = 'customer_search_unavailable'
        return request_data
    
    def _simulate_llm_unavailable(self, request_data: Dict) -> Dict:
        """Add flag to simulate LLM service unavailability."""
        request_data['simulateError'] = 'llm_unavailable'
        return request_data
    
    def _simulate_llm_empty_response(self, request_data: Dict) -> Dict:
        """Add flag to simulate empty LLM response."""
        request_data['simulateError'] = 'llm_empty_response'
        return request_data
    
    def _simulate_llm_malformed_response(self, request_data: Dict) -> Dict:
        """Add flag to simulate malformed LLM response."""
        request_data['simulateError'] = 'llm_malformed_response'
        return request_data
    
    # === Helper Methods ===
    
    def _no_modification(self, request_data: Dict) -> Dict:
        """Return request unchanged (for header-level errors)."""
        return request_data
    
    def _remove_data_element(self, request_data: Dict, element_name: str) -> Dict:
        """Remove specific data element from request."""
        if 'dataElements' in request_data and request_data['dataElements']:
            request_data['dataElements'] = [
                e for e in request_data['dataElements'] 
                if e.get('name') != element_name
            ]
        return request_data
    
    def _update_data_element(self, request_data: Dict, element_name: str, new_value: Any) -> Dict:
        """Update specific data element value."""
        if 'dataElements' in request_data and request_data['dataElements']:
            for element in request_data['dataElements']:
                if element.get('name') == element_name:
                    element['value'] = new_value
                    break
            else:
                # If element doesn't exist, add it
                request_data['dataElements'].append({
                    'name': element_name,
                    'value': new_value
                })
        return request_data
    
    def get_available_scenarios(self) -> list:
        """Get list of all available error scenarios."""
        return list(self.error_scenarios.keys())
    
    def describe_scenario(self, scenario: str) -> str:
        """Get description of what an error scenario does."""
        descriptions = {
            'missing_conversation_id': 'Removes conversationId from request',
            'missing_customer_name': 'Removes customerFullName from data elements',
            'empty_payload': 'Returns completely empty request payload',
            'invalid_request_type': 'Sets requestType to an invalid value',
            'invalid_action': 'Sets action to an invalid value',
            'invalid_date': 'Sets date to invalid format (MM/DD/YYYY)',
            'future_date': 'Sets date to a future date (1 year from now)',
            'past_date': 'Sets date to very old date (01/01/1900)',
            'null_chat_text': 'Sets chatText to null',
            'empty_chat_text': 'Sets chatText to empty string',
            'junk_input': 'Sets chatText to junk/special characters',
            'service_unavailable': 'Simulates service unavailability',
            'llm_unavailable': 'Simulates LLM service unavailability',
            'llm_empty_response': 'Simulates empty response from LLM',
        }
        return descriptions.get(scenario, 'No description available')
