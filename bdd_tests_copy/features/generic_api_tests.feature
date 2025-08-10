Feature: Generic API Testing for Complaint Capture System

Background:
    Given the test environment is configured
    And the API client is initialized

@positive @api
Scenario Outline: Execute positive API test case
    Given I have test data for test case "<test_case_id>"
    And the request response tracker is initialized for this test
    When I prepare the initial request from file "<initial_request_file>"
    And I send the initial API request
    Then the API response should be successful
    And the response should match expected result "<expected_result>"
    When I execute the complete workflow if applicable
    Then all workflow steps should complete successfully

@negative @api
Scenario Outline: Execute negative API test case
    Given I have test data for test case "<test_case_id>"
    And the request response tracker is initialized for this test
    When I prepare the initial request from file "<initial_request_file>"
    And I apply error scenario "<error_scenario>" if defined
    And I send the initial API request
    Then the API response should contain error "<expected_result>"
    And the error should be properly handled

@integration
Scenario Outline: Execute integration test case
    Given I have test data for test case "<test_case_id>"
    And the request response tracker is initialized for this test
    And external services are configured for "<category>"
    When I prepare the initial request from file "<initial_request_file>"
    And I send the initial API request with integration points
    Then the integration should behave as expected "<expected_result>"
    And all services should be called appropriately

Examples:
    | test_case_id | category | initial_request_file | error_scenario | expected_result |
    | From CSV     | From CSV | From CSV            | From CSV       | From CSV        |
"""
