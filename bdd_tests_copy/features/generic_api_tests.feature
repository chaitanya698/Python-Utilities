Feature: Generic API Testing for Complaint Capture System

  Background:
    Given the test environment is configured
    And the API client is initialized

  @positive @api
  Scenario Outline: Execute positive API test case
    Given I have test data for test case "<test_case_id>"
    And the request response tracker is initialized for this test
    When I prepare the initial request from file "initial_request.json"
    And I send the initial API request
    Then the API response should be successful
    And the response should match expected result "<expected_result>"
    When I execute the complete workflow if applicable
    Then all workflow steps should complete successfully

    Examples:
      | test_case_id | expected_result |
      | TC001        | Success         |
      | TC002        | Success         |

  @negative @api
  Scenario Outline: Execute negative API test case
    Given I have test data for test case "<test_case_id>"
    And the request response tracker is initialized for this test
    When I prepare the initial request from file "initial_request.json"
    And I apply error scenario "<error_scenario>" if defined
    And I send the initial API request
    Then the API response should contain error "<expected_result>"
    And the error should be properly handled

    Examples:
      | test_case_id | error_scenario            | expected_result                  |
      | TC003        | missing_conversation_id   | Bad request - conversationID should be non-empty |
      | TC005        | empty_payload             | Invalid request format           |
      | TC006        | invalid_request_type      | Invalid RequestType              |
      | TC007        | invalid_action            | Invalid action                   |
      | TC008        | invalid_date              | Invalid date format              |
