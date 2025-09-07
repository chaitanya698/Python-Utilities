Feature: Complaint Capture End-to-End Workflow
  As a user I want to capture customer complaints through the chatbot
  So that complaints can be properly recorded and tracked

  Background:
    Given the chatbot API is available and test data is loaded
    And the expected responses are loaded from "complaint_api_expected_response.json"

  @smoke @regression
  Scenario Outline: Execute complaint capture workflow for test case "<test_case_id>"
    Given I have test case "<test_case_id>" with data from CSV
    When I send the initial complaint request with error scenario if specified
    Then the API response should be validated based on expected initial response
    
    # Dynamic step execution based on CSV data availability
    When I execute dynamic workflow steps based on available data
    Then all executed steps should be validated against expected responses
    
    # Final verification steps
    When the workflow is complete
    Then verify the conversation details are stored properly in the Complaints AI database if applicable
    And verify the complaint details are stored properly in the Complaints database if applicable
