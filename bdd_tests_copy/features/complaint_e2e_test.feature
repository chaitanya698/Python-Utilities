Feature: Complaint Capture End-to-End Workflow
  As a user, I want to capture customer complaints through the chatbot
  so that they can be properly recorded and tracked.

  @smoke @regression
  Scenario Outline: Execute complaint capture workflow for test case "<test_case_id>"
    Given I have the test data for test case "<test_case_id>"
    When I send the initial complaint request
    Then the initial API response should be successful and contain a valid conversation ID

    When I execute the dynamic complaint capture workflow
    Then all executed steps should be validated against their expected responses

    And the conversation and complaint details should be verified in the database

  Examples:
    | test_case_id |
    | TC001        |
    | TC002        |
    | TC004        |
