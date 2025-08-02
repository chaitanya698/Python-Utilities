# features/complaint_capture.feature

@complaintFlow
Feature: AI Chatbot Complaint Workflow

  Scenario: Verify end-to-end complaint capture process using data from an external source
    Given the chatbot API is available and test data is loaded
    When I send the initial complaint request
    Then the API response should be successful and contain a valid conversation ID
    And the conversation ID must exist in the database
    And the initial response action and text should be as expected

    When the user responds with the complaint date
    Then the API response should prompt for the method of complaint

    When the user responds with the method of complaint
    Then the API response should prompt for the account number

    When the user responds with the account number
    Then the API response should prompt for complaint details

    When the user responds with the complaint details
    Then the API response should contain a valid chat text

    When the user provides a final summary comment
    Then the API response should ask for clarification

    When the user confirms the summary
    Then the API response should ask for contact willingness

    When the user responds with their contact willingness
    Then the API response should prompt to submit the complaint

    When the user submits the complaint
    Then the final response should contain a confirmation and a valid Interaction ID

