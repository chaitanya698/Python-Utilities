
Feature: Complaint Capture Workflow
  As a user
  I want to capture customer complaints through the chatbot
  So that complaints can be properly recorded and tracked
  
  @regression @complaint
  Scenario: Verify end-to-end complaint capture process for a new complaint
    Given the chatbot API is available
    And I have the complaint data for test case "<test_case_id>"
    
    When I send the initial complaint request
    Then the API response should be successful and contain a valid conversation ID
    And the initial response action should be "<initial_action>" and text should be "<initial_text>"

    When the user responds with the complaint date
    Then the API response should prompt for the method of complaint with action "<method_prompt_action>"

    When the user responds with the method of complaint
    Then the API response should prompt for the account number with action "<account_prompt_action>"

    When the user responds with the account number
    Then the API response should prompt for complaint details with action "<details_prompt_action>"

    When the user responds with the complaint details
    Then the API response should ask for clarification with action "<clarification_action>"

    When the user provides a final summary comment
    Then the API response should prompt for contact willingness with action "<contact_willingness_action>"

    When the user confirms they want to be contacted
    Then the final response corresponds with the complaint classification "<final_classification>"
    And the final response should contain a confirmation and a valid Interaction ID
