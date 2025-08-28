Feature: Complaint Capture End-to-End Workflow
  As a user I want to capture customer complaints through the chatbot
  So that complaints can be properly recorded and tracked

  Background:
    Given the chatbot API is available and test data is loaded
    And the expected responses are loaded from "complaint_api_expected_response.json"

  @smoke @regression
  Scenario Outline: Execute complaint capture workflow for test case "<test_case_id>"
    Given I have test case "<test_case_id>" with data from CSV
    When I send the initial complaint request
    Then the API response should be successful and contain a valid conversation ID
    And the initial response action and text should be as expected
    
    # Step 1: Complaint Date
    When I respond with complaint date from "chatText1" if available
    Then the API response should match expected key "show_comp_response" if step was executed
    
    # Step 2: Complaint Method  
    When I respond with complaint method from "chatText2" if available
    Then the API response should match expected key "account_number_select_response" if step was executed
    
    # Step 3: Account Number Selection
    When I respond with account number option from "chatText3" if available  
    Then the API response should match expected key "account_number_response" if step was executed
    
    # Step 4: Account Number Entry
    When I respond with account number from "chatText4" if available
    Then the API response should match expected key "elaborate_quest_response" if step was executed
    
    # Step 5: Complaint Details
    When I provide complaint description from "chatText5" if available
    Then the API response should contain a followup question from LLM if step was executed
    
    # Step 6: Followup Question Response
    When I respond to followup question from "chatText6" if available
    Then the API response should contain a followup indicator question if step was executed
    
    # Step 7: Risk Indicator Response
    When I respond to risk indicator from "chatText7" if available
    Then the API response should return the clarification summary if step was executed
    
    # Step 8: Proceed Decision
    When I respond with proceed from "chatText8" if available
    Then the API response should match expected key "preferred_comm_select_response" if step was executed
    
    # Step 9: Communication Preference
    When I respond with communication preference from "chatText9" if available
    Then the API response should match expected key "preferred_comm_response" if step was executed
    
    # Step 10: Communication Details
    When I respond with communication details from "chatText10" if available
    Then the API response should return the classification summary if step was executed
    
    # Step 11: Final Submission
    When I respond with final proceed from "chatText11" if available
    Then the API response should match expected key "comp_submission_response" if step was executed
    And verify the conversation details are stored properly in the Complaints AI database
    And verify the complaint details are stored properly in the Complaints database

    Examples:
      | test_case_id |
      | TC001        |
      | TC002        |
      | TC004        |
      | TC013        |
      | TC014        |
      | TC017        |
      | TC018        |
