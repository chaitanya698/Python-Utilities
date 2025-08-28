Feature: Complaint Capture Workflow
  As a user I want to capture customer complaints through the chatbot
  So that complaints can be properly recorded and tracked

  Background:
    Given the chatbot API is available and the test data is loaded

  @smoke
  Scenario Outline: Complaints AI API Test Case: Complaint Capture End to End Workflow
    When I send the initial complaint request
    Then the API response should be successful as "<when_date_response>" and contain a valid conversation ID
    And the initial response action and text should be as expected

    When the user responds with the complaint date as "<chatText1>"
    Then the API response should prompt for the method of complaint as "<how_comp_response>"

    When the user responds with the method of complaint as "<chatTest2>"
    Then the API response should prompt for the account number "<account_number_select_response>"

    When the user responds with "<chatTest3>" for Account Number
    Then the API response should prompt to enter Account Number "<account_number_response>"

    When the user responds with the Account Number as "<chatTest4>"
    Then the API response should prompt the user with "<elaborate_quest_response>"

    When the user provides detailed description of the complaint as "<chatText5>"
    Then the API response should prompt with a followup question from LLM

    When the user responds to the followup question as "<chatText6>"
    Then the API response should prompt with a followup indicator question

    When the user responds to the risk indicator question as "<chatText7>"
    Then the API response should return the clarification Summary

    When the user responds with proceed as "<chatText8>"
    Then the API response should prompt for "<preferred_comm_select_response>"

    When the user responds to preferred communication as "<chatText9>"
    Then the API response should prompt to "<preferred_comm_response>"

    When the user responds to communication details as "<chatText10>"
    Then the API response should return the classification Summary

    When the user responds with proceed as "<chatText11>"
    Then the API response should prompt with the Complaint Submission details "<comp_submission_response>"
    And verify the conversation details are stored properly in the Complaints AI database
    And verify the complaint details are stored properly in the Complaints database
