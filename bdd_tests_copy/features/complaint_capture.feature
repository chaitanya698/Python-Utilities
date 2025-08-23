Feature: EO Escalated Complaint end-to-end capture
  Verify that the EO Escalated Complaint end-to-end complaint capture process
  is working using the AI Chatbot API.

  Background:
    Given the chatbot API is available and test data is loaded

  Scenario Outline: Verify complaint workflow from user input to database storage
    When the user responds with "<field_key>"
    Then the API response should prompt with "<expected_key>"

  Examples:
    | field_key  | expected_key               |
    | chatTest3  | account_number_response    |
    | chatTest4  | elaborate_quest_response   |
    | chatTest5  | followup_question_response |
    | chatTest6  | indicator_question_response|
    | chatTest7  | clarification_summary      |
    | chatText8  | preferred_comm_select_response |
    | chatText9  | preferred_comm_response    |
    | chatText10 | classification_summary     |
    | chatText11 | comp_submission_response   |

  Scenario: Verify final confirmation and persistence
    When the user responds with "chatText11"
    Then the final response should contain a confirmation and a valid interaction ID
    And verify the conversation details are stored properly in the Complaints AI database
    And verify the complaint details are stored properly in the Complaints database
