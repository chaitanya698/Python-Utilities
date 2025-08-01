# bdd_tests/features/complaint_capture.feature

@complaintFlow
Feature: AI Chatbot Complaint Workflow

  Scenario: Verify initial complaint request creates a conversation and is saved to the database
    Given the chatbot API is available
    When I send the initial complaint request from "initial_request.json"
    Then the API response should be successful
    And the response should contain a valid conversation ID
    And the response action should be to "proceed" with label "Confirm date"
    And the chat response text should be "When was the complaint received?"
    And the conversation ID should follow the pattern "CVD-########-####-####-####-############"
    And the initial chat interaction should be saved in the database
