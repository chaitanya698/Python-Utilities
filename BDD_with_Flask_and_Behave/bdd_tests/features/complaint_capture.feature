@complaintFlow
Feature: AI Chatbot Complaint Capture Workflow

  As a bank teller, I need to use the chatbot to accurately capture a customer's complaint so that it can be processed correctly.

  Scenario Outline: Full complaint capture flow for various complaint types
    Given the chatbot API is available for "<channel>"
    When I start a new complaint conversation for "<complainant_name>"
    And I provide the complaint received date as "10/07/2024"
    And I provide the complaint received method as "<method>"
    And I provide the account number
    Then the final summary should be generated correctly

    Examples:
      | channel | complainant_name | method |
      | BBVA    | John Doe         | Phone  |
      | CHASE   | Jane Smith       | Web    |