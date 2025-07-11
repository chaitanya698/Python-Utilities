@ComplaintFlow
Feature: AI Chatbot Complaint Capture Workflow

  As a bank teller, I need to use the chatbot to accurately capture a customer's complaint
  so that it can be processed correctly.

  Scenario Outline: Full complaint capture flow for various complaint types
    Given the chatbot API is available for "<channel>"
    When I start a new complaint conversation for "<complainant_name>"
    And I provide the complaint received date as "10/07/2025"
    And I provide the complaint received method as "<method>"
    And I provide the account number
    And I describe the initial complaint about "<initial_complaint>"
    And I answer the AI's first clarifying question regarding the complaint
    And I answer the AI's second escalation question
    Then the chatbot should generate a final summary containing the key details

    Examples:
      | channel | complainant_name | method | initial_complaint                                    |
      | BBVA    | John Doe         | Phone  | "Loan application denied despite good credit score"    |
      | BBVA    | Jane Smith       | Email  | "Unauthorised transaction on my credit card"           |