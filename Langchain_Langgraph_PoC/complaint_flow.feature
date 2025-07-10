Feature: AI Chatbot Complaint Capture Workflow

  Scenario Outline: Full complaint capture flow for Misconduct Sales
    Given the user is a bank teller starting a new complaint
    When the teller starts a "ComplaintCapture" request
    And the teller provides the complaint date as "yesterday"
    And the teller provides the receipt method as "Phone"
    And the teller provides the account number
    And the teller submits the initial complaint summary: "<complaint_summary>"
    Then the bot should ask a Step 1 clarification question

    When the teller answers the Step 1 question with "<step1_answer>"
    Then the bot should ask a Step 2 escalation detection question

    When the teller answers the Step 2 question with "<step2_answer>"
    Then the bot provides the final summary and closes the conversation
    And the final summary should correctly identify the issue as "<expected_reason>"

    Examples:
      | complaint_summary                                       | step1_answer                         | step2_answer | expected_reason   |
      | "The customer says they were denied a loan but have a    | "No other issues were mentioned."    | "Yes"        | "Misconduct Sales"|
      | good credit score. They believe it is due to misconduct"|                                      |              |                   |