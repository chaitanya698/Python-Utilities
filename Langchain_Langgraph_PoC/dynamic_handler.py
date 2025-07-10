def get_dynamic_response(question, step_text):
    """
    Parses the LLM question and provides a suitable response based on the test's intent.
    This is a simplified example.
    """
    question_lower = question.lower()
    
    # Example logic for Step 2 escalation questions
    if "misconduct" in question_lower or "sales practices" in question_lower:
        # If the Gherkin step text implies an affirmative answer...
        if "yes" in step_text.lower() or "confirm" in step_text.lower():
            return "Yes, the customer claims it involves Misconduct Sales."
        else:
            return "No, that is not the issue."
    
    # Fallback for generic Step 1 questions
    return "The customer did not provide more details on that."