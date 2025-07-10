INTENT_KEYWORDS = {
    
    "card_issue": [
        "lost card", "stolen card", "block my card", "card replacement",
        "card fraud", "unauthorised", "credit card", "debit card"
    ],

    
    "balance": [
        "balance", "available funds", "how much money", "statement"
    ],

    
    "transfer": [
        "transfer", "send money", "wire", "imps", "neft", "rtgs",
        "pay", "payment to"
    ],

    
    "loan": [
        "loan", "mortgage", "emi", "interest rate", "installment"
    ],

    
    "fees": [
        "fee", "charges", "overdraft", "maintenance fee"
    ],
}

DEFAULT_INTENT = "other"

def classify(text: str) -> str :
    """Return the first matching intent label (or None)."""
    text_l = text.lower()
    for label, phrases in INTENT_KEYWORDS.items():
        if any(p in text_l for p in phrases):
            return label
    return None



complaint_summary,step1_answer,step2_answer,expected_reason
"The customer says they were denied a loan despite a good credit score. They believe it is due to misconduct from the sales team.","No other issues were mentioned.","Yes","Misconduct Sales"
"A customer is reporting an unauthorized ACH debit from their checking account for $250. They did not authorize this payment and want it reversed.","The transaction was dated two days ago.","Yes","Regulation E"
"The customer feels they were treated unfairly and given dirty looks when applying for a new checking account because of how they were dressed.","It happened at the main downtown branch last Tuesday.","Yes","Discrimination"
"The customer mentioned that an employee at the branch was rude and dismissive, and they want to file a formal complaint about the unprofessional behavior.","The employee's name was not mentioned.","Yes","Unethical Behavior"
"My new debit card was supposed to arrive in 3 days but it's been a week and it's still not here. The tracking number does not work.","This is the first time I've had an issue with card delivery.","No","No escalation concerns identified"
"A customer is complaining that the carpet is dirty and there's a strange smell in the downtown branch lobby.","The customer visits the branch weekly.","Yes","LOB Escalated Complaint"
"The customer was told their monthly account fee would always be waived, but this month they were charged. They are very upset.","They have been a customer for over 10 years.","Yes","Misleading Information"
