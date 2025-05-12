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