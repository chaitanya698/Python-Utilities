# rules.py      (tiny example so the demo runs)
keywords = {
    "lost card":          "card_block",
    "account balance":    "balance_inquiry",
    "transfer":           "fund_transfer",
}

def classify(text: str) -> str:
    text = text.lower()
    for kw, tag in keywords.items():
        if kw in text:
            return tag
    return None
