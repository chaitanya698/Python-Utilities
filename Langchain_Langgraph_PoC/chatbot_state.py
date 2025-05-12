from typing import List, Optional, TypedDict


class ChatState(TypedDict, total=False):
    # ── rolling chat log ───────────────────────────
    messages: List[str]            # "user: …", "assistant: …"
    user_input: str
    assistant_response: str

    # ── control flags ──────────────────────────────
    awaiting: Optional[str]        # "confirm" when the bot awaits yes/no

    # ── business data ──────────────────────────────
    intent: Optional[str]
    confirmed: Optional[bool]
    result: Optional[str]

    # ── error channel ──────────────────────────────
    error_msg: Optional[str]

    done: bool                     # set True to exit the CLI loop
