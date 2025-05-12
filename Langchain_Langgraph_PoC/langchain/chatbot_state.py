
from typing import List, Optional, TypedDict

class ChatState(TypedDict, total=False):
    messages: List[str]
    user_input: str
    assistant_response: str

    awaiting: Optional[str]        
    intent: Optional[str]
    confirmed: Optional[bool]
    result: Optional[str]
    error_msg: Optional[str]

    done: bool
