from typing import List, Optional, TypedDict


class ChatState(TypedDict, total=False):
    
    messages: List[str]            
    user_input: str
    assistant_response: str

    
    awaiting: Optional[str]        
    done: bool                     

    
    intent: Optional[str]
    confirmed: Optional[bool]
    result: Optional[str]

    
    exit_requested: bool   
    history_requested: bool                   

    
    error_msg: Optional[str]
