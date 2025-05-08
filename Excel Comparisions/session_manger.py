import logging
from typing import Dict, Any, List, Optional

from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

logger = logging.getLogger('SessionManager_logger')

class Message:
    def __init__(self, content, is_ai=False):
        self.content = content
        self.is_ai = is_ai

class SessionManager:
    def __init__(self):
        # Dictionary to store session data
        self.session_data: Dict[str, ConversationBufferMemory] = {}
        # Dictionary to store last messages
        self.last_message: Dict[str, Message] = {}
        # Dictionary to store last response metadata
        self.last_response_metadata: Dict[str, Dict[str, Any]] = {}
    
    def create_session(self, session_id: str) -> None:
        """Create a new session with the given ID."""
        if session_id not in self.session_data:
            self.session_data[session_id] = ConversationBufferMemory(
                memory_key="chat_history",
                return_messages=True
            )
            logger.info(f"Session created: {session_id}")
        else:
            logger.info(f"Session already exists: {session_id}")
    
    def validate_session(self, session_id: str) -> bool:
        """Check if the session ID is valid."""
        return session_id in self.session_data
    
    def get_session_history(self, session_id: str) -> ConversationBufferMemory:
        """Get the conversation history for a session."""
        if not self.validate_session(session_id):
            raise ValueError(f"Invalid session ID: {session_id}")
        
        return self.session_data[session_id]
    
    def add_message_to_session(self, session_id: str, human_message: str, ai_message: str = None) -> None:
        """Add messages to the session history."""
        if not self.validate_session(session_id):
            raise ValueError(f"Invalid session ID: {session_id}")
        
        memory = self.session_data[session_id]
        
        # Add human message
        memory.chat_memory.add_message(HumanMessage(content=human_message))
        self.last_message[session_id] = Message(human_message, is_ai=False)
        
        # Add AI message if provided
        if ai_message:
            memory.chat_memory.add_message(AIMessage(content=ai_message))
            self.last_response_metadata[session_id] = {'response': ai_message}
    
    def get_last_message(self, session_id: str) -> Optional[Message]:
        """Get the last message for a session."""
        return self.last_message.get(session_id)
    
    def get_last_response_metadata(self, session_id: str) -> Dict[str, Any]:
        """Get metadata for the last response in a session."""
        return self.last_response_metadata.get(session_id, {})
    
    def get_last_response_metadata_as_string(self, session_id: str) -> str:
        """Get metadata for the last response as a string."""
        metadata = self.get_last_response_metadata(session_id)
        if metadata:
            return str(metadata)
        return ""
    
    def update_last_human_message(self, session_id: str, new_message: str, new_response: str) -> None:
        """Update the last human message and response in the session."""
        if session_id in self.last_message:
            self.last_message[session_id].content = new_message
            self.last_response_metadata[session_id] = {'response': new_response}
    
    def clear_session(self, session_id: str) -> None:
        """Clear a specific session."""
        if session_id in self.session_data:
            del self.session_data[session_id]
        
        if session_id in self.last_message:
            del self.last_message[session_id]
        
        if session_id in self.last_response_metadata:
            del self.last_response_metadata[session_id]
        
        logger.info(f"Session cleared: {session_id}")
    
    def clear_sessions_data(self) -> None:
        """Clear all session data."""
        self.session_data.clear()
        self.last_message.clear()
        self.last_response_metadata.clear()
        logger.info("All sessions cleared")
    
    def get_messages_as_langchain_format(self, session_id: str) -> List[Dict[str, str]]:
        """Get messages in a format suitable for LangChain."""
        if not self.validate_session(session_id):
            raise ValueError(f"Invalid session ID: {session_id}")
        
        memory = self.session_data[session_id]
        messages = memory.chat_memory.messages
        
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append({"role": "user", "content": message.content})
            elif isinstance(message, AIMessage):
                formatted_messages.append({"role": "assistant", "content": message.content})
            elif isinstance(message, SystemMessage):
                formatted_messages.append({"role": "system", "content": message.content})
        
        return formatted_messages