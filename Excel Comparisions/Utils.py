import uuid
import logging
import json
import os
import re
from typing import Dict, Any, Optional, List

logger = logging.getLogger("util")

class Util:
    """Utility functions for the complaint system"""
    
    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID"""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_human_readable_complaint_id() -> str:
        """Generate a human-readable complaint ID"""
        return f"COMP-{uuid.uuid4().hex[:8].upper()}"
    
    @staticmethod
    def replace_variable_in_file(file_path: str, variable: str, replacement: str) -> str:
        """Replace a variable in a file with a dynamic value"""
        try:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                raise FileNotFoundError(f"File not found: {file_path}")
                
            with open(file_path, 'r') as file:
                content = file.read()
            
            # Replace the variable with the provided value
            replaced_content = content.replace(f"{{{variable}}}", replacement)
            
            return replaced_content
            
        except Exception as e:
            logger.error(f"Error replacing variable in file: {e}")
            raise
    
    @staticmethod
    def process_chat_completion(session_id: str, prompt: str) -> str:
        """Process a chat completion request (placeholder for actual implementation)"""
        # In a real implementation, this would call an LLM API
        logger.info(f"Processing chat completion for session {session_id}")
        return f"Response to: {prompt[:50]}..."
    
    @staticmethod
    def remove_newline_characters(text: str) -> str:
        """Remove newline characters from text"""
        return re.sub(r'[\r\n]+', ' ', text)
    
    @staticmethod
    def format_json_response(data: Dict[str, Any]) -> str:
        """Format a dictionary as a pretty JSON string"""
        return json.dumps(data, indent=2)
    