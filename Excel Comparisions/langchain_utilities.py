import os
import uuid
import re
import logging
import random
import string
from typing import Dict, Any, Optional

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

logger = logging.getLogger('Util_logger')

class Util:
    @staticmethod
    def generate_session_id() -> str:
        """Generate a unique session ID."""
        return str(uuid.uuid4())
    
    @staticmethod
    def generate_human_readable_complaint_id() -> str:
        """Generate a human-readable complaint ID."""
        prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
        number = random.randint(1000, 9999)
        return f"{prefix}{number}"
    
    @staticmethod
    def replace_variable_in_file(file_path: str, variable_name: str, replacement: str) -> str:
        """
        Read a file and replace a variable placeholder with the specified replacement.
        
        Args:
            file_path: Path to the file to read
            variable_name: Name of the variable to replace
            replacement: Text to substitute for the variable
            
        Returns:
            The file content with the variable replaced
        """
        try:
            with open(file_path, 'r') as file:
                content = file.read()
                
            # Replace the variable placeholder with the replacement text
            updated_content = content.replace(f"{{{variable_name}}}", replacement)
            
            return updated_content
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error replacing variable in file: {e}")
            raise
    
    @staticmethod
    def process_chat_completion(session_id: str, prompt_content: str) -> str:
        """
        Process a chat completion request using LangChain.
        
        Args:
            session_id: The session ID
            prompt_content: The prompt content
            
        Returns:
            The response from the language model
        """
        try:
            # Initialize OpenAI LLM
            llm = OpenAI(temperature=0.3)
            
            # Create prompt template
            prompt_template = PromptTemplate(
                template=prompt_content,
                input_variables=[]
            )
            
            # Create LLMChain
            chain = LLMChain(
                llm=llm,
                prompt=prompt_template,
                verbose=True
            )
            
            # Run the chain
            response = chain.run({})
            
            return response
        except Exception as e:
            logger.error(f"Error processing chat completion: {e}")
            raise
    
    @staticmethod
    def remove_newline_characters(text: str) -> str:
        """Remove newline characters from text."""
        if not text:
            return ""
        return re.sub(r'[\r\n]+', ' ', text).strip()
    
    @staticmethod
    def extract_json_from_text(text: str) -> Dict[str, Any]:
        """Extract JSON from text."""
        try:
            import json
            # Find JSON-like pattern in text
            json_pattern = r'\{.*\}'
            match = re.search(json_pattern, text, re.DOTALL)
            
            if match:
                json_str = match.group(0)
                return json.loads(json_str)
            else:
                logger.warning("No JSON found in text")
                return {}
        except Exception as e:
            logger.error(f"Error extracting JSON from text: {e}")
            return {}