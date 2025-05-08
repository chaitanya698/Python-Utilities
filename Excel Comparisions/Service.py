import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from flask import jsonify, abort

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.schema import Document, HumanMessage, AIMessage
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

import langgraph
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import Dict, Any, List, Tuple, Optional

from app.chat.search_service import SearchService
from app.common.util import Util
from pydantic import BaseModel, Field

# Global variables
logger = logging.getLogger('ChatService_logger')
variable = 'CASEDESCRIPTION'

class ComplaintState(BaseModel):
    """Pydantic model for the state of the complaint graph."""
    session_id: str
    chat_text: str
    action: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    history: List[Dict[str, Any]] = Field(default_factory=list)
    response: Optional[str] = None
    extracted_fields: Optional[Dict[str, Any]] = None
    complaint_type: Optional[str] = None
    summary: Optional[str] = None
    attempts: int = 0
    max_attempts: int = 3
    enable_yes_or_no: str = "No"
    error: Optional[str] = None


class LLMChainManager:
    """Manager for all LLM chains used in the complaint processing"""
    def __init__(self, temperature=0.3):
        self.llm = OpenAI(temperature=temperature)
        self.field_extraction_chain = self._create_field_extraction_chain()
        self.summary_chain = self._create_summary_chain()
        self.complaint_type_chain = self._create_complaint_type_chain()
    
    def _create_field_extraction_chain(self):
        """Create chain for extracting fields from complaint text"""
        template = """
        Extract key fields from the following complaint:
        
        COMPLAINT: {chat_text}
        
        Return the fields in the following JSON format:
        {
            "fields": {
                "name": "customer name",
                "product": "product mentioned",
                "issue": "main issue",
                "date": "date of incident"
            }
        }
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_text"]
        )
        parser = JsonOutputParser()
        
        return prompt | self.llm | parser
    
    def _create_summary_chain(self):
        """Create chain for summarizing complaints"""
        template = """
        Summarize the following complaint concisely:
        
        COMPLAINT: {chat_text}
        
        Return the summary in the following JSON format:
        {
            "summary": "concise summary of the complaint"
        }
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_text"]
        )
        parser = JsonOutputParser()
        
        return prompt | self.llm | parser
    
    def _create_complaint_type_chain(self):
        """Create chain for determining complaint type"""
        template = """
        Determine the type of the following complaint:
        
        COMPLAINT: {chat_text}
        
        Return the complaint type in the following JSON format:
        {
            "complaint_type": "type of complaint (e.g., billing, service, quality, delivery)"
        }
        """
        prompt = PromptTemplate(
            template=template,
            input_variables=["chat_text"]
        )
        parser = JsonOutputParser()
        
        return prompt | self.llm | parser

class ChatService:
    def __init__(self, payload, session_manager):
        self.payload = payload
        self.chat_text = payload.get('chatText', '')
        self.session_manager = session_manager
        self.session_id = payload.get('sessionId', '')
        self.action = payload.get('actionType', 'process_complaint')
        self.llm_chain = LLMChainManager()
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for complaint processing workflow."""
        # Define the workflow graph
        graph = StateGraph(ComplaintState)
        
        # Add nodes to the graph
        graph.add_node("validate_input", self.validate_input)
        graph.add_node("invoke_model", self.invoke_model)
        graph.add_node("process_task", self.process_task)
        graph.add_node("handle_retry", self.handle_retry)
        graph.add_node("complete", self.complete_processing)
        
        # Add edges
        graph.set_entry_point("validate_input")
        
        # From validate_input, conditionally route based on decision
        graph.add_conditional_edges(
            "validate_input",
            self.decide_processing_path,
            {
                "invoke_model": "invoke_model",
                "process_task": "process_task",
                "complete": "complete"
            }
        )
        
        # From invoke_model, check if we need to retry
        graph.add_conditional_edges(
            "invoke_model",
            self.should_retry,
            {
                "retry": "handle_retry",
                "complete": "complete"
            }
        )
        
        # From process_task, check if we need to retry
        graph.add_conditional_edges(
            "process_task",
            self.should_retry,
            {
                "retry": "handle_retry",
                "complete": "complete"
            }
        )
        
        # From handle_retry, go back to the decision point
        graph.add_edge("handle_retry", "validate_input")
        
        # Mark the complete node as the end
        graph.add_edge("complete", END)
        
        # Compile the graph
        return graph.compile()
    
    def validate_input(self, state: ComplaintState) -> ComplaintState:
        """Validate the input parameters and business rules"""
        logger.info(f"Validating input for session {state.session_id}")
        
        try:
            # Check for required fields
            if not state.session_id or not state.chat_text:
                state.error = "Missing required fields"
                state.response = "Please provide the details of your complaint."
                return state
            
            # Validate session with session manager
            if not self.session_manager.validate_session(state.session_id):
                state.error = "Invalid session ID."
                return state
            
            # Add the current message to history
            state.history.append({"role": "user", "content": state.chat_text})
            
            return state
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            state.error = f"Validation error: {e}"
            state.response = "I'm having trouble processing your request. Please try again."
            return state
    
    def decide_processing_path(self, state: ComplaintState) -> str:
        """Decide whether to invoke a model or go directly to a task"""
        logger.info(f"Deciding processing path for session {state.session_id}, action: {state.action}")
        
        # Check if there's an error already
        if state.error:
            return "complete"
            
        # Route based on the action
        if state.action in ["extract_fields", "generate_summary", "generate_complaint_type", "process_complaint"]:
            return "invoke_model"
        elif state.action in ["submit_complaint", "initialize_chat", "clear_session", "clear_all_sessions"]:
            return "process_task"
        else:
            # Unknown action, handle as error
            state.error = f"Unknown action: {state.action}"
            state.response = "I'm not sure how to process that request. Please try again."
            return "complete"
    
    def should_retry(self, state: ComplaintState) -> str:
        """Check if we should retry or end"""
        if state.error and state.attempts < state.max_attempts:
            logger.info(f"Retrying for session {state.session_id}, attempt {state.attempts}")
            return "retry"
        elif state.error:
            # Max attempts reached, set a friendly error message
            logger.error(f"Max retry attempts reached for session {state.session_id}")
            state.response = "I'm having trouble processing your request. Please try again later."
            return "complete"
        else:
            return "complete"
    
    def invoke_model(self, state: ComplaintState) -> ComplaintState:
        """Invoke the LLM based on the action"""
        logger.info(f"Invoking model for session {state.session_id}, action: {state.action}")
        
        try:
            # Construct the history in the format expected by the LLM
            formatted_history = []
            for msg in state.history:
                if msg["role"] == "user":
                    formatted_history.append(HumanMessage(content=msg["content"]))
                elif msg["role"] == "assistant":
                    formatted_history.append(AIMessage(content=msg["content"]))
            
            # Prepare input for the LLM chain
            chain_input = {
                "chat_text": state.chat_text,
                "history": formatted_history,
                "session_id": state.session_id
            }
            
            # Invoke the appropriate chain based on the action
            if state.action == "extract_fields":
                result = self.llm_chain.field_extraction_chain.invoke(chain_input)
                state.extracted_fields = result.get("fields", {})
                state.response = "I've extracted the key details from your complaint."
                state.enable_yes_or_no = "Yes"
                
            elif state.action == "generate_summary":
                result = self.llm_chain.summary_chain.invoke(chain_input)
                state.summary = result.get("summary", "")
                state.response = f"Here's a summary of your complaint: {state.summary}"
                
            elif state.action == "generate_complaint_type" or state.action == "process_complaint":
                result = self.llm_chain.complaint_type_chain.invoke(chain_input)
                state.complaint_type = result.get("complaint_type", "general")
                state.response = f"This appears to be a {state.complaint_type} complaint. Would you like to provide more details?"
                
            else:
                raise ValueError(f"No model handler for action {state.action}")
                
            # Add the assistant response to history
            state.history.append({"role": "assistant", "content": state.response})
            
            # In a real implementation, update the session manager with new history
            if state.response:
                self.session_manager.add_message_to_session(
                    state.session_id, 
                    state.chat_text, 
                    state.response
                )
            
            return state
            
        except Exception as e:
            logger.error(f"Error invoking model: {e}")
            state.attempts += 1
            state.error = str(e)
            return state
    
    def handle_retry(self, state: ComplaintState) -> ComplaintState:
        """Handle retry logic"""
        logger.info(f"Handling retry for session {state.session_id}")
        
        # Clear the error before retry
        state.error = None
        
        # Implement different retry strategies based on the attempt number
        if state.attempts == 1:
            # First retry - just try again
            pass
        elif state.attempts == 2:
            # Second retry - try with simplified input
            state.chat_text = " ".join(state.chat_text.split()[:50])  # Use first 50 words
        
        return state
    
    def process_task(self, state: ComplaintState) -> ComplaintState:
        """Process non-model tasks based on action"""
        logger.info(f"Processing task for session {state.session_id}, action: {state.action}")
        
        try:
            if state.action == "initialize_chat":
                # Generate session id if not provided
                if not state.session_id:
                    state.session_id = Util.generate_session_id()
                
                # Create session
                self.session_manager.create_session(state.session_id)
                
                # Add welcome message
                state.response = "Welcome to Complaint Capture Assistant, please provide the details of your complaint."
                state.history.append({"role": "assistant", "content": state.response})
                
                # Update session manager
                self.session_manager.add_message_to_session(
                    state.session_id, 
                    state.chat_text, 
                    state.response
                )
                
            elif state.action == "submit_complaint":
                # Generate a unique complaint ID
                complaint_id = Util.generate_human_readable_complaint_id()
                
                # In a real implementation, you would store the complaint in a database
                # db.insert_complaint(complaint_id, state.session_id, state.extracted_fields, state.complaint_type, state.summary)
                
                state.response = f"Great! Complaint#{complaint_id} submitted with details. We'll process your complaint and get back to you."
                state.enable_yes_or_no = "No"
                
                # Add response to history
                state.history.append({"role": "assistant", "content": state.response})
                
                # Update session manager
                self.session_manager.add_message_to_session(
                    state.session_id, 
                    state.chat_text, 
                    state.response
                )
                
            elif state.action == "clear_session":
                self.session_manager.clear_session(state.session_id)
                state.response = f"Session {state.session_id} cleared successfully"
                
            elif state.action == "clear_all_sessions":
                self.session_manager.clear_sessions_data()
                state.response = "All sessions cleared successfully"
                
            return state
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            state.attempts += 1
            state.error = str(e)
            return state
    
    def complete_processing(self, state: ComplaintState) -> ComplaintState:
        """Complete the processing"""
        logger.info(f"Completing processing for session {state.session_id}")
        
        # If there was an error and we didn't set a response yet
        if state.error and not state.response:
            state.response = "I'm having trouble processing your request. Please try again later."
        
        # Final processing, like updating session history in a database
        # In a real implementation: session_manager.update_session(state.session_id, state.history)
        
        return state
    
    def initialize_chat(self):
        """Initialize a new chat session."""
        try:
            # Generate session id and create history
            self.session_id = Util.generate_session_id()
            self.session_manager.create_session(self.session_id)
            print(f"Session ID: {self.session_id}")
            
            self.session_manager.add_message_to_session(self.session_id, self.chat_text, 
                                                       "Welcome to Complaint Capture Assistant")
            
            return jsonify({
                'sessionId': self.session_id,
                'chatResponse': 'Welcome to Complaint Capture Assistant,' 
                               ' please provide the details of your complaint.',
                'enableYesOrNo': 'No'
            })
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            abort(500, 
                  description="An unexpected error occurred processing complaint determination. Please try again later.")
    
    def submit_complaint(self):
        """Submit a complaint."""
        try:
            if not self.session_manager.validate_session(self.session_id):
                abort(400, description="Invalid session ID.")
                
            print("Start submit complaint")
            # Submit the complaint
            complaint_id = Util.generate_human_readable_complaint_id()
            
            return jsonify({
                'chatResponse': f"Great! Complaint#{complaint_id} submitted with WellSfs!",
                'sessionId': self.payload['sessionId'],
                'enableYesOrNo': 'No'
            })
        except Exception as e:
            logger.error(f"An unexpected error occurred processing complaint submission: {e}")
            abort(500, description="error")
    
    def extract_fields(self):
        """Extract fields from complaint text."""
        try:
            if not self.session_manager.validate_session(self.session_id):
                abort(400, description="Invalid session ID.")
                
            print("Start file read fields extraction")
            # Extract the fields
            complaint_fields_extraction_path = './config/prompts/ComplaintFieldsExtraction.txt'
            fields_extraction_content = Util.replace_variable_in_file(
                complaint_fields_extraction_path,
                variable,
                self.chat_text + " " + self.session_manager.get_last_response_metadata_as_string(self.session_id)
            )
            
            print("End file read fields extraction")
            extracted_fields = Util.process_chat_completion(self.session_id, fields_extraction_content)
            # Shows both sessions and their message history
            self.session_manager.add_message_to_session(self.session_id, self.payload['chatText'], extracted_fields)
            
            return jsonify({
                'chatResponse': extracted_fields,
                'sessionId': self.payload['sessionId'],
                'enableYesOrNo': 'Yes'
            })
        except Exception as e:
            logger.error(f"An unexpected error occurred processing complaint field extraction: {e}")
            abort(500, description="Field extraction error")
    
    def generate_complaint_type(self):
        """Generate complaint type from text."""
        try:
            if not self.session_manager.validate_session(self.session_id):
                abort(400, description="Invalid session ID.")
                
            print("Start file read complaint type rules")
            complaintType_rules_input_file_path = './config/prompts/ComplaintTypeRules.txt'
            rules_content = Util.replace_variable_in_file(
                complaintType_rules_input_file_path,
                variable,
                self.chat_text + " " + self.session_manager.get_last_response_metadata_as_string(self.session_id)
            )
            
            print(f"Rules Content: {rules_content}")
            print("End file read Rules edit")
            
            with ThreadPoolExecutor() as executor:
                future_rules = executor.submit(Util.process_chat_completion, self.session_id, rules_content)
                rulesContentResponse = future_rules.result()
                
            self.session_manager.add_message_to_session(self.session_id, self.payload['chatText'], rulesContentResponse)
            
            return jsonify({
                'chatResponse': rulesContentResponse,
                'sessionId': self.payload['sessionId'],
                'enableYesOrNo': 'No'
            })
        except FileNotFoundError:
            print(f"ComplaintTypeRules File not found: {complaintType_rules_input_file_path}")
            abort(500, description="An unexpected error occurred processing complaint determination. Please try again later.")
        except Exception as e:
            logger.error(f"An unexpected error occurred processing complaint determination: {e}")
            abort(500, description="An unexpected error occurred processing complaint determination. Please try again later.")
    
    def generate_complaint_type_edit(self):
        """Generate edited complaint type."""
        try:
            if not self.session_manager.validate_session(self.session_id):
                abort(400, description="Invalid session ID.")
                
            print("Start file read Rules edit")
            complaintType_rules_input_file_path = './config/prompts/ComplaintTypeRules.txt'
            rules_content = Util.replace_variable_in_file(
                complaintType_rules_input_file_path,
                variable,
                self.chat_text + " " + self.session_manager.get_last_response_metadata_as_string(self.session_id)
            )
            
            print(f"Rules Content: {rules_content}")
            print("End file read Rules edit")
            
            with ThreadPoolExecutor() as executor:
                future_rules = executor.submit(Util.process_chat_completion, self.session_id, rules_content)
                rulesContentResponse = future_rules.result()
                
            self.session_manager.add_message_to_session(self.session_id, self.payload['chatText'], rulesContentResponse)
            
            return jsonify({
                'chatResponse': f"This appears to be a {Util.remove_newline_characters(rulesContentResponse)}",
                'sessionId': self.payload['sessionId'],
                'enableYesOrNo': 'No'
            })
        except Exception as e:
            logger.error(f"An unexpected error occurred processing complaint determination: {e}")
            abort(500, description="An unexpected error occurred processing complaint determination. Please try again later.")
    
    def generate_summary(self):
        """Generate summary of complaint."""
        try:
            if not self.session_manager.validate_session(self.session_id):
                abort(400, description="Invalid session ID.")
                
            print("Start file read Summarization")
            complaint_summarization_input_file_path = './config/prompts/ComplaintSummarization.txt'
            summary_content = Util.replace_variable_in_file(
                complaint_summarization_input_file_path,
                variable,
                self.chat_text + " " + self.session_manager.get_last_response_metadata_as_string(self.session_id)
            )
            
            print("End file read Summarization")
            
            with ThreadPoolExecutor() as executor:
                future_summary = executor.submit(Util.process_chat_completion, self.session_id, summary_content)
                summaryResponse = future_summary.result()
                
            self.session_manager.add_message_to_session(self.session_id, self.payload['chatText'], summaryResponse)
            
            return jsonify({
                'chatResponse': summaryResponse,
                'sessionId': self.payload['sessionId'],
                'enableYesOrNo': 'No'
            })
        except Exception as e:
            logger.error(f"An unexpected error occurred processing complaint summary: {e}")
            abort(500, description="An unexpected error occurred processing complaint summary. Please try again later.")
    
    def clear_session(self, session_id: str):
        """Clear a specific session."""
        try:
            if session_id in self.session_manager.session_data:
                del self.session_manager.session_data[session_id]
                
            if session_id in self.session_manager.last_message:
                del self.session_manager.last_message[session_id]
                
            if session_id in self.session_manager.last_response_metadata:
                del self.session_manager.last_response_metadata[session_id]
                
            return f"Session {session_id} cleared successfully"
        except Exception as e:
            logger.error(f"Error clearing session {session_id}: {e}")
            raise
    
    def clear_sessions_data(self):
        """Clear all session data."""
        try:
            self.session_manager.session_data.clear()
            self.session_manager.last_message.clear()
            self.session_manager.last_response_metadata.clear()
            
            return "All sessions cleared successfully"
        except Exception as e:
            logger.error(f"Error clearing all sessions: {e}")
            raise
    
    def update_last_human_message(self, session_id: str, new_message: str, new_response: str):
        """Update the last human message in the session."""
        if session_id in self.session_manager.last_message:
            self.session_manager.last_message[session_id].content = new_message
            self.session_manager.last_response_metadata[session_id] = {'response': new_response}
    
    def get_last_response_metadata(self, session_id: str):
        """Get last response metadata for a session."""
        return self.session_manager.last_response_metadata.get(session_id)
    
    def get_last_message(self, session_id: str):
        """Get last message for a session."""
        return self.session_manager.last_message[session_id].content
    
    def add_response_to_session(self, session_id: str, message: str):
        """Add a response to the session history."""
        history = self.session_manager.get_session_history(session_id)
        history.add_ai_message(message)
        
    def execute_workflow(self) -> Dict[str, Any]:
        """Execute the full LangGraph workflow."""
        initial_state = ComplaintState(
            session_id=self.session_id,
            chat_text=self.chat_text,
            action=self.action,
            payload=self.payload
        )
        
        # Execute the graph
        final_state = self.graph.invoke(initial_state)
        
        # Format the response for the API
        return {
            "sessionId": final_state.session_id,
            "chatResponse": final_state.response or "I couldn't process your request.",
            "enableYesOrNo": final_state.enable_yes_or_no,
            "extractedFields": final_state.extracted_fields,
            "complaintType": final_state.complaint_type,
            "summary": final_state.summary,
            "error": final_state.error
        }