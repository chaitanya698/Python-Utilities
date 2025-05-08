from typing import Dict, Any, List, Tuple, Optional
import logging
from langchain.schema import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

# Configure logging
logger = logging.getLogger("complaint_graph")

# Define the state schema
class ComplaintState(BaseModel):
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
    def __init__(self, llm):
        self.llm = llm
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

def create_complaint_graph(llm_chain, session_manager):
    """Create the LangGraph for complaint processing"""
    # Initialize graph
    graph = StateGraph(ComplaintState)
    
    # Define nodes
    
    # 1. Validate Input node
    def validate_input(state: ComplaintState) -> ComplaintState:
        """Validate the input and check for required fields"""
        logger.info(f"Validating input for session {state.session_id}")
        
        try:
            # Check for required fields
            if not state.session_id or not state.chat_text:
                state.error = "Missing required fields"
                state.response = "Please provide the details of your complaint."
                return state
            
            # Validate session with session manager
            if not session_manager.validate_session(state.session_id):
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
    
    # 2. Complaint Capture Decision node
    def decide_processing_path(state: ComplaintState) -> str:
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
    
    # 3. Invoke Model node
    def invoke_model(state: ComplaintState) -> ComplaintState:
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
                result = llm_chain.field_extraction_chain.invoke(chain_input)
                state.extracted_fields = result.get("fields", {})
                state.response = "I've extracted the key details from your complaint."
                state.enable_yes_or_no = "Yes"
                
            elif state.action == "generate_summary":
                result = llm_chain.summary_chain.invoke(chain_input)
                state.summary = result.get("summary", "")
                state.response = f"Here's a summary of your complaint: {state.summary}"
                
            elif state.action == "generate_complaint_type" or state.action == "process_complaint":
                result = llm_chain.complaint_type_chain.invoke(chain_input)
                state.complaint_type = result.get("complaint_type", "general")
                state.response = f"This appears to be a {state.complaint_type} complaint. Would you like to provide more details?"
                
            else:
                raise ValueError(f"No model handler for action {state.action}")
                
            # Add the assistant response to history
            state.history.append({"role": "assistant", "content": state.response})
            
            # In a real implementation, update the session manager with new history
            if state.response:
                session_manager.add_message_to_session(
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
    
    # 4. Process Task node
    def process_task(state: ComplaintState) -> ComplaintState:
        """Process non-model tasks based on action"""
        logger.info(f"Processing task for session {state.session_id}, action: {state.action}")
        
        try:
            if state.action == "initialize_chat":
                # Generate session id if not provided
                if not state.session_id:
                    import uuid
                    state.session_id = str(uuid.uuid4())
                
                # Create session
                session_manager.create_session(state.session_id)
                
                # Add welcome message
                state.response = "Welcome to Complaint Capture Assistant, please provide the details of your complaint."
                state.history.append({"role": "assistant", "content": state.response})
                
                # Update session manager
                session_manager.add_message_to_session(
                    state.session_id, 
                    state.chat_text, 
                    state.response
                )
                
            elif state.action == "submit_complaint":
                # Generate a unique complaint ID
                import random
                import string
                prefix = ''.join(random.choices(string.ascii_uppercase, k=2))
                number = random.randint(1000, 9999)
                complaint_id = f"{prefix}{number}"
                
                # In a real implementation, you would store the complaint in a database
                # db.insert_complaint(complaint_id, state.session_id, state.extracted_fields, state.complaint_type, state.summary)
                
                state.response = f"Great! Complaint#{complaint_id} submitted with details. We'll process your complaint and get back to you."
                state.enable_yes_or_no = "No"
                
                # Add response to history
                state.history.append({"role": "assistant", "content": state.response})
                
                # Update session manager
                session_manager.add_message_to_session(
                    state.session_id, 
                    state.chat_text, 
                    state.response
                )
                
            elif state.action == "clear_session":
                session_manager.clear_session(state.session_id)
                state.response = f"Session {state.session_id} cleared successfully"
                
            elif state.action == "clear_all_sessions":
                session_manager.clear_sessions_data()
                state.response = "All sessions cleared successfully"
                
            return state
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            state.attempts += 1
            state.error = str(e)
            return state
    
    # 5. Check for Retry node
    def should_retry(state: ComplaintState) -> str:
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
    
    # 6. Retry Logic node
    def handle_retry(state: ComplaintState) -> ComplaintState:
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
    
    # 7. Complete node
    def complete_processing(state: ComplaintState) -> ComplaintState:
        """Complete the processing"""
        logger.info(f"Completing processing for session {state.session_id}")
        
        # If there was an error and we didn't set a response yet
        if state.error and not state.response:
            state.response = "I'm having trouble processing your request. Please try again later."
        
        # Final processing, like updating session history in a database
        # In a real implementation: session_manager.update_session(state.session_id, state.history)
        
        return state
    
    # Add nodes to the graph
    graph.add_node("validate_input", validate_input)
    graph.add_node("invoke_model", invoke_model)
    graph.add_node("process_task", process_task)
    graph.add_node("handle_retry", handle_retry)
    graph.add_node("complete", complete_processing)
    
    # Add edges
    graph.set_entry_point("validate_input")
    
    # From validate_input, conditionally route based on decision
    graph.add_conditional_edges(
        "validate_input",
        decide_processing_path,
        {
            "invoke_model": "invoke_model",
            "process_task": "process_task",
            "complete": "complete"
        }
    )
    
    # From invoke_model, check if we need to retry
    graph.add_conditional_edges(
        "invoke_model",
        should_retry,
        {
            "retry": "handle_retry",
            "complete": "complete"
        }
    )
    
    # From process_task, check if we need to retry
    graph.add_conditional_edges(
        "process_task",
        should_retry,
        {
            "retry": "handle_retry",
            "complete": "complete"
        }
    )
    
    # From handle_retry, go back to the decision point
    graph.add_edge("handle_retry", "validate_input")
    
    # Mark the complete node as the end
    graph.add_edge("complete", END)
    
    return graph.compile()