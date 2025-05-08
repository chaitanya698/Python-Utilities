import logging
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from flask import jsonify, abort

# LangChain imports
from langchain.prompts import ChatPromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from typing import TypedDict, List, Dict, Any, Literal

# Local imports
from app.chat.search_service import SearchService
from app.common.util import Util

# Load environment variables
load_dotenv()

# Global variables
logger = logging.getLogger('ChatService_logger')
variable = 'CASEDESCRIPTION'

class SessionState(TypedDict):
    """State for the session graph."""
    session_id: str
    chat_text: str
    payload: Dict[str, Any]
    session_manager: Any
    status: Literal["in_progress", "complete", "error"]
    error: Dict[str, Any]
    response: Dict[str, Any]

class ChatService:
    def __init__(self, payload, session_manager):
        """Initialize the ChatService with payload and session manager."""
        self.payload = payload
        self.chat_text = payload.get('chatText', '')
        self.session_manager = session_manager
        self.session_id = payload.get('sessionId')

        # Initialize LangChain components
        self.llm = ChatOpenAI(temperature=0)
        
        # Initialize workflow graph
        self._init_workflow_graph()

    def _init_workflow_graph(self):
        """Initialize the LangGraph workflow."""
        # Define the graph
        self.workflow = StateGraph(SessionState)
        
        # Add nodes
        self.workflow.add_node("validate_input", self._validate_input)
        self.workflow.add_node("route_request", self._route_request)
        self.workflow.add_node("invoke_model", self._invoke_model)
        self.workflow.add_node("process_task", self._process_task)
        self.workflow.add_node("complete", self._complete_session)
        
        # Add edges
        self.workflow.add_edge("validate_input", "route_request")
        self.workflow.add_conditional_edges(
            "route_request",
            self._route_condition,
            {
                "model": "invoke_model",
                "task": "process_task",
                "error": END
            }
        )
        self.workflow.add_edge("invoke_model", "process_task")
        self.workflow.add_edge("process_task", "complete")
        self.workflow.add_edge("complete", END)
        
        # Set entry point
        self.workflow.set_entry_point("validate_input")
        
        # Compile the graph
        self.compiled_workflow = self.workflow.compile()

    def _validate_input(self, state: SessionState) -> SessionState:
        """Validate input schema and business rules."""
        try:
            # Basic validation
            if not state.get("session_id") or not state.get("chat_text"):
                return {
                    **state, 
                    "status": "error", 
                    "error": {"code": 400, "description": "Missing required fields"}
                }
            return {**state, "status": "in_progress"}
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return {
                **state, 
                "status": "error", 
                "error": {"code": 500, "description": str(e)}
            }

    def _route_condition(self, state: SessionState) -> str:
        """Determine the next node based on state."""
        if state.get("status") == "error":
            return "error"
        
        # Check request type to determine if model invocation is needed
        request_type = state.get("payload", {}).get("requestType", "")
        if request_type in ["complaint_type", "summary", "field_extraction"]:
            return "model"
        return "task"

    def _invoke_model(self, state: SessionState) -> SessionState:
        """Invoke LangChain model processing."""
        try:
            request_type = state.get("payload", {}).get("requestType", "")
            chat_text = state.get("chat_text", "")
            
            # Create appropriate chain based on request type
            if request_type == "complaint_type":
                # Get rules for processing
                rules_path = './config/prompts/ComplaintTypeRules.txt'
                rules_content = Util.replace_variable_in_file(
                    rules_path, 
                    variable, 
                    chat_text + " " + self.session_manager.get_last_response(state["session_id"])
                )
                
                prompt = ChatPromptTemplate.from_template(rules_content)
                chain = LLMChain(llm=self.llm, prompt=prompt)
                result = chain.run(chat_text=chat_text)
                
            elif request_type == "summary":
                # Get summarization prompt
                summary_path = './config/prompts/ComplaintSummarization.txt'
                summary_content = Util.replace_variable_in_file(
                    summary_path, 
                    variable, 
                    chat_text + " " + self.session_manager.get_last_response(state["session_id"])
                )
                
                prompt = ChatPromptTemplate.from_template(summary_content)
                chain = LLMChain(llm=self.llm, prompt=prompt)
                result = chain.run(chat_text=chat_text)
                
            elif request_type == "field_extraction":
                # Get field extraction prompt
                fields_path = './config/prompts/ComplaintFieldsExtraction.txt'
                fields_content = Util.replace_variable_in_file(
                    fields_path, 
                    variable, 
                    chat_text + " " + self.session_manager.get_last_response(state["session_id"])
                )
                
                prompt = ChatPromptTemplate.from_template(fields_content)
                chain = LLMChain(llm=self.llm, prompt=prompt)
                result = chain.run(chat_text=chat_text)
                
            else:
                result = "Unknown request type"
                
            return {
                **state, 
                "response": {"content": result}, 
                "status": "in_progress"
            }
            
        except FileNotFoundError as e:
            logger.error(f"File not found: {e}")
            return {
                **state, 
                "status": "error", 
                "error": {"code": 500, "description": f"Configuration file not found: {e}"}
            }
        except Exception as e:
            logger.error(f"Model invocation error: {e}")
            return {
                **state, 
                "status": "error", 
                "error": {"code": 500, "description": f"An unexpected error occurred: {e}"}
            }

    def _process_task(self, state: SessionState) -> SessionState:
        """Process task-based workflows."""
        try:
            request_type = state.get("payload", {}).get("requestType", "")
            
            # If we already have a response from model invocation, just pass it through
            if state.get("response"):
                # Add the response to session history
                self.session_manager.add_response_to_session(
                    state["session_id"], 
                    state["chat_text"], 
                    state["response"]["content"]
                )
                return state
            
            # Handle direct tasks that don't need model invocation
            if request_type == "submit_complaint":
                complaint_id = Util.generate_human_readable_complaint_id()
                response = f"Great! Complaint#{complaint_id} submitted with WellStar."
                
                self.session_manager.add_response_to_session(
                    state["session_id"], 
                    state["chat_text"], 
                    response
                )
                
                return {
                    **state, 
                    "response": {"content": response}, 
                    "status": "in_progress"
                }
                
            # Default response for unknown tasks
            return {
                **state, 
                "response": {"content": "I'm not sure how to process that request."}, 
                "status": "in_progress"
            }
            
        except Exception as e:
            logger.error(f"Task processing error: {e}")
            return {
                **state, 
                "status": "error", 
                "error": {"code": 500, "description": f"An unexpected error occurred: {e}"}
            }

    def _complete_session(self, state: SessionState) -> SessionState:
        """Mark session as complete and finalize."""
        if state.get("status") == "error":
            # If there's an error, abort with appropriate status
            abort(
                state.get("error", {}).get("code", 500), 
                description=state.get("error", {}).get("description", "An unknown error occurred")
            )
            
        # Finalize the response
        return {
            **state, 
            "status": "complete"
        }

    def initialize_chat(self):
        """Initialize a chat session and return welcome message."""
        try:
            # Generate session id and create history
            self.session_id = Util.generate_session_id()
            self.session_manager.create_session(self.session_id)
            
            # Add welcome message
            welcome_msg = 'Welcome to Complaint Capture Assistant'
            self.session_manager.add_message_to_session(
                self.session_id, 
                self.chat_text, 
                welcome_msg
            )
            
            return jsonify({
                'sessionId': self.session_id,
                'chatResponse': 'Welcome to Complaint Capture Assistant, please provide the details of your complaint.',
                'enableYesOrNo': 'No'
            })
            
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            abort(500, description="An unexpected error occurred processing complaint determination. Please try again later.")

    def process_request(self):
        """Process the user request using the LangGraph workflow."""
        # Initialize state
        initial_state = {
            "session_id": self.session_id,
            "chat_text": self.chat_text,
            "payload": self.payload,
            "session_manager": self.session_manager,
            "status": "in_progress",
            "error": {},
            "response": {}
        }
        
        # Execute workflow
        try:
            result = self.compiled_workflow.invoke(initial_state)
            
            # Format response for jsonify
            return jsonify({
                'chatResponse': result["response"].get("content", ""),
                'sessionId': self.session_id,
                'enableYesOrNo': 'No'  # Default, can be customized based on response
            })
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            abort(500, description="An unexpected error occurred. Please try again later.")

    # Delegate methods for API endpoints
    def generate_complaint_type(self):
        self.payload["requestType"] = "complaint_type"
        return self.process_request()
        
    def generate_summary(self):
        self.payload["requestType"] = "summary"
        return self.process_request()
        
    def extract_fields(self):
        self.payload["requestType"] = "field_extraction"
        return self.process_request()
        
    def submit_complaint(self):
        self.payload["requestType"] = "submit_complaint"
        return self.process_request()