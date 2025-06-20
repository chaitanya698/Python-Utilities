import logging
import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

# --- Setup Logger ---
logger = logging.getLogger(__name__)

# --- Enums and State Definition ---

class RequestType(Enum):
    """
    Enumeration for the types of requests the orchestrator can handle.
    """
    COMPLAINT_CAPTURE = "ComplaintCapture"
    GENERAL = "General"

@dataclass
class WorkflowState:
    """
    Dataclass representing the state of the workflow.
    This state is passed between nodes in the graph.
    """
    request_type: str
    channel_id: str
    conversation_id: str
    data_elements: Dict[str, Any]
    chat_text: str
    action: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# --- Orchestrator Class ---

class ComplaintOrchestrator:
    """
    Orchestrates the processing of user requests using a stateful graph (LangGraph).
    It validates, routes, and processes requests based on their type.
    """
    workflow: CompiledStateGraph

    def __init__(self):
        """
        Initializes the orchestrator by building and compiling the workflow graph.
        """
        self.workflow = self._build_workflow()

    def _validate_request(self, state: dict) -> dict:
        """
        Validates the incoming request for required fields.
        Inside graph nodes, state is a dict.
        """
        logger.info(f"Validating request for conversation {state['conversation_id']}")
        if not state.get('channel_id') or not state.get('conversation_id'):
            state['errors'].append("Missing required fields: channel_id or conversation_id")
        return state

    def _preprocess_complaint(self, state: dict) -> dict:
        """
        Performs preprocessing specific to complaint capture requests.
        Inside graph nodes, state is a dict.
        """
        logger.info(f"Preprocessing complaint for conversation {state['conversation_id']}")
        
        if not state['errors']:
            state['metadata'].update({
                "processed_timestamp": str(datetime.datetime.now()),
                "request_type": state['request_type']
            })

            if isinstance(state['data_elements'], list):
                data_dict = {}
                for item in state['data_elements']:
                    if isinstance(item, dict) and 'name' in item and 'value' in item:
                        data_dict[item['name']] = item['value']
                state['data_elements'] = data_dict
                
        return state

    def _handle_error(self, state: dict) -> dict:
        """
        Handles states that have accumulated errors.
        Inside graph nodes, state is a dict.
        """
        logger.error(f"Error in workflow for conversation {state['conversation_id']}: {state['errors']}")
        return state

    def _handle_general(self, state: dict) -> dict:
        """
        Handles general, non-complaint-related requests.
        Inside graph nodes, state is a dict.
        """
        logger.info(f"Handling general request for conversation {state['conversation_id']}")
        return state

    def _route_request(self, state: dict) -> str:
        """
        Determines the next step in the workflow based on the current state.
        Inside graph nodes, state is a dict.
        """
        if state['errors']:
            return "error_handler"
        if state['request_type'] == RequestType.COMPLAINT_CAPTURE.value:
            return "process_complaint"
        return "general_handler"

    def _build_workflow(self) -> CompiledStateGraph:
        """
        Builds and compiles the workflow graph using StateGraph.
        """
        workflow = StateGraph(WorkflowState)

        # Add nodes
        workflow.add_node("validate", self._validate_request)
        workflow.add_node("process_complaint", self._preprocess_complaint)
        workflow.add_node("error_handler", self._handle_error)
        workflow.add_node("general_handler", self._handle_general)

        # Define edges
        workflow.set_entry_point("validate")

        # Add conditional edges
        workflow.add_conditional_edges(
            source="validate",
            path=self._route_request,
            path_map={
                "process_complaint": "process_complaint",
                "error_handler": "error_handler",
                "general_handler": "general_handler"
            }
        )
        
        # Set finish points for the graph
        workflow.add_edge("process_complaint", END)
        workflow.add_edge("error_handler", END)
        workflow.add_edge("general_handler", END)

        return workflow.compile()

    def process_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for processing requests. It initializes the state,
        invokes the workflow, and formats the final response.
        """
        try:
            # The initial state is a dictionary that matches the WorkflowState schema
            initial_state = {
                "request_type": request_data.get("requestType", RequestType.GENERAL.value),
                "channel_id": request_data.get("channelID"),
                "conversation_id": request_data.get("conversationID"),
                "data_elements": request_data.get("dataElements", {}),
                "chat_text": request_data.get("chatText", ""),
                "action": request_data.get("action", ""),
                "metadata": {},
                "errors": []
            }

            # Invoke the workflow with the initial state dictionary
            final_state_dict = self.workflow.invoke(initial_state)

            if final_state_dict.get('errors'):
                return {
                    "status": "error",
                    "errors": final_state_dict['errors']
                }

            # Return the successfully processed state
            return {
                "status": "success",
                "channelID": final_state_dict.get('channel_id'),
                "conversationID": final_state_dict.get('conversation_id'),
                "dataElements": final_state_dict.get('data_elements'),
                "chatText": final_state_dict.get('chat_text'),
                "action": final_state_dict.get('action'),
                "metadata": final_state_dict.get('metadata')
            }
            
        except Exception as e:
            logger.error(f"Critical error in workflow processing: {e}", exc_info=True)
            return {
                "status": "error",
                "errors": [f"An unexpected server error occurred: {str(e)}"]
            }

