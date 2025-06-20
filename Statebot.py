import logging
import datetime
from typing import Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

from langgraph.graph import StateGraph, END

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

    def __init__(self):
        """
        Initializes the orchestrator by building and compiling the workflow graph.
        """
        self.workflow = self._build_workflow()

    def _validate_request(self, state: WorkflowState) -> WorkflowState:
        """
        Validates the incoming request for required fields.
        Populates the 'errors' list in the state if validation fails.
        """
        logger.info(f"Validating request for conversation {state.conversation_id}")
        if not state.channel_id or not state.conversation_id:
            state.errors.append("Missing required fields: channel_id or conversation_id")
        return state

    def _preprocess_complaint(self, state: WorkflowState) -> WorkflowState:
        """
        Performs preprocessing specific to complaint capture requests.
        Enriches metadata and can transform data elements.
        """
        logger.info(f"Preprocessing complaint for conversation {state.conversation_id}")
        
        # This node should only be reached if there are no validation errors
        if not state.errors:
            state.metadata.update({
                "processed_timestamp": str(datetime.datetime.now()),
                "request_type": state.request_type
            })

            # Example of transforming data_elements from list to dict if necessary
            if isinstance(state.data_elements, list):
                data_dict = {}
                for item in state.data_elements:
                    if isinstance(item, dict) and 'name' in item and 'value' in item:
                        data_dict[item['name']] = item['value']
                state.data_elements = data_dict
                
        return state

    def _handle_error(self, state: WorkflowState) -> WorkflowState:
        """
        Handles states that have accumulated errors during the workflow.
        """
        logger.error(f"Error in workflow for conversation {state.conversation_id}: {state.errors}")
        # Error handling logic can be expanded here (e.g., notifying a monitoring system)
        return state

    def _handle_general(self, state: WorkflowState) -> WorkflowState:
        """
        Handles general, non-complaint-related requests.
        """
        logger.info(f"Handling general request for conversation {state.conversation_id}")
        # Logic for general conversation can be added here
        return state

    def _route_request(self, state: WorkflowState) -> str:
        """
        Determines the next step in the workflow based on the current state.
        This is the main conditional branching logic for the graph.
        """
        if state.errors:
            return "error_handler"
        if state.request_type == RequestType.COMPLAINT_CAPTURE.value:
            return "process_complaint"
        return "general_handler"

    def _build_workflow(self):
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

        # Add conditional edges based on the output of the router function
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
            initial_state = WorkflowState(
                request_type=request_data.get("requestType", RequestType.GENERAL.value),
                channel_id=request_data.get("channelID"),
                conversation_id=request_data.get("conversationID"),
                data_elements=request_data.get("dataElements", {}),
                chat_text=request_data.get("chatText", ""),
                action=request_data.get("action", "")
            )

            # Invoke the workflow with the initial state
            final_state = self.workflow.invoke(initial_state)

            if final_state.errors:
                return {
                    "status": "error",
                    "errors": final_state.errors
                }

            # Return the successfully processed state
            return {
                "status": "success",
                "channelID": final_state.channel_id,
                "conversationID": final_state.conversation_id,
                "dataElements": final_state.data_elements,
                "chatText": final_state.chat_text,
                "action": final_state.action,
                "metadata": final_state.metadata
            }
            
        except Exception as e:
            logger.error(f"Critical error in workflow processing: {e}", exc_info=True)
            return {
                "status": "error",
                "errors": [f"An unexpected server error occurred: {str(e)}"]
            }

