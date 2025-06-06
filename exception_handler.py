import logging
import traceback
from flask import jsonify, request
from werkzeug.exceptions import HTTPException

# Import our custom exceptions and error enum
from exceptions import APIException, ComplaintException, ComplaintError

logger = logging.getLogger(__name__)

def get_conversation_id():
    """
    Safely retrieves the conversationID from the incoming request's JSON payload.
    """
    if request.is_json and request.json:
        # Check for 'conversationID' (from spec) or 'conversation_id' as a fallback.
        return request.json.get('conversationID') or request.json.get('conversation_id')
    return 'N/A' # Default if not found or if the request is not JSON.

def register_exception_handlers(app):
    """
    Registers all global exception handlers with the Flask application.
    """

    @app.errorhandler(ComplaintException)
    def handle_complaint_exception(e: ComplaintException):
        """
        Handles custom ComplaintException instances, which are explicitly raised
        within the application logic.
        """
        conversation_id = get_conversation_id()
        log_message = (
            f"[{conversation_id}] ComplaintException Caught: Code={e.error_code}, "
            f"HTTP={e.http_status}, Description='{e.error_enum.description}'"
        )

        if e.details:
            log_message += f", Details='{e.details}'"

        if e.http_status >= 500 and e.original_exception:
            logger.exception(f"{log_message}\nOriginal Exception: {e.original_exception}")
        else:
            logger.warning(log_message)

        response = jsonify(e.to_json(conversation_id=conversation_id))
        response.status_code = e.http_status
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(e: HTTPException):
        """
        Handles standard Werkzeug/Flask HTTP exceptions (e.g., 404, 400, 500)
        and maps them to the appropriate ComplaintError response.
        """
        conversation_id = get_conversation_id()
        
        if e.code == 400:
            error_enum = ComplaintError.BAD_REQUEST
        else:
            error_enum = ComplaintError.SERVICE_TEMPORARILY_DOWN

        logger.warning(
            f"[{conversation_id}] HTTPException Caught: HTTP={e.code}, Mapped to Code={error_enum.code}. "
            f"Original Description: {e.description}"
        )

        generic_chat_message = "Complaint Capture agent is temporarily unavailable"

        response_payload = {
            "chatResponseText": generic_chat_message,
            "errorResponse": {
                "code": error_enum.code,
                "desc": error_enum.description,
                "conversationID": conversation_id
            }
        }

        response = jsonify(response_payload)
        response.status_code = error_enum.http_status
        return response

    @app.errorhandler(Exception)
    def handle_generic_exception(e: Exception):
        """
        Handles ALL uncaught exceptions as a final fallback.
        """
        conversation_id = get_conversation_id()
        error_enum = ComplaintError.SERVICE_TEMPORARILY_DOWN

        logger.exception(f"[{conversation_id}] Uncaught Exception. Mapping to {error_enum.code}. Details: {e}")
        
        generic_chat_message = "Complaint Capture agent is temporarily unavailable"
        
        response_payload = {
            "chatResponseText": generic_chat_message,
            "errorResponse": {
                "code": error_enum.code,
                "desc": error_enum.description,
                "conversationID": conversation_id
            }
        }

        response = jsonify(response_payload)
        response.status_code = error_enum.http_status
        return response
