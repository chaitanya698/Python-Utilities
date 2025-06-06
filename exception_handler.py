import logging
import traceback
import uuid
from flask import jsonify, request, g
from werkzeug.exceptions import HTTPException

# Import our custom exceptions and error enum
from exceptions import APIException, ComplaintException, ComplaintError

logger = logging.getLogger(__name__)

def generate_request_id():
    """Generates a unique request ID (e.g., UUID) for tracing."""
    return str(uuid.uuid4())

def register_exception_handlers(app):
    """
    Registers all global exception handlers with the Flask application.
    """

    @app.before_request
    def before_request():
        """Attach a unique request ID to Flask's global context (g)."""
        g.request_id = generate_request_id()
        logger.debug(f"Request ID generated: {g.request_id} for path: {request.path}")

    @app.errorhandler(ComplaintException)
    def handle_complaint_exception(e: ComplaintException):
        """
        Handles custom ComplaintException instances, which are explicitly raised
        within the application logic.
        """
        request_id = getattr(g, 'request_id', 'N/A')
        log_message = (
            f"[{request_id}] ComplaintException Caught: Code={e.error_code}, "
            f"HTTP={e.http_status}, Description='{e.error_enum.description}'"
        )

        if e.details:
            log_message += f", Details='{e.details}'"

        # Log the full traceback for critical server errors (5xx)
        if e.http_status >= 500 and e.original_exception:
            logger.exception(f"{log_message}\nOriginal Exception: {e.original_exception}")
        else:
            logger.warning(log_message)

        response = jsonify(e.to_json())
        response.status_code = e.http_status
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(e: HTTPException):
        """
        Handles standard Werkzeug/Flask HTTP exceptions (e.g., 404, 400, 500)
        and maps them to the appropriate ComplaintError response based on an
        exact mapping.
        """
        request_id = getattr(g, 'request_id', 'N/A')
        
        # Default to a generic server error for unmapped HTTP exceptions
        error_enum = ComplaintError.SERVICE_TEMPORARILY_DOWN

        # Exact mapping for specific HTTP codes defined in the ComplaintError enum
        if e.code == 400:
            error_enum = ComplaintError.BAD_REQUEST
        elif e.code == 422:
            # Map a generic 422 to a default; specific 422 errors should be
            # raised via ComplaintException in the application logic.
            error_enum = ComplaintError.SUBMISSION_PROCESSING_FAILED
        elif e.code == 500:
            error_enum = ComplaintError.SERVICE_TEMPORARILY_DOWN

        logger.warning(
            f"[{request_id}] HTTPException Caught: HTTP={e.code}, Mapped to Code={error_enum.code}. "
            f"Original Description: {e.description}"
        )

        # Use the generic chat message for all HTTP-based errors
        generic_chat_message = "Complaint Capture agent is temporarily unavailable"

        response_payload = {
            "chatResponseText": generic_chat_message,
            "errorResponse": {
                "code": error_enum.code,
                "desc": error_enum.description
            }
        }

        response = jsonify(response_payload)
        # The response status code is dictated by the mapped enum
        response.status_code = error_enum.http_status
        return response

    @app.errorhandler(Exception)
    def handle_generic_exception(e: Exception):
        """
        Handles ALL uncaught exceptions as a final fallback.
        This should always return a generic 500-level error.
        """
        request_id = getattr(g, 'request_id', 'N/A')
        error_enum = ComplaintError.SERVICE_TEMPORARILY_DOWN

        # Log the full traceback for all uncaught exceptions for debugging
        logger.exception(f"[{request_id}] Uncaught Exception. Mapping to {error_enum.code}. Details: {e}")

        # Use the generic chat message for all uncaught exceptions
        generic_chat_message = "Complaint Capture agent is temporarily unavailable"
        
        response_payload = {
            "chatResponseText": generic_chat_message,
            "errorResponse": {
                "code": error_enum.code,
                "desc": error_enum.description
            }
        }

        response = jsonify(response_payload)
        response.status_code = error_enum.http_status
        return response
