# app/exception_handlers/exception_handler.py
import logging
import traceback
import sys
from flask import jsonify, request, g # Import g
from werkzeug.exceptions import HTTPException, default_exceptions

# Import our custom exceptions and error enum
from exceptions import APIException, ComplaintException, ComplaintError

# Configure logging (Best practice: configure this in your app's __init__.py or config)
logger = logging.getLogger(__name__) # Use the module name as logger name

def generate_request_id():
    """Generates a unique request ID (e.g., UUID) for tracing."""
    import uuid
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


    @app.errorhandler(ComplaintException) # Catch your specific ComplaintException first
    def handle_complaint_exception(e: ComplaintException):
        """
        Handles custom ComplaintException instances.
        These are errors explicitly raised within your chatbot workflow logic.
        """
        request_id = getattr(g, 'request_id', 'N/A') # Access request_id from g
        log_message = (
            f"[{request_id}] ComplaintException Caught: Code={e.error_code}, "
            f"HTTP={e.http_status}, UserMessage='{e.error_enum.default_message}', "
            f"DevMessage='{e.error_enum.developer_message}'"
        )

        if e.details:
            log_message += f", Details='{e.details}'"
        if e.original_exception:
            # Log the full traceback of the original exception for critical server errors
            if e.http_status >= 500: # Log full traceback for server errors
                logger.exception(f"{log_message}\nOriginal Exception: {e.original_exception}")
            else: # For client errors (e.g., 400), just log the original exception message/details
                logger.warning(f"{log_message}\nOriginal Exception Message: {e.original_exception}")
        else:
            logger.warning(log_message)

        response = jsonify(e.to_json(request_id))
        response.status_code = e.http_status
        return response

    @app.errorhandler(HTTPException)
    def handle_http_exception(e: HTTPException):
        """
        Handles standard Werkzeug/Flask HTTP exceptions (e.g., 404, 400, 500).
        These are often raised by Flask itself or extensions.
        """
        request_id = getattr(g, 'request_id', 'N/A') # Access request_id from g
        error_code = f"HTTP{e.code}" if e.code else "HTTP500"
        developer_message = e.description if e.description else "No specific HTTP error description."
        user_message = "An API error occurred. Please try again." # Generic for HTTP errors by default

        # Use GENERIC_API_ERROR for all unmapped HTTP exceptions.
        # The provided ComplaintError enum does not contain specific mappings for
        # 400, 401, 403, 404, 429.
        if e.code in [400, 401, 403, 404, 429, 500]:
            user_message = ComplaintError.GENERIC_API_ERROR.default_message
            error_code = ComplaintError.GENERIC_API_ERROR.code
        
        logger.warning(
            f"[{request_id}] HTTPException Caught: Code={error_code}, "
            f"HTTP={e.code}, UserMessage='{user_message}', "
            f"DevMessage='{developer_message}'"
        )

        response_payload = {
            "status": "error",
            "code": error_code,
            "message": user_message,
            "request_id": request_id
        }
        response = jsonify(response_payload)
        response.status_code = e.code if e.code else 500
        return response

    @app.errorhandler(Exception)
    def handle_generic_exception(e: Exception):
        """
        Handles ALL uncaught exceptions (the ultimate fallback).
        This catches anything not specifically handled by other error handlers.
        """
        request_id = getattr(g, 'request_id', 'N/A') # Access request_id from g
        error_enum = ComplaintError.GENERIC_API_ERROR

        # Log the full traceback for all uncaught exceptions
        logger.exception(
            f"[{request_id}] Uncaught Exception: "
            f"UserMessage='{error_enum.default_message}', "
            f"DevMessage='{e}'" # Use 'e' to capture the exception message
        )

        response_payload = {
            "status": "error",
            "code": error_enum.code,
            "message": error_enum.default_message, # Always show generic message to user
            "request_id": request_id
        }
        response = jsonify(response_payload)
        response.status_code = error_enum.http_status # Always 500 for generic server error
        return response