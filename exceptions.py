from enum import Enum

class ComplaintError(Enum):
    """
    Defines standardized error codes, HTTP statuses, and descriptions
    based on the project's OpenAPI and Jira specifications.
    """
    SERVICE_TEMPORARILY_DOWN = ("EC-100", 500, "Complaint AI temporarily unavailable")
    BAD_REQUEST = ("EC-101", 400, "Bad request")
    AI_MODEL_PROCESSING_NOT_AVAILABLE = ("EC-102", 422, "AI model processing not available")
    AI_MODEL_PROCESSING_ERROR = ("EC-103", 422, "Error occurred during AI model processing")
    SUBMISSION_PROCESSING_FAILED = ("EC-104", 422, "Complaints submission processing failed/unavailable")

    def __init__(self, code, http_status, description):
        self.code = code
        self.http_status = http_status
        self.description = description

class APIException(Exception):
    """
    Base exception class for all custom API-related errors.
    Inherits from Python's built-in Exception.
    """
    def __init__(self, error_enum: ComplaintError, details: str = None, original_exception: Exception = None):
        super().__init__(error_enum.description)
        self.error_enum = error_enum
        self.details = details
        self.original_exception = original_exception # Store the original exception for logging

    @property
    def http_status(self) -> int:
        return self.error_enum.http_status

    @property
    def error_code(self) -> str:
        return self.error_enum.code

    def to_json(self, conversation_id: str = None) -> dict:
        """
        Generates a standardized JSON response for the client, conforming to the OpenAPI spec.
        Includes conversation_id in the errorResponse object.
        """
        # Per the specification, chatResponseText can be a generic message in error scenarios.
        generic_chat_message = "Complaint Capture agent is temporarily unavailable"

        response_payload = {
            "chatResponseText": generic_chat_message,
            "errorResponse": {
                "code": self.error_code,
                "desc": self.error_enum.description,
                "conversationID": conversation_id
            }
        }
        return response_payload

class ComplaintException(APIException):
    """
    Specific custom exception for chatbot workflow errors.
    Ensures that only a valid ComplaintError enum member is used.
    """
    def __init__(self, error_enum: ComplaintError, details: str = None, original_exception: Exception = None):
        if not isinstance(error_enum, ComplaintError):
            raise TypeError("ComplaintException must be initialized with a ComplaintError enum member.")
        super().__init__(error_enum, details, original_exception)



views:


# Add these imports at the top of your views.py file
from flask import request, jsonify
from jsonschema import validate, ValidationError
from exceptions import ComplaintException, ComplaintError # <-- Import custom exceptions

# Assuming 'logger', 'chat_bp', 'request_schema', 'response_schema',
# and 'process_chat' are defined elsewhere in the file.


@chat_bp.route('/agentic-chat/v2', methods=['POST'])
def complaint_capture_chat2():
    """
    Handles the chat request, now using global exception handlers for errors.
    """
    try:
        logger.info("complaint_capture_chat called")

        # request.get_json() will automatically raise a 400-level HTTPException
        # if the request body is not valid JSON. This will be caught by the
        # global handle_http_exception handler.
        request_data = request.get_json()

        # Perform schema validation.
        validate(instance=request_data, schema=request_schema)

        # Extract data from the request
        channel_id = request_data.get("channelID")
        conversation_id = request_data.get("conversationID")
        data_elements = request_data.get("dataElements")
        chat_text = request_data.get("chatText")
        action = request_data.get("action")

        response_data = process_chat(channel_id, conversation_id, data_elements, chat_text, action)

        # It's assumed process_chat might return its own error structure.
        # This logic is preserved from the original code.
        if "statusMessages" in response_data:
            return jsonify(response_data), 400

        # Validate the successful response against the response schema
        validate(instance=response_data, schema=response_schema)

        return jsonify(response_data), 200

    except ValidationError as e:
        # Catch the specific schema validation error.
        logger.error(f"Schema validation failed for request: {e.message}")
        # Raise our custom APIException, which the global handle_complaint_exception
        # handler will catch and format into the standard error response.
        raise ComplaintException(ComplaintError.BAD_REQUEST, details=e.message)

    # NOTE: The generic 'except Exception' block has been removed.
    # Any other unexpected exception will be caught by the global
    # handle_generic_exception handler, which will return a standard 500 error response.
