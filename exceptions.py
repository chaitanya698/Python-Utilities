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
        self.original_exception = original_exception

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
