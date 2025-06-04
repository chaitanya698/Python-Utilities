from enum import Enum, auto

class ComplaintError(Enum):
  
    GENERIC_API_ERROR = ("CHAT0001", 500, "An unexpected error occurred. Please try again later.", "An unexpected error occurred on the server.")
    MODEL_UNAVAILABLE = ("C-101", 442, "AI model processing is not available.", "The AI model service is currently unreachable or down.")
    CUSTOMER_SEARCH_UNAVAILABLE = ("C-102", 442, "Customer search model processing is not available.", "The customer search service is not responding.")
    HR_DATA_UNAVAILABLE = ("C-103", 442, "HR data processing is not available.", "The HR data processing service encountered an issue.")
    SUBMISSION_FAILED = ("C-104", 442, "Complaint submission process failed/unavailable.", "Failed to submit the complaint due to an internal processing error.")
    SERVICE_TEMPORARILY_DOWN = ("COMPL1005", 500, "Failed to submit the complaint. Please try again.", "Complaint submission process failed.")
    # Add more as you identify specific error/exception conditions within our flow

    def __init__(self, code, http_status, default_message, developer_message):
        self.code = code
        self.http_status = http_status
        self.default_message = default_message
        self.developer_message = developer_message

class APIException(Exception):
    """
    Base exception class for all custom API-related errors.
    Inherits from Python's built-in Exception.
    """
    def __init__(self, error_enum: ComplaintError, details: str = None, original_exception: Exception = None):
        super().__init__(error_enum.developer_message)
        self.error_enum = error_enum
        self.details = details
        self.original_exception = original_exception # Store the original exception for logging

    @property
    def http_status(self) -> int:
        return self.error_enum.http_status

    @property
    def error_code(self) -> str:
        return self.error_enum.code

    def to_json(self, request_id: str = None) -> dict:
        """
        Generates a standardized JSON response for the chatbot client.
        """
        response_payload = {
            "status": "error",
            "code": self.error_code,
            "message": self.error_enum.default_message,
        }
        if request_id:
            response_payload["request_id"] = request_id # Important for tracing

        return response_payload

# Specific custom exceptions for clarity in code, inheriting from APIException
class ComplaintException(APIException): # This aligns with your current usage
    def __init__(self, error_enum: ComplaintError, details: str = None, original_exception: Exception = None):
        if not isinstance(error_enum, ComplaintError):
            raise TypeError("ComplaintException must be initialized with a ComplaintError enum member.")
        super().__init__(error_enum, details, original_exception)
