# api_service.py (Playwright version)
import uuid
import json
import time
from typing import Dict, Any, Optional
from playwright.sync_api import APIRequestContext, Playwright, sync_playwright

from bdd_tests.config.settings import Settings
from bdd_tests.utils.logger_config import get_logger
from bdd_tests.utils.request_response_tracker import RequestResponseTracker


class ChatbotAPIClient:
    """Playwright-based API client for chatbot interactions."""

    def __init__(self, config: Settings, request_context: APIRequestContext,
                 tracker: Optional[RequestResponseTracker] = None):
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.API_BASE_URL.rstrip('/')
        self.timeout = config.API_TIMEOUT * 1000  # Playwright expects ms
        self.context = request_context
        self.tracker = tracker

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform an API request via Playwright with tracking."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        correlation_id = correlation_id or str(uuid.uuid4())

        req_headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "CLIENT-CORRELATION-ID": correlation_id
        }
        if headers:
            req_headers.update(headers)

        # Track request
        if self.tracker:
            self.tracker.add_request(method, url, req_headers, data, correlation_id)

        self.logger.info(f"[{correlation_id}] {method} {url}")

        start = time.time()
        response = self.context.request(
            method=method,
            url=url,
            data=json.dumps(data) if data else None,
            headers=req_headers,
            timeout=self.timeout
        )

        duration = time.time() - start
        self.logger.info(f"[{correlation_id}] Response Status: {response.status}")

        try:
            response.raise_for_status()
            resp_data = response.json()
        except Exception:
            resp_data = response.text()

        if self.tracker:
            self.tracker.add_response(
                response.status,
                dict(response.headers),
                resp_data,
                duration,
                correlation_id
            )

        return resp_data

    def initiate_chat(self, request_data: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
        correlation_id = correlation_id or f"INIT-{uuid.uuid4()}"
        if "conversationId" not in request_data:
            request_data["conversationId"] = "initial"

        self.logger.info(f"Initiating chat with correlation ID: {correlation_id}")

        return self._make_request(
            "POST",
            "/api/agentic-chat/v1",
            data=request_data,
            correlation_id=correlation_id
        )

    def send_message(
        self,
        conversation_id: str,
        chat_text: str,
        action: str = "proceed",
        headers: Optional[Dict] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        correlation_id = correlation_id or (headers.get("CLIENT-CORRELATION-ID") if headers else None) \
                         or f"MSG-{uuid.uuid4()}"

        payload = {
            "channelID": "BBVA",
            "conversationID": conversation_id,
            "requestType": "ComplaintCapture",
            "chatText": chat_text,
            "action": action
        }

        self.logger.info(f"Sending message to conversation: {conversation_id}")

        return self._make_request(
            "POST",
            "/api/agentic-chat/v1",
            data=payload,
            headers=headers,
            correlation_id=correlation_id
        )
