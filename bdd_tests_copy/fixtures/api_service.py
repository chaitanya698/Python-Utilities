import uuid
import time
from typing import Dict, Any, Optional, Callable
from urllib.parse import urlparse

from playwright.sync_api import Playwright, APIRequestContext, Response

from bdd_tests.config.settings import Settings
from bdd_tests.utils.logger_config import get_logger
from bdd_tests.utils.request_response_tracker import RequestResponseTracker


class ChatbotAPIClient:
    """Playwright-based API client for chatbot interactions."""

    def __init__(self, config: Settings, playwright: Playwright,
                 tracker: Optional[RequestResponseTracker] = None):
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.API_BASE_URL.rstrip('/')
        self.timeout = int(config.API_TIMEOUT * 1000)  # ms
        self.tracker = tracker
        self.context: Optional[APIRequestContext] = None
        self._create_session(playwright)

    # ------------------------
    # Session / Certificates
    # ------------------------
    def _create_session(self, playwright: Playwright) -> None:
        """
        Create a Playwright APIRequestContext with SSL and client certificate handling.
        IMPORTANT: The 'origin' must EXACTLY match scheme + host + port of requests using the cert.
        """
        extra_headers = {
            "User-Agent": "ChatbotAutomation/Playwright",
            # Do NOT force Content-Type; Playwright sets it for json=... automatically.
            "Accept": "application/json",
        }

        # SSL verification
        ignore_https_errors = not bool(self.config.VERIFY_SSL)

        # Compute exact origin for client certs
        parsed = urlparse(self.base_url)
        scheme = parsed.scheme or "https"
        host = parsed.hostname
        port = parsed.port or (443 if scheme == "https" else 80)
        exact_origin = f"{scheme}://{host}:{port}"

        client_certs = None
        if self.config.CERT_PEM_PATH and self.config.KEY_PEM_PATH:
            client_certs = [{
                "origin": exact_origin,
                "cert": self.config.CERT_PEM_PATH,
                "key": self.config.KEY_PEM_PATH,
            }]
            self.logger.info(
                "Configured client certificate for origin=%s | cert=%s | key=%s",
                exact_origin, self.config.CERT_PEM_PATH, self.config.KEY_PEM_PATH
            )

        self.context = playwright.request.new_context(
            base_url=self.base_url,                  # allows passing just endpoint paths
            ignore_https_errors=ignore_https_errors, # mirrors requests.verify=False
            extra_http_headers=extra_headers,
            client_certificates=client_certs
        )

        if ignore_https_errors:
            self.logger.warning("SSL verification DISABLED (ignore_https_errors=True)")
        else:
            self.logger.info("SSL verification ENABLED")

    # ------------------------
    # Core request with retry
    # ------------------------
    def _call_with_retry(
        self,
        fn: Callable[..., Response],
        *,
        url_or_path: str,
        json_body: Optional[Dict[str, Any]],
        headers: Dict[str, str],
        correlation_id: str,
        max_retries: int = 2,
        backoff_base: float = 0.4,
    ) -> Response:
        """
        Call a Playwright request function with small bounded retries on 5xx/429.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(0, max_retries + 1):
            start = time.time()
            try:
                resp = fn(
                    url_or_path,
                    json=json_body if json_body is not None else None,
                    headers=headers,
                    timeout=self.timeout,
                    verify_ssl=False
                )
                duration = time.time() - start
                self.logger.info(
                    "[%s] HTTP %s (%d ms) -> %s",
                    correlation_id, resp.status, int(duration * 1000), url_or_path
                )

                # Retry on transient statuses
                if resp.status in (429, 500, 502, 503, 504):
                    if attempt < max_retries:
                        body_preview = resp.text()[:300]
                        self.logger.warning(
                            "[%s] Transient status %s, retrying... (attempt %d/%d) Body: %s",
                            correlation_id, resp.status, attempt + 1, max_retries, body_preview
                        )
                        time.sleep(backoff_base * (2 ** attempt))
                        continue
                return resp
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    self.logger.warning(
                        "[%s] Exception during request (%s). Retrying attempt %d/%d ...",
                        correlation_id, type(e).__name__, attempt + 1, max_retries
                    )
                    time.sleep(backoff_base * (2 ** attempt))
                    continue
                break

        # If we get here, all retries failed
        if last_exc:
            self.logger.error("[%s] Request failed after retries: %s", correlation_id, last_exc)
            raise last_exc
        raise RuntimeError("Request failed after retries with unknown error")

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform an API request via Playwright with tracking and error handling.
        NOTE: Use endpoint PATH (e.g., '/api/agentic-chat/v1') since base_url is set.
        """
        if not self.context:
            raise RuntimeError("API request context not initialized")

        path = f"/{endpoint.lstrip('/')}"
        correlation_id = correlation_id or str(uuid.uuid4())

        # Per-request headers
        req_headers = {"CLIENT-CORRELATION-ID": correlation_id}
        if headers:
            req_headers.update(headers)

        # Tracker (request)
        if self.tracker:
            self.tracker.add_request(method, self.base_url + path, req_headers, data, correlation_id)

        # Choose the right verb function (Playwright Python style)
        method_lower = method.lower()
        if method_lower == "get":
            verb = self.context.get
        elif method_lower == "post":
            verb = self.context.post
        elif method_lower == "put":
            verb = self.context.put
        elif method_lower == "patch":
            verb = self.context.patch
        elif method_lower == "delete":
            verb = self.context.delete
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        # Do the request with a few retries for 5xx/429
        resp = self._call_with_retry(
            verb,
            url_or_path=path,
            json_body=data,
            headers=req_headers,
            correlation_id=correlation_id
        )

        # Basic logging + body capture on failure
        try:
            resp.raise_for_status()
        except Exception as e:
            body = ""
            try:
                body = resp.text()
            except Exception:
                pass
            self.logger.error("[%s] HTTP error %s | Body: %s", correlation_id, resp.status, body[:1000])
            if self.tracker:
                self.tracker.add_error("HTTPError", f"status={resp.status} body={body[:1000]}", correlation_id)
            raise

        # Parse JSON (fallback to text)
        try:
            resp_data: Any = resp.json()
        except Exception:
            resp_data = resp.text()

        # Tracker (response)
        if self.tracker:
            self.tracker.add_response(
                resp.status,
                dict(resp.headers),
                resp_data,
                0.0,  # duration is captured in logs above; you can wire it through if you prefer
                correlation_id
            )

        return resp_data

    # ------------------------
    # Domain functions
    # ------------------------
    def initiate_chat(self, request_data: Dict[str, Any],
                      correlation_id: Optional[str] = None) -> Dict[str, Any]:
        """Initiate a new chat conversation."""
        correlation_id = correlation_id or f"INIT-{uuid.uuid4()}"
        if "conversationId" not in request_data:
            request_data["conversationId"] = "initial"

        self.logger.info("Initiating chat | correlation=%s", correlation_id)
        return self._make_request("POST", "/api/agentic-chat/v1",
                                  data=request_data,
                                  correlation_id=correlation_id)

    def send_message(
        self,
        conversation_id: str,
        chat_text: str,
        action: str = "proceed",
        headers: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Send a message in an existing conversation."""
        correlation_id = correlation_id or (headers.get("CLIENT-CORRELATION-ID") if headers else None) \
                         or f"MSG-{uuid.uuid4()}"

        payload = {
            "channelID": "BBVA",
            "conversationID": conversation_id,
            "requestType": "ComplaintCapture",
            "chatText": chat_text,
            "action": action
        }

        self.logger.info("Sending message | conversationID=%s | correlation=%s",
                         conversation_id, correlation_id)
        return self._make_request("POST", "/api/agentic-chat/v1",
                                  data=payload,
                                  headers=headers,
                                  correlation_id=correlation_id)

    def close(self) -> None:
        """Dispose Playwright request context."""
        if self.context:
            self.context.dispose()
        self.logger.info("API Client session closed")
