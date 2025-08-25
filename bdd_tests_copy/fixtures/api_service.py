import os
import ssl
import time
import uuid
from typing import Dict, Any, Optional, Callable, List, Union
from urllib.parse import urlparse

import playwright  # for version log
from playwright.sync_api import Playwright, APIRequestContext, Response

from bdd_tests.config.settings import Settings
from bdd_tests.utils.logger_config import get_logger
from bdd_tests.utils.request_response_tracker import RequestResponseTracker


def _candidate_origins(base_url: str) -> List[str]:
    """
    Produce a set of origins that might be involved due to redirects / default ports.
    """
    p = urlparse(base_url)
    scheme = p.scheme or "https"
    host = p.hostname
    if not host:
        return []
    ports = set()
    # explicit port if present
    if p.port:
        ports.add(p.port)
    # default ports
    ports.add(443 if scheme == "https" else 80)

    origins = {f"{scheme}://{host}:{port}" for port in ports}
    # also include without explicit port (some gateways match on that format)
    origins.add(f"{scheme}://{host}")
    return sorted(origins)


class ChatbotAPIClient:
    """Playwright-based API client with robust mTLS configuration and retries."""

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
    def _create_session(self, pw: Playwright) -> None:
        """
        Create APIRequestContext with SSL/mTLS. Supports:
          - PEM pair: CERT_PEM_PATH + KEY_PEM_PATH (+ optional chain in cert)
          - PKCS12/PFX: PFX_PATH + PFX_PASSPHRASE
          - Multiple candidate origins (scheme+host and scheme+host:port)
        """
        self.logger.info("Playwright version: %s", getattr(playwright, "__version__", "unknown"))

        extra_headers = {
            "Accept": "application/json",
            "User-Agent": "ChatbotAutomation/Playwright",
        }

        ignore_https_errors = not bool(self.config.VERIFY_SSL)
        client_certs: Optional[List[Dict[str, str]]] = None

        cert_origins = _candidate_origins(self.base_url)

        # ----- Pick certificate mode -----
        # Mode 1: PKCS#12 / PFX
        if getattr(self.config, "PFX_PATH", None):
            pfx_path = self.config.PFX_PATH
            if not os.path.isfile(pfx_path):
                raise FileNotFoundError(f"PFX file not found: {pfx_path}")
            passphrase = getattr(self.config, "PFX_PASSPHRASE", None) or ""
            client_certs = [{
                "origin": o,
                "pfx": pfx_path,
                "passphrase": passphrase
            } for o in cert_origins]
            self.logger.info("Configured PKCS#12 client certificate for origins=%s | pfx=%s",
                             cert_origins, pfx_path)

        # Mode 2: PEM pair
        elif getattr(self.config, "CERT_PEM_PATH", None) and getattr(self.config, "KEY_PEM_PATH", None):
            cert_path = self.config.CERT_PEM_PATH
            key_path = self.config.KEY_PEM_PATH
            if not os.path.isfile(cert_path):
                raise FileNotFoundError(f"CERT PEM not found: {cert_path}")
            if not os.path.isfile(key_path):
                raise FileNotFoundError(f"KEY PEM not found: {key_path}")

            client_certs = [{
                "origin": o,
                "cert": cert_path,
                "key": key_path
            } for o in cert_origins]
            self.logger.info("Configured PEM client certificate for origins=%s | cert=%s | key=%s",
                             cert_origins, cert_path, key_path)

        else:
            self.logger.warning("No client certificate configured; mTLS endpoints will fail.")

        # Build context
        self.context = pw.request.new_context(
            base_url=self.base_url,
            ignore_https_errors=ignore_https_errors,
            extra_http_headers=extra_headers,
            client_certificates=client_certs
        )

        if ignore_https_errors:
            self.logger.warning("SSL verification DISABLED (ignore_https_errors=True)")
        else:
            self.logger.info("SSL verification ENABLED")

        # Optional: quick probe to force TLS and surface early mTLS issues
        try:
            probe = self.context.get("/", timeout=10_000)
            if probe.status >= 400:
                self.logger.warning("TLS probe status=%s body=%s",
                                    probe.status, probe.text()[:300])
        except Exception as e:
            self.logger.warning("TLS probe raised %s (usually harmless): %s", type(e).__name__, e)

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
        last_exc: Optional[Exception] = None
        for attempt in range(0, max_retries + 1):
            t0 = time.time()
            try:
                resp = fn(
                    url_or_path,
                    json=json_body if json_body is not None else None,  # ✅ proper JSON body
                    headers=headers,
                    timeout=self.timeout,
                )
                ms = int((time.time() - t0) * 1000)
                self.logger.info("[%s] HTTP %s (%d ms) -> %s", correlation_id, resp.status, ms, url_or_path)

                if resp.status in (429, 500, 502, 503, 504):
                    if attempt < max_retries:
                        self.logger.warning("[%s] Transient %s. Retrying %d/%d. Body: %s",
                                            correlation_id, resp.status, attempt + 1, max_retries,
                                            resp.text()[:300])
                        time.sleep(backoff_base * (2 ** attempt))
                        continue
                return resp
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    self.logger.warning("[%s] Exception %s. Retrying %d/%d...",
                                        correlation_id, type(e).__name__, attempt + 1, max_retries)
                    time.sleep(backoff_base * (2 ** attempt))
                    continue
                break

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
        if not self.context:
            raise RuntimeError("API request context not initialized")

        path = f"/{endpoint.lstrip('/')}"
        correlation_id = correlation_id or str(uuid.uuid4())

        req_headers = {"CLIENT-CORRELATION-ID": correlation_id}
        if headers:
            req_headers.update(headers)

        if self.tracker:
            self.tracker.add_request(method, self.base_url + path, req_headers, data, correlation_id)

        m = method.lower()
        if m == "get":
            verb = self.context.get
        elif m == "post":
            verb = self.context.post
        elif m == "put":
            verb = self.context.put
        elif m == "patch":
            verb = self.context.patch
        elif m == "delete":
            verb = self.context.delete
        else:
            raise ValueError(f"Unsupported HTTP method: {method}")

        resp = self._call_with_retry(
            verb,
            url_or_path=path,
            json_body=data,
            headers=req_headers,
            correlation_id=correlation_id
        )

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

        try:
            payload: Any = resp.json()
        except Exception:
            payload = resp.text()

        if self.tracker:
            self.tracker.add_response(resp.status, dict(resp.headers), payload, 0.0, correlation_id)

        return payload

    # ------------------------
    # Domain calls
    # ------------------------
    def initiate_chat(self, request_data: Dict[str, Any],
                      correlation_id: Optional[str] = None) -> Dict[str, Any]:
        correlation_id = correlation_id or f"INIT-{uuid.uuid4()}"
        if "conversationId" not in request_data:
            request_data["conversationId"] = "initial"
        self.logger.info("Initiating chat | correlation=%s", correlation_id)
        return self._make_request("POST", "/api/agentic-chat/v1", data=request_data, correlation_id=correlation_id)

    def send_message(
        self,
        conversation_id: str,
        chat_text: str,
        action: str = "proceed",
        headers: Optional[Dict[str, str]] = None,
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
        self.logger.info("Sending message | conversationID=%s | correlation=%s", conversation_id, correlation_id)
        return self._make_request("POST", "/api/agentic-chat/v1", data=payload, headers=headers,
                                  correlation_id=correlation_id)

    def close(self) -> None:
        if self.context:
            self.context.dispose()
        self.logger.info("API Client session closed")
