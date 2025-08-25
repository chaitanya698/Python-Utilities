import os
import time
import uuid
import threading
from typing import Dict, Any, Optional, Callable, List, Tuple, Union
from urllib.parse import urlparse
from queue import Queue, Empty

from playwright.sync_api import sync_playwright, APIRequestContext, Response

from bdd_tests.config.settings import Settings
from bdd_tests.utils.logger_config import get_logger
from bdd_tests.utils.request_response_tracker import RequestResponseTracker


# ---------- helpers ----------

def _candidate_origins(base_url: str) -> List[str]:
    """Generate plausible origins for mTLS binding (with and without default port)."""
    p = urlparse(base_url)
    scheme = p.scheme or "https"
    host = p.hostname
    if not host:
        return []
    port = p.port or (443 if scheme == "https" else 80)

    origins = {f"{scheme}://{host}:{port}", f"{scheme}://{host}"}
    return sorted(origins)


# Task signature for the worker queue
Task = Tuple[
    str,                    # method: GET/POST/PUT/PATCH/DELETE
    str,                    # path (leading slash ok)
    Optional[Dict[str, Any]],  # json body
    Dict[str, str],         # headers
    int,                    # timeout (ms)
    int,                    # max_retries
    float,                  # backoff base
]


class _PlaywrightWorker:
    """Owns the sync_playwright() lifecycle in a dedicated thread."""

    def __init__(self, base_url: str, ignore_https_errors: bool,
                 client_certs: Optional[List[Dict[str, str]]],
                 extra_headers: Dict[str, str], logger):
        self.base_url = base_url.rstrip("/")
        self.ignore_https_errors = ignore_https_errors
        self.client_certs = client_certs
        self.extra_headers = extra_headers
        self.logger = logger

        self._thread: Optional[threading.Thread] = None
        self._task_q: Queue = Queue()
        self._result_q: Queue = Queue()
        self._stop = threading.Event()

        self._context: Optional[APIRequestContext] = None

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="playwright-worker", daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        # send a sentinel to unblock queue if needed
        self._task_q.put(None)
        if self._thread:
            self._thread.join(timeout=10)

    def _run(self):
        try:
            with sync_playwright() as pw:
                self._context = pw.request.new_context(
                    base_url=self.base_url,
                    ignore_https_errors=self.ignore_https_errors,
                    extra_http_headers=self.extra_headers,
                    client_certificates=self.client_certs
                )

                # Optional probe for early TLS/mTLS feedback
                try:
                    probe = self._context.get("/", timeout=10_000)
                    if probe.status >= 400:
                        self.logger.warning("TLS probe status=%s body=%s",
                                            probe.status, probe.text()[:300])
                except Exception as e:
                    self.logger.debug("TLS probe raised %s: %s", type(e).__name__, e)

                while not self._stop.is_set():
                    task = self._task_q.get()
                    if task is None:
                        break  # sentinel for shutdown
                    try:
                        result = self._execute_task(task)
                        self._result_q.put((True, result, None))
                    except Exception as e:
                        self._result_q.put((False, None, e))
        finally:
            try:
                if self._context:
                    self._context.dispose()
            except Exception:
                pass

    def _execute_task(self, task: Task) -> Response:
        method, path, json_body, headers, timeout_ms, max_retries, backoff_base = task
        path = f"/{path.lstrip('/')}"
        m = method.lower()
        verb_map: Dict[str, Callable[..., Response]] = {
            "get": self._context.get,
            "post": self._context.post,
            "put": self._context.put,
            "patch": self._context.patch,
            "delete": self._context.delete,
        }
        if m not in verb_map:
            raise ValueError(f"Unsupported method {method}")

        verb = verb_map[m]
        last_exc: Optional[Exception] = None

        for attempt in range(0, max_retries + 1):
            t0 = time.time()
            try:
                resp = verb(path, json=json_body if json_body is not None else None,
                            headers=headers, timeout=timeout_ms)
                ms = int((time.time() - t0) * 1000)
                self.logger.info("HTTP %s (%d ms) %s", resp.status, ms, path)

                if resp.status in (429, 500, 502, 503, 504) and attempt < max_retries:
                    self.logger.warning("Transient %s. Retry %d/%d. Body: %s",
                                        resp.status, attempt + 1, max_retries, resp.text()[:300])
                    time.sleep(backoff_base * (2 ** attempt))
                    continue
                return resp
            except Exception as e:
                last_exc = e
                if attempt < max_retries:
                    self.logger.warning("Exception %s. Retry %d/%d...", type(e).__name__, attempt + 1, max_retries)
                    time.sleep(backoff_base * (2 ** attempt))
                    continue
                break

        if last_exc:
            raise last_exc
        raise RuntimeError("Request failed after retries (unknown)")

    def submit(self, task: Task, wait: bool = True, timeout: Optional[float] = None) -> Response:
        """Submit a task to the worker; block until result if wait=True."""
        self._task_q.put(task)
        if not wait:
            return None
        ok, val, err = self._result_q.get(timeout=timeout)
        if ok:
            return val
        raise err


# ---------- public client ----------

class ChatbotAPIClient:
    """Sync facade; internally uses a thread to keep Playwright out of any asyncio loop."""

    def __init__(self, config: Settings, tracker: Optional[RequestResponseTracker] = None):
        self.config = config
        self.logger = get_logger(__name__)
        self.base_url = config.API_BASE_URL.rstrip("/")
        self.timeout_ms = int(config.API_TIMEOUT * 1000)
        self.tracker = tracker

        # SSL / mTLS config
        ignore_https_errors = not bool(self.config.VERIFY_SSL)
        extra_headers = {
            "Accept": "application/json",
            "User-Agent": "ChatbotAutomation/Playwright",
        }

        # Build client certificates list (PFX OR PEM)
        client_certs: Optional[List[Dict[str, str]]] = None
        origins = _candidate_origins(self.base_url)

        if getattr(self.config, "PFX_PATH", None):
            pfx = self.config.PFX_PATH
            if not os.path.isfile(pfx):
                raise FileNotFoundError(f"PFX file not found: {pfx}")
            passphrase = getattr(self.config, "PFX_PASSPHRASE", "") or ""
            client_certs = [{"origin": o, "pfx": pfx, "passphrase": passphrase} for o in origins]
            self.logger.info("Configured PFX mTLS for origins=%s | pfx=%s", origins, pfx)
        elif getattr(self.config, "CERT_PEM_PATH", None) and getattr(self.config, "KEY_PEM_PATH", None):
            cert = self.config.CERT_PEM_PATH
            key = self.config.KEY_PEM_PATH
            if not os.path.isfile(cert):
                raise FileNotFoundError(f"CERT PEM not found: {cert}")
            if not os.path.isfile(key):
                raise FileNotFoundError(f"KEY PEM not found: {key}")
            client_certs = [{"origin": o, "cert": cert, "key": key} for o in origins]
            self.logger.info("Configured PEM mTLS for origins=%s | cert=%s | key=%s", origins, cert, key)
        else:
            self.logger.warning("No client certificate configured; mTLS endpoints will fail.")

        # Spin up worker thread that owns sync_playwright()
        self._worker = _PlaywrightWorker(
            base_url=self.base_url,
            ignore_https_errors=ignore_https_errors,
            client_certs=client_certs,
            extra_headers=extra_headers,
            logger=self.logger
        )
        self._worker.start()

    # ---- internal request ----
    def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]],
        headers: Optional[Dict[str, str]],
        correlation_id: Optional[str],
        retries: int = 2,
        backoff_base: float = 0.4
    ) -> Dict[str, Any]:
        path = f"/{endpoint.lstrip('/')}"
        cid = correlation_id or str(uuid.uuid4())

        req_headers = {"CLIENT-CORRELATION-ID": cid}
        if headers:
            req_headers.update(headers)

        if self.tracker:
            self.tracker.add_request(method, self.base_url + path, req_headers, data, cid)

        resp = self._worker.submit(
            task=(method, path, data, req_headers, self.timeout_ms, retries, backoff_base),
            wait=True, timeout=(self.timeout_ms / 1000.0) + 10
        )

        try:
            resp.raise_for_status()
        except Exception as e:
            body = ""
            try:
                body = resp.text()
            except Exception:
                pass
            self.logger.error("[%s] HTTP error %s | Body: %s", cid, resp.status, body[:1000])
            if self.tracker:
                self.tracker.add_error("HTTPError", f"status={resp.status} body={body[:1000]}", cid)
            raise

        try:
            payload: Any = resp.json()
        except Exception:
            payload = resp.text()

        if self.tracker:
            self.tracker.add_response(resp.status, dict(resp.headers), payload, 0.0, cid)

        return payload

    # ---- domain calls ----
    def initiate_chat(self, request_data: Dict[str, Any], correlation_id: Optional[str] = None) -> Dict[str, Any]:
        cid = correlation_id or f"INIT-{uuid.uuid4()}"
        if "conversationId" not in request_data:
            request_data["conversationId"] = "initial"
        self.logger.info("Initiating chat | correlation=%s", cid)
        return self._request("POST", "/api/agentic-chat/v1", data=request_data, headers=None, correlation_id=cid)

    def send_message(
        self,
        conversation_id: str,
        chat_text: str,
        action: str = "proceed",
        headers: Optional[Dict[str, str]] = None,
        correlation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        cid = correlation_id or (headers.get("CLIENT-CORRELATION-ID") if headers else None) or f"MSG-{uuid.uuid4()}"
        payload = {
            "channelID": "BBVA",
            "conversationID": conversation_id,
            "requestType": "ComplaintCapture",
            "chatText": chat_text,
            "action": action
        }
        self.logger.info("Sending message | conversationID=%s | correlation=%s", conversation_id, cid)
        return self._request("POST", "/api/agentic-chat/v1", data=payload, headers=headers, correlation_id=cid)

    def close(self) -> None:
        """Dispose the worker (which disposes the Playwright context)."""
        try:
            self._worker.stop()
        finally:
            self.logger.info("API Client session closed")
