"""Durable background worker for deferred-memory extraction pipelines (TASK A-5b).

Standard-library-only execution and lifecycle manager providing opt-in background
draining, deterministic synchronous execution hooks, clean daemon lifecycle,
and robust failure containment around an injected extraction processor.
"""

from __future__ import annotations

import math
import threading
from typing import Any, Callable

from memory_deferred_queue import DeferredMemoryQueue

__all__ = ["DeferredMemoryWorker"]

REQUIRED_QUEUE_METHODS = ("capture", "recover", "claim_due", "complete", "fail", "status")


class DeferredMemoryWorker:
    """Daemon background worker and deterministic execution hook for DeferredMemoryQueue."""

    def __init__(
        self,
        queue: DeferredMemoryQueue,
        processor: Callable[[str], dict[str, Any]],
        poll_seconds: float = 5,
        lease_seconds: float = 300,
        retry_seconds: float = 30,
    ) -> None:
        if queue is None:
            raise ValueError("queue must be provided")
        for method_name in REQUIRED_QUEUE_METHODS:
            method = getattr(queue, method_name, None)
            if method is None or not callable(method):
                raise ValueError(f"queue missing required callable: {method_name}")

        if not callable(processor):
            raise ValueError("processor must be callable")

        if (
            isinstance(poll_seconds, bool)
            or not isinstance(poll_seconds, (int, float))
            or math.isnan(poll_seconds)
            or math.isinf(poll_seconds)
            or poll_seconds <= 0
            or poll_seconds > 86400
        ):
            raise ValueError("poll_seconds must be a finite positive number <= 86400")

        if (
            isinstance(lease_seconds, bool)
            or not isinstance(lease_seconds, (int, float))
            or math.isnan(lease_seconds)
            or math.isinf(lease_seconds)
            or lease_seconds <= 0
            or lease_seconds > 86400
        ):
            raise ValueError("lease_seconds must be a finite positive number <= 86400")

        if (
            isinstance(retry_seconds, bool)
            or not isinstance(retry_seconds, (int, float))
            or math.isnan(retry_seconds)
            or math.isinf(retry_seconds)
            or retry_seconds < 0
            or retry_seconds > 86400
        ):
            raise ValueError("retry_seconds must be a finite non-negative number <= 86400")

        self.queue = queue
        self.processor = processor
        self.poll_seconds = float(poll_seconds)
        self.lease_seconds = float(lease_seconds)
        self.retry_seconds = float(retry_seconds)

        self._process_lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._state_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._wake_event = threading.Event()

        self._thread: threading.Thread | None = None
        self._last_run_status: dict[str, Any] | None = None

    def _record_status(self, result: dict[str, Any]) -> dict[str, Any]:
        with self._state_lock:
            self._last_run_status = dict(result)
        return result

    def is_running(self) -> bool:
        with self._lifecycle_lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def running(self) -> bool:
        return self.is_running()

    def capture(self, text: str | None = None, *, content: str | None = None) -> str:
        """Append user message to queue and signal worker wake event."""
        payload = text if text is not None else content
        if payload is None:
            raise ValueError("text must be a string")
        msg_id = self.queue.capture(payload)
        self._wake_event.set()
        return msg_id

    def process_once(self) -> dict[str, Any]:
        """Deterministic single-item synchronous processing hook."""
        if not self._process_lock.acquire(blocking=False):
            return self._record_status({"status": "busy"})

        try:
            try:
                self.queue.recover()
                claim = self.queue.claim_due(lease_seconds=self.lease_seconds)
            except Exception:
                return self._record_status(
                    {"status": "error", "error_category": "storage_failed"}
                )

            if claim is None:
                return self._record_status({"status": "idle"})

            msg_id = claim["message_id"]
            raw_content = claim["content"]
            claim_token = claim["claim_token"]

            is_success = False
            error_category = "processor_failed"

            try:
                res = self.processor(raw_content)
                if isinstance(res, dict) and res.get("status") == "ok":
                    is_success = True
                elif isinstance(res, dict) and res.get("status") == "failed":
                    if res.get("error_category") == "index_failed":
                        error_category = "index_failed"
                    else:
                        error_category = "processor_failed"
                else:
                    error_category = "invalid_result"
            except Exception:
                error_category = "processor_failed"

            if is_success:
                try:
                    ack = self.queue.complete(msg_id, claim_token)
                except Exception:
                    return self._record_status(
                        {"status": "error", "error_category": "storage_failed"}
                    )
                return self._record_status({"status": "ok"} if ack else {"status": "stale"})
            else:
                try:
                    ack = self.queue.fail(
                        msg_id,
                        claim_token,
                        error_category=error_category,
                        retry_seconds=self.retry_seconds,
                    )
                except Exception:
                    return self._record_status(
                        {"status": "error", "error_category": "storage_failed"}
                    )
                if ack:
                    return self._record_status(
                        {"status": "retry", "error_category": error_category}
                    )
                return self._record_status({"status": "stale"})
        finally:
            self._process_lock.release()

    def start(self) -> bool:
        """Start a single daemon background processing thread if not already running."""
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return False
            self._stop_event.clear()
            self._wake_event.set()
            thread = threading.Thread(
                target=self._run,
                name="DeferredMemoryWorker",
                daemon=True,
            )
            self._thread = thread
            thread.start()
            return True

    def _run(self) -> None:
        """Background thread main loop."""
        while not self._stop_event.is_set():
            try:
                res = self.process_once()
                status = res.get("status")
            except Exception:
                with self._state_lock:
                    self._last_run_status = {"status": "error", "error_category": "storage_failed"}
                status = "error"

            if self._stop_event.is_set():
                break

            if status == "ok":
                continue

            self._wake_event.wait(timeout=self.poll_seconds)
            self._wake_event.clear()

    def stop(self, timeout: float = 5) -> bool:
        """Signal background thread to stop and wait up to timeout seconds."""
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or math.isnan(timeout)
            or math.isinf(timeout)
            or timeout < 0
            or timeout > 86400
        ):
            raise ValueError("timeout must be a finite non-negative number <= 86400")
        timeout = float(timeout)

        if self._thread is not None and threading.current_thread() is self._thread:
            raise RuntimeError("Cannot join self")

        with self._lifecycle_lock:
            self._stop_event.set()
            self._wake_event.set()
            thread = self._thread
            if thread is None or not thread.is_alive():
                self._thread = None
                return not self._process_lock.locked()

        thread.join(timeout=timeout)

        with self._lifecycle_lock:
            if not thread.is_alive():
                if self._thread is thread:
                    self._thread = None
                return (self._thread is None or not self._thread.is_alive()) and not self._process_lock.locked()
            return False

    def status(self) -> dict[str, Any]:
        """Return worker running state, last execution status, and queue snapshot."""
        running = self.is_running()
        with self._state_lock:
            last_status = dict(self._last_run_status) if self._last_run_status is not None else None
        q_status = self.queue.status()
        return {"running": running, "last_run_status": last_status, **q_status}
