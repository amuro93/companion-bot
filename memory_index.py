"""Scoped companion memory index-job processor and optional stdlib Qdrant adapter (P1b).

Provides:
- `Embedder` and `VectorBackend` protocols (dependency-injected; no model loaded here).
- `IndexWorker`: single-instance, non-autonomous batch processor for scoped `index_jobs`
  with canonical SQLite pre-check and post-write recheck before acknowledgement.
- `QdrantHTTPBackend`: optional standard-library HTTP adapter with deterministic
  model-bound collection naming, lazy collection verification/creation, and
  defensive response parsing.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import re
import threading
from typing import Any, Callable, Protocol, Sequence, runtime_checkable
import urllib.error
import urllib.parse
import urllib.request
import uuid

from memory_core import (
    MAX_ID_LENGTH,
    MAX_LIMIT,
    VALID_DOMAINS,
    MemoryStore,
    _validate_id,
    _validate_limit,
)

MAX_DIMENSION = 65_536
MAX_COLLECTION_PREFIX_LENGTH = 64
MAX_HTTP_RESPONSE_BYTES = 1_048_576

ALLOWED_PAYLOAD_KEYS = frozenset(
    {
        "memory_id",
        "owner_id",
        "creator_character_id",
        "visible_to",
        "domain",
        "embedding_model_id",
    }
)

_COLLECTION_PREFIX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


class IndexErrorBase(Exception):
    """Base exception for sanitized indexing errors."""

    category: str = "index_error"
    reason: str = "index_failure"

    def __init__(
        self,
        message: str = "Index operation failed",
        *,
        category: str | None = None,
        reason: str | None = None,
    ) -> None:
        super().__init__(message)
        if category is not None:
            self.category = category
        if reason is not None:
            self.reason = reason


class QdrantBackendError(IndexErrorBase):
    """Sanitized error raised when Qdrant HTTP communication or response validation fails."""

    category = "backend_error"
    reason = "qdrant_backend_failure"


class QdrantConfigMismatchError(QdrantBackendError, ValueError):
    """Raised when existing collection or caller vector configuration mismatches."""

    category = "validation_error"
    reason = "collection_config_mismatch"


class _JobProcessingError(IndexErrorBase):
    """Internal wrapper carrying sanitized category, reason, and original exception type."""

    def __init__(
        self,
        category: str,
        reason: str,
        error_type: str,
    ) -> None:
        super().__init__(reason, category=category, reason=reason)
        self.error_type = error_type


@runtime_checkable
class Embedder(Protocol):
    """Protocol for an injected text embedding provider."""

    model_id: str
    dimension: int

    def embed(self, text: str) -> list[float]:
        """Embed a single text string into a finite float vector of length `dimension`."""
        ...


@runtime_checkable
class VectorBackend(Protocol):
    """Protocol for an injected external vector index backend."""

    def ensure_collection(self, dimension: int, model_id: str) -> None:
        """Ensure the target collection exists and matches `(dimension, model_id)`."""
        ...

    def upsert(
        self, memory_id: str, vector: list[float], payload: dict[str, Any]
    ) -> None:
        """Upsert a point by canonical UUID `memory_id` with content-free metadata payload."""
        ...

    def delete(self, memory_id: str) -> None:
        """Delete a point by canonical UUID `memory_id` idempotently."""
        ...


def _validate_uuid(value: Any, name: str = "memory_id") -> str:
    """Validate that value is a canonical lowercase UUID string."""
    raw = _validate_id(value, name)
    try:
        parsed = uuid.UUID(raw)
    except (ValueError, AttributeError, TypeError) as exc:
        raise ValueError(f"{name} must be a valid UUID string") from exc
    if str(parsed) != raw:
        raise ValueError(f"{name} must be in canonical lowercase UUID format")
    return raw


def _validate_dimension(dimension: Any, name: str = "dimension") -> int:
    """Validate that dimension is a bounded positive integer (rejecting booleans)."""
    if isinstance(dimension, bool) or not isinstance(dimension, int):
        raise ValueError(f"{name} must be an integer, not bool or other type")
    if dimension <= 0 or dimension > MAX_DIMENSION:
        raise ValueError(f"{name} must be between 1 and {MAX_DIMENSION}")
    return dimension


def _validate_vector(vector: Any, expected_dimension: int) -> list[float]:
    """Validate that vector is a sequence of finite numeric values matching expected_dimension."""
    expected_dimension = _validate_dimension(expected_dimension, "expected_dimension")
    if isinstance(vector, (str, bytes, bytearray, dict)) or not isinstance(
        vector, Sequence
    ):
        raise ValueError("vector must be a sequence of finite numbers")
    if len(vector) != expected_dimension:
        raise ValueError(
            f"vector length {len(vector)} does not match expected dimension {expected_dimension}"
        )

    cleaned: list[float] = []
    for idx, val in enumerate(vector):
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise ValueError(f"vector[{idx}] must be a float or int, not bool or other type")
        fval = float(val)
        if not math.isfinite(fval):
            raise ValueError(f"vector[{idx}] must be finite (NaN/Inf not allowed)")
        cleaned.append(fval)
    return cleaned


def _validate_index_payload(
    payload: Any,
    *,
    expected_memory_id: str,
    expected_model_id: str,
) -> dict[str, Any]:
    """Validate that backend payload contains strictly the content-free schema."""
    if not isinstance(payload, dict):
        raise ValueError("payload must be a dict")
    keys = set(payload.keys())
    if keys != ALLOWED_PAYLOAD_KEYS:
        extra = sorted(keys - ALLOWED_PAYLOAD_KEYS)
        missing = sorted(ALLOWED_PAYLOAD_KEYS - keys)
        raise ValueError(
            f"payload keys mismatch; extra={extra}, missing={missing}"
        )

    memory_id = _validate_uuid(payload["memory_id"], "payload.memory_id")
    if memory_id != expected_memory_id:
        raise ValueError("payload.memory_id must match point memory_id")

    owner_id = _validate_id(payload["owner_id"], "payload.owner_id")
    creator_character_id = _validate_id(
        payload["creator_character_id"], "payload.creator_character_id"
    )

    domain = payload["domain"]
    if not isinstance(domain, str) or domain not in VALID_DOMAINS:
        raise ValueError(f"payload.domain must be one of {sorted(VALID_DOMAINS)}")

    model_id = _validate_id(
        payload["embedding_model_id"], "payload.embedding_model_id"
    )
    if model_id != expected_model_id:
        raise ValueError("payload.embedding_model_id does not match expected model_id")

    raw_visible = payload["visible_to"]
    if isinstance(raw_visible, (str, bytes, bytearray, dict)) or not isinstance(
        raw_visible, (list, tuple)
    ):
        raise ValueError("payload.visible_to must be a list of character IDs")
    if not raw_visible:
        raise ValueError("payload.visible_to must not be empty")

    visible_list: list[str] = []
    seen: set[str] = set()
    for idx, cid in enumerate(raw_visible):
        valid_cid = _validate_id(cid, f"payload.visible_to[{idx}]")
        if valid_cid in seen:
            raise ValueError("payload.visible_to must not contain duplicate IDs")
        seen.add(valid_cid)
        visible_list.append(valid_cid)

    if visible_list != sorted(visible_list):
        raise ValueError("payload.visible_to must be sorted ascending")
    if creator_character_id not in seen:
        raise ValueError("payload.visible_to must include creator_character_id")

    return {
        "memory_id": memory_id,
        "owner_id": owner_id,
        "creator_character_id": creator_character_id,
        "visible_to": visible_list,
        "domain": domain,
        "embedding_model_id": model_id,
    }


def _validate_collection_prefix(prefix: Any) -> str:
    """Validate that a logical collection prefix is non-empty and URL/identifier safe."""
    if not isinstance(prefix, str):
        raise ValueError("collection prefix must be a string")
    if not prefix or prefix != prefix.strip():
        raise ValueError(
            "collection prefix must be a non-empty string without surrounding whitespace"
        )
    if len(prefix) > MAX_COLLECTION_PREFIX_LENGTH:
        raise ValueError(
            f"collection prefix exceeds max length of {MAX_COLLECTION_PREFIX_LENGTH}"
        )
    if not _COLLECTION_PREFIX_RE.match(prefix):
        raise ValueError(
            "collection prefix must start with alphanumeric and contain only [A-Za-z0-9_-]"
        )
    return prefix


def derive_collection_name(
    collection_prefix: str, model_id: str, dimension: int
) -> str:
    """Derive canonical Qdrant collection name from `(collection_prefix, model_id, dimension)`.

    Formula:
        `{collection_prefix}_d{dimension}_{sha256_12}`
    where `sha256_12` is the first 12 lowercase hex characters of
    `SHA256("{collection_prefix}\\n{model_id}\\n{dimension}")`.
    This guarantees that collections for different embedding models (even when they
    share the exact same vector dimension) never mix vectors in the same collection.
    """
    prefix = _validate_collection_prefix(collection_prefix)
    norm_model = _validate_id(model_id, "model_id")
    norm_dim = _validate_dimension(dimension, "dimension")
    canonical_input = f"{prefix}\n{norm_model}\n{norm_dim}".encode("utf-8")
    suffix = hashlib.sha256(canonical_input).hexdigest()[:12]
    return f"{prefix}_d{norm_dim}_{suffix}"


def _validate_base_url(base_url: Any) -> str:
    """Validate that base_url is an explicit http/https URL without credentials, query, or fragment."""
    if not isinstance(base_url, str):
        raise ValueError("base_url must be a string")
    if not base_url or base_url != base_url.strip():
        raise ValueError(
            "base_url must be a non-empty string without leading or trailing whitespace"
        )
    if any(ord(ch) < 32 for ch in base_url):
        raise ValueError("base_url must not contain control characters")

    parsed = urllib.parse.urlsplit(base_url)
    if parsed.scheme.lower() not in ("http", "https"):
        raise ValueError("base_url scheme must be 'http' or 'https'")
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("base_url must include a valid host")
    if (
        parsed.username is not None
        or parsed.password is not None
        or "@" in parsed.netloc
    ):
        raise ValueError("base_url must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("base_url must not contain query parameters or fragments")
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ValueError("base_url contains an invalid port") from exc

    normalized_path = parsed.path.rstrip("/")
    if ".." in normalized_path.split("/"):
        raise ValueError("base_url path must not contain '..' segments")

    reconstructed = urllib.parse.urlunsplit(
        (parsed.scheme.lower(), parsed.netloc, normalized_path, "", "")
    )
    return reconstructed.rstrip("/")


def _validate_api_key(api_key: Any) -> str | None:
    """Validate optional API key to prevent header injection or empty credentials."""
    if api_key is None:
        return None
    if not isinstance(api_key, str):
        raise ValueError("api_key must be a string or None")
    if not api_key or api_key != api_key.strip():
        raise ValueError(
            "api_key must be a non-empty string without surrounding whitespace"
        )
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in api_key):
        raise ValueError("api_key must not contain control characters")
    return api_key


def _validate_timeout(timeout: Any) -> float:
    """Validate that timeout is a finite positive number (rejecting booleans)."""
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout must be a positive finite int or float")
    fval = float(timeout)
    if not math.isfinite(fval) or fval <= 0.0:
        raise ValueError("timeout must be a finite number greater than 0")
    return fval


class IndexWorker:
    """Synchronous, scoped index-job batch processor.

    Operational precondition:
        At most one worker process may execute against a given scope/collection at a time.
        This class uses a non-blocking per-instance lock to prevent overlapping or
        re-entrant `run_once` calls on the same `IndexWorker` instance; it does not
        implement a distributed cross-process lock.
    """

    def __init__(
        self,
        store: MemoryStore,
        embedder: Embedder | None,
        backend: VectorBackend,
    ) -> None:
        if store is None or not (
            callable(getattr(store, "pending_jobs", None))
            and callable(getattr(store, "get_index_record", None))
            and callable(getattr(store, "ack_index_job", None))
        ):
            raise ValueError(
                "store must provide pending_jobs, get_index_record, and ack_index_job"
            )
        if backend is None or not (
            callable(getattr(backend, "ensure_collection", None))
            and callable(getattr(backend, "upsert", None))
            and callable(getattr(backend, "delete", None))
        ):
            raise ValueError(
                "backend must provide ensure_collection, upsert, and delete"
            )

        self._store = store
        self._embedder = embedder
        self._backend = backend
        self._instance_lock = threading.Lock()

    def _validate_embedder_metadata(self) -> tuple[str, int]:
        """Validate embedder availability, model_id, dimension, and backend compatibility."""
        if self._embedder is None:
            raise _JobProcessingError(
                category="embedding_error",
                reason="embedder_unavailable",
                error_type="RuntimeError",
            )

        try:
            raw_model_id = getattr(self._embedder, "model_id", None)
            raw_dimension = getattr(self._embedder, "dimension", None)
            embed_fn = getattr(self._embedder, "embed", None)
        except Exception as exc:
            raise _JobProcessingError(
                category="embedding_error",
                reason="embedder_metadata_error",
                error_type=type(exc).__name__,
            ) from None

        if not callable(embed_fn):
            raise _JobProcessingError(
                category="embedding_error",
                reason="embedder_not_callable",
                error_type="TypeError",
            )

        try:
            model_id = _validate_id(raw_model_id, "embedder.model_id")
            dimension = _validate_dimension(raw_dimension, "embedder.dimension")
        except Exception as exc:
            raise _JobProcessingError(
                category="validation_error",
                reason="invalid_model_metadata",
                error_type=type(exc).__name__,
            ) from None

        backend_model = getattr(self._backend, "model_id", None)
        if backend_model is not None and backend_model != model_id:
            raise _JobProcessingError(
                category="validation_error",
                reason="model_id_mismatch",
                error_type="ValueError",
            )

        backend_dim = getattr(self._backend, "dimension", None)
        if backend_dim is not None and backend_dim != dimension:
            raise _JobProcessingError(
                category="validation_error",
                reason="dimension_mismatch",
                error_type="ValueError",
            )

        return model_id, dimension

    def _execute_delete(self, memory_id: str) -> None:
        """Validate UUID memory_id and invoke backend.delete with error sanitization."""
        try:
            valid_memory_id = _validate_uuid(memory_id, "memory_id")
        except Exception as exc:
            raise _JobProcessingError(
                category="validation_error",
                reason="invalid_memory_id",
                error_type=type(exc).__name__,
            ) from None

        try:
            self._backend.delete(valid_memory_id)
        except _JobProcessingError:
            raise
        except QdrantConfigMismatchError as exc:
            raise _JobProcessingError(
                category="validation_error",
                reason=exc.reason,
                error_type=type(exc).__name__,
            ) from None
        except Exception as exc:
            raise _JobProcessingError(
                category="backend_error",
                reason="backend_delete_failed",
                error_type=type(exc).__name__,
            ) from None

    def _ack_or_raise(self, owner_id: str, character_id: str, job_id: str) -> None:
        """Acknowledge a processed job or raise sanitized ack_error."""
        try:
            acked = self._store.ack_index_job(owner_id, character_id, job_id)
        except Exception as exc:
            raise _JobProcessingError(
                category="ack_error",
                reason="ack_exception",
                error_type=type(exc).__name__,
            ) from None

        if not acked:
            raise _JobProcessingError(
                category="ack_error",
                reason="ack_returned_false",
                error_type="RuntimeError",
            )

    def _ack_after_compensating_delete(
        self, owner_id: str, character_id: str, job_id: str, memory_id: str
    ) -> None:
        """Acknowledge job after post-upsert recheck found record gone/inactive and deleted vector.

        Handles two cases:
        1. `correct_memory` superseded `memory_id` during upsert: `job_id` is still in
           `index_jobs` and is deleted by `ack_index_job`.
        2. `delete_memory` deleted `memory_id` during upsert: `delete_memory` already
           removed `job_id` (the pending upsert) and queued a `delete` job for `memory_id`
           before our compensating `backend.delete(memory_id)` executed.
        """
        try:
            acked = self._store.ack_index_job(owner_id, character_id, job_id)
        except Exception as exc:
            raise _JobProcessingError(
                category="ack_error",
                reason="ack_exception",
                error_type=type(exc).__name__,
            ) from None

        if acked:
            return

        try:
            remaining = self._store.pending_jobs(
                owner_id, character_id, limit=MAX_LIMIT
            )
        except Exception as exc:
            raise _JobProcessingError(
                category="ack_error",
                reason="ack_recheck_failed",
                error_type=type(exc).__name__,
            ) from None

        if any(j["job_id"] == job_id for j in remaining):
            raise _JobProcessingError(
                category="ack_error",
                reason="ack_returned_false",
                error_type="RuntimeError",
            )

        for rem_job in remaining:
            if (
                rem_job["memory_id"] == memory_id
                and rem_job["action"] == "delete"
            ):
                try:
                    self._store.ack_index_job(
                        owner_id, character_id, str(rem_job["job_id"])
                    )
                except Exception as exc:
                    raise _JobProcessingError(
                        category="ack_error",
                        reason="ack_exception",
                        error_type=type(exc).__name__,
                    ) from None

    def _process_single_job(
        self, owner_id: str, character_id: str, job: dict[str, Any]
    ) -> str:
        """Process one job in sequence and return executed action ('upsert' or 'delete')."""
        job_id = _validate_id(job.get("job_id"), "job.job_id")
        memory_id = _validate_uuid(job.get("memory_id"), "job.memory_id")
        action = job.get("action")

        if action not in ("upsert", "delete"):
            raise _JobProcessingError(
                category="validation_error",
                reason="invalid_job_action",
                error_type="ValueError",
            )

        if action == "delete":
            self._execute_delete(memory_id)
            self._ack_or_raise(owner_id, character_id, job_id)
            return "delete"

        # action == 'upsert': read current canonical active record first
        try:
            record = self._store.get_index_record(owner_id, character_id, memory_id)
        except Exception as exc:
            raise _JobProcessingError(
                category="store_error",
                reason="store_read_failed",
                error_type=type(exc).__name__,
            ) from None

        if record is None:
            # Stale upsert for deleted or superseded memory: execute delete instead,
            # never call embedder, and never resurrect inactive memory.
            self._execute_delete(memory_id)
            self._ack_or_raise(owner_id, character_id, job_id)
            return "delete"

        snap_text = record.get("text")
        snap_domain = record.get("domain")
        snap_visible_to = list(record.get("visible_to", []))
        snap_creator = record.get("creator_character_id", record.get("character_id"))
        snap_owner = record.get("owner_id")

        if (
            snap_owner != owner_id
            or snap_creator != character_id
            or not isinstance(snap_text, str)
            or not snap_text.strip()
        ):
            raise _JobProcessingError(
                category="validation_error",
                reason="invalid_canonical_record",
                error_type="ValueError",
            )

        model_id, dimension = self._validate_embedder_metadata()

        try:
            self._backend.ensure_collection(dimension, model_id)
        except _JobProcessingError:
            raise
        except QdrantConfigMismatchError as exc:
            raise _JobProcessingError(
                category="validation_error",
                reason=exc.reason,
                error_type=type(exc).__name__,
            ) from None
        except ValueError as exc:
            raise _JobProcessingError(
                category="validation_error",
                reason="collection_validation_failed",
                error_type=type(exc).__name__,
            ) from None
        except Exception as exc:
            raise _JobProcessingError(
                category="backend_error",
                reason="backend_ensure_collection_failed",
                error_type=type(exc).__name__,
            ) from None

        assert self._embedder is not None
        try:
            raw_vector = self._embedder.embed(snap_text)
        except Exception as exc:
            raise _JobProcessingError(
                category="embedding_error",
                reason="embedder_execution_failed",
                error_type=type(exc).__name__,
            ) from None

        try:
            vector = _validate_vector(raw_vector, dimension)
            payload = _validate_index_payload(
                {
                    "memory_id": memory_id,
                    "owner_id": owner_id,
                    "creator_character_id": character_id,
                    "visible_to": snap_visible_to,
                    "domain": snap_domain,
                    "embedding_model_id": model_id,
                },
                expected_memory_id=memory_id,
                expected_model_id=model_id,
            )
        except Exception as exc:
            raise _JobProcessingError(
                category="validation_error",
                reason="invalid_vector_or_payload",
                error_type=type(exc).__name__,
            ) from None

        try:
            self._backend.upsert(memory_id, vector, payload)
        except _JobProcessingError:
            raise
        except QdrantConfigMismatchError as exc:
            raise _JobProcessingError(
                category="validation_error",
                reason=exc.reason,
                error_type=type(exc).__name__,
            ) from None
        except Exception as exc:
            raise _JobProcessingError(
                category="backend_error",
                reason="backend_upsert_failed",
                error_type=type(exc).__name__,
            ) from None

        # Post-write canonical state recheck before ACK
        try:
            post_record = self._store.get_index_record(
                owner_id, character_id, memory_id
            )
        except Exception as exc:
            raise _JobProcessingError(
                category="store_error",
                reason="store_recheck_failed",
                error_type=type(exc).__name__,
            ) from None

        if post_record is None:
            # Memory was deleted or superseded while backend.upsert was in flight:
            # immediately delete the vector from the backend before acknowledging.
            self._execute_delete(memory_id)
            self._ack_after_compensating_delete(
                owner_id, character_id, job_id, memory_id
            )
            return "delete"

        post_text = post_record.get("text")
        post_domain = post_record.get("domain")
        post_visible_to = list(post_record.get("visible_to", []))

        if (
            post_text != snap_text
            or post_domain != snap_domain
            or post_visible_to != snap_visible_to
        ):
            # Canonical ACL/domain/text changed during external write:
            # never acknowledge knowingly stale write; leave job pending for retry.
            raise _JobProcessingError(
                category="concurrent_modification",
                reason="stale_write_detected",
                error_type="StaleWriteError",
            )

        self._ack_or_raise(owner_id, character_id, job_id)
        return "upsert"

    def run_once(
        self, owner_id: str, character_id: str, limit: int = 100
    ) -> dict[str, Any]:
        """Process up to `limit` pending index jobs for `(owner_id, character_id)` in sequence.

        Returns a content-free structured report with processed, failed, and pending job details.
        Never includes memory text, message content, raw exception messages, or credentials.
        """
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        limit = _validate_limit(limit)

        if not self._instance_lock.acquire(blocking=False):
            raise RuntimeError(
                "IndexWorker.run_once is already executing on this worker instance"
            )

        try:
            jobs = self._store.pending_jobs(owner_id, character_id, limit=limit)
            processed_items: list[dict[str, Any]] = []
            processed_job_ids: list[str] = []
            upserted_count = 0
            deleted_count = 0
            failed_info: dict[str, Any] | None = None

            for idx, job in enumerate(jobs):
                job_id = str(job.get("job_id", ""))
                memory_id = str(job.get("memory_id", ""))
                action = str(job.get("action", ""))

                try:
                    executed_action = self._process_single_job(
                        owner_id, character_id, job
                    )
                except _JobProcessingError as jerr:
                    failed_info = {
                        "job_id": job_id,
                        "memory_id": memory_id,
                        "action": action,
                        "error_category": jerr.category,
                        "error_type": jerr.error_type,
                        "reason": jerr.reason,
                    }
                    break
                except Exception as exc:
                    failed_info = {
                        "job_id": job_id,
                        "memory_id": memory_id,
                        "action": action,
                        "error_category": "unexpected_error",
                        "error_type": type(exc).__name__,
                        "reason": "unexpected_failure",
                    }
                    break

                if executed_action == "upsert":
                    upserted_count += 1
                else:
                    deleted_count += 1

                processed_job_ids.append(job_id)
                processed_items.append(
                    {
                        "job_id": job_id,
                        "memory_id": memory_id,
                        "action": action,
                        "executed_action": executed_action,
                        "status": "processed",
                    }
                )

            try:
                remaining_jobs = self._store.pending_jobs(
                    owner_id, character_id, limit=MAX_LIMIT
                )
                pending_job_ids = [str(r["job_id"]) for r in remaining_jobs]
                pending_count_status = (
                    "lower_bound" if len(remaining_jobs) >= MAX_LIMIT else "exact"
                )
                pending_count_lower_bound = len(remaining_jobs)
            except Exception:
                # Fallback if store is unreachable after failure
                ack_set = set(processed_job_ids)
                pending_job_ids = [
                    str(j["job_id"])
                    for j in jobs
                    if str(j.get("job_id", "")) not in ack_set
                ]
                pending_count_status = "unknown"
                pending_count_lower_bound = 0

            return {
                "status": "ok" if failed_info is None else "failed",
                "owner_id": owner_id,
                "character_id": character_id,
                "fetched_count": len(jobs),
                "processed_count": len(processed_items),
                "processed_job_ids": processed_job_ids,
                "processed": processed_items,
                "upserted_count": upserted_count,
                "deleted_count": deleted_count,
                "failed_count": 0 if failed_info is None else 1,
                "failed_job_id": None if failed_info is None else failed_info["job_id"],
                "failed": failed_info,
                "failure": failed_info,
                "errors": [] if failed_info is None else [failed_info],
                "pending_count": (
                    pending_count_lower_bound if pending_count_status == "exact" else None
                ),
                "pending_count_status": pending_count_status,
                "pending_count_exact": pending_count_status == "exact",
                "pending_count_lower_bound": pending_count_lower_bound,
                "pending_job_ids": pending_job_ids,
            }
        finally:
            self._instance_lock.release()


class _NoRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Reject redirects before urllib can forward request headers to a new host."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class QdrantHTTPBackend:
    """Standard-library HTTP adapter for Qdrant point indexing and deletion.

    Design notes:
    - Performs zero network I/O at import or construction time.
    - Binds `(collection, model_id, dimension)` deterministically into `collection_name`
      so vectors from different embedding models cannot mix even if dimensions match.
    - Supports an injected `transport(method, url, body, headers, timeout)` callable
      for offline unit tests.
    - Never logs or includes `api_key` or raw response bodies in exceptions.
    """

    def __init__(
        self,
        base_url: str,
        collection: str,
        dimension: int,
        model_id: str,
        api_key: str | None = None,
        timeout: float = 10.0,
        transport: Callable[..., Any] | None = None,
        *,
        http_transport: Callable[..., Any] | None = None,
    ) -> None:
        self.base_url: str = _validate_base_url(base_url)
        self.collection_prefix: str = _validate_collection_prefix(collection)
        self.dimension: int = _validate_dimension(dimension, "dimension")
        self.model_id: str = _validate_id(model_id, "model_id")
        self._api_key: str | None = _validate_api_key(api_key)
        self.timeout: float = _validate_timeout(timeout)

        if (
            transport is not None
            and http_transport is not None
            and transport is not http_transport
        ):
            raise ValueError("Provide at most one of transport or http_transport")
        chosen_transport = transport if transport is not None else http_transport
        if chosen_transport is not None and not callable(chosen_transport):
            raise ValueError("transport must be callable or None")

        self._transport: Callable[..., Any] = (
            chosen_transport
            if chosen_transport is not None
            else self._default_urllib_transport
        )
        self.collection_name: str = derive_collection_name(
            self.collection_prefix, self.model_id, self.dimension
        )
        self._collection_ensured: bool = False

    def __repr__(self) -> str:
        return (
            f"QdrantHTTPBackend(base_url={self.base_url!r}, "
            f"collection_name={self.collection_name!r}, "
            f"dimension={self.dimension}, "
            f"model_id={self.model_id!r}, "
            f"api_key={'***' if self._api_key is not None else None})"
        )

    @staticmethod
    def _default_urllib_transport(
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, bytes]:
        """Execute an HTTP request using Python's standard-library urllib.request."""
        req = urllib.request.Request(
            url=url,
            data=body,
            headers=dict(headers),
            method=method,
        )
        try:
            with urllib.request.build_opener(_NoRedirectHandler()).open(req, timeout=timeout) as resp:
                status = int(getattr(resp, "status", resp.getcode()))
                raw = resp.read(MAX_HTTP_RESPONSE_BYTES + 1)
                if len(raw) > MAX_HTTP_RESPONSE_BYTES:
                    raise QdrantBackendError(
                        "Qdrant response exceeded maximum allowed size",
                        reason="response_too_large",
                    )
                return status, raw
        except urllib.error.HTTPError as exc:
            try:
                err_raw = exc.read(MAX_HTTP_RESPONSE_BYTES)
            except Exception:
                err_raw = b""
            finally:
                try:
                    exc.close()
                except Exception:
                    pass
            return int(exc.code), err_raw
        except QdrantBackendError:
            raise
        except TimeoutError as exc:
            raise QdrantBackendError(
                "Qdrant HTTP request timed out",
                reason="http_timeout",
            ) from None
        except urllib.error.URLError as exc:
            raise QdrantBackendError(
                "Qdrant network transport error",
                reason="network_error",
            ) from None
        except OSError as exc:
            raise QdrantBackendError(
                "Qdrant socket or OS transport error",
                reason="os_network_error",
            ) from None

    def _invoke_transport(
        self,
        method: str,
        url: str,
        body_bytes: bytes | None,
        headers: dict[str, str],
    ) -> tuple[int, Any]:
        """Invoke the configured transport callable and normalize `(status, body)`."""
        args: tuple[Any, ...] = (
            method,
            url,
            body_bytes,
            dict(headers),
            self.timeout,
        )
        try:
            sig = inspect.signature(self._transport)
            params = list(sig.parameters.values())
            has_varargs = any(
                p.kind == inspect.Parameter.VAR_POSITIONAL for p in params
            )
            if not has_varargs:
                positional = [
                    p
                    for p in params
                    if p.kind
                    in (
                        inspect.Parameter.POSITIONAL_ONLY,
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    )
                ]
                if 0 < len(positional) < len(args):
                    args = args[: len(positional)]
        except (ValueError, TypeError):
            pass

        try:
            raw_resp = self._transport(*args)
        except IndexErrorBase:
            raise
        except TimeoutError:
            raise QdrantBackendError(
                "Qdrant transport timed out",
                reason="http_timeout",
            ) from None
        except Exception as exc:
            raise QdrantBackendError(
                "Qdrant transport call failed",
                reason="transport_failure",
            ) from None

        status_code: Any = None
        resp_body: Any = None

        if isinstance(raw_resp, tuple) and len(raw_resp) >= 2:
            status_code = raw_resp[0]
            resp_body = raw_resp[1]
        elif isinstance(raw_resp, dict):
            status_code = raw_resp.get("status", raw_resp.get("status_code"))
            resp_body = raw_resp.get("body")
        elif hasattr(raw_resp, "status") or hasattr(raw_resp, "status_code"):
            status_code = getattr(
                raw_resp, "status", getattr(raw_resp, "status_code", None)
            )
            if hasattr(raw_resp, "read") and callable(raw_resp.read):
                resp_body = raw_resp.read(MAX_HTTP_RESPONSE_BYTES)
            else:
                resp_body = getattr(raw_resp, "body", None)
        else:
            raise QdrantBackendError(
                "Invalid response structure from HTTP transport",
                reason="invalid_transport_response",
            )

        if isinstance(status_code, bool) or not isinstance(status_code, int):
            raise QdrantBackendError(
                "HTTP status code must be an integer",
                reason="invalid_http_status",
            )

        return status_code, resp_body

    def _request_json(
        self,
        method: str,
        path: str,
        payload_obj: dict[str, Any] | None = None,
    ) -> tuple[int, dict[str, Any] | None]:
        """Send an HTTP request and parse JSON response defensively."""
        url = f"{self.base_url}{path}"
        headers: dict[str, str] = {"Accept": "application/json"}
        body_bytes: bytes | None = None

        if payload_obj is not None:
            headers["Content-Type"] = "application/json"
            body_bytes = json.dumps(
                payload_obj,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")

        if self._api_key is not None:
            headers["api-key"] = self._api_key

        status, raw_body = self._invoke_transport(
            method, url, body_bytes, headers
        )

        if raw_body is None or raw_body == b"" or raw_body == "":
            return status, None

        if isinstance(raw_body, dict):
            return status, dict(raw_body)

        if isinstance(raw_body, (bytes, bytearray)):
            if len(raw_body) > MAX_HTTP_RESPONSE_BYTES:
                raise QdrantBackendError(
                    "Qdrant response body too large",
                    reason="response_too_large",
                )
            try:
                text_body = bytes(raw_body).decode("utf-8")
            except UnicodeDecodeError:
                raise QdrantBackendError(
                    "Qdrant response is not valid UTF-8",
                    reason="malformed_response_encoding",
                ) from None
        elif isinstance(raw_body, str):
            text_body = raw_body
        else:
            raise QdrantBackendError(
                "Unsupported response body type from transport",
                reason="invalid_response_body_type",
            )

        try:
            parsed = json.loads(text_body)
        except Exception:
            raise QdrantBackendError(
                "Malformed JSON in Qdrant response",
                reason="malformed_json_response",
            ) from None

        if not isinstance(parsed, dict):
            raise QdrantBackendError(
                "Qdrant JSON response root must be an object",
                reason="non_object_json_response",
            )

        return status, parsed

    def _validate_existing_collection_body(
        self, body: dict[str, Any] | None
    ) -> None:
        """Validate GET /collections/{name} 2xx body matches dimension and Cosine distance."""
        if not isinstance(body, dict):
            raise QdrantBackendError(
                "Empty or non-object response when inspecting Qdrant collection",
                reason="malformed_collection_info",
            )

        if body.get("status") != "ok":
            raise QdrantBackendError(
                "Qdrant collection inspection returned non-ok status",
                reason="qdrant_status_not_ok",
            )

        result = body.get("result")
        if not isinstance(result, dict):
            raise QdrantBackendError(
                "Qdrant collection inspection missing result object",
                reason="missing_collection_result",
            )

        vectors_cfg: Any = None
        config = result.get("config")
        if isinstance(config, dict):
            params = config.get("params")
            if isinstance(params, dict):
                vectors_cfg = params.get("vectors")
        if vectors_cfg is None and isinstance(result.get("params"), dict):
            vectors_cfg = result["params"].get("vectors")
        if vectors_cfg is None:
            vectors_cfg = result.get("vectors")

        if not isinstance(vectors_cfg, dict):
            raise QdrantConfigMismatchError(
                "Qdrant collection missing standard vectors configuration",
                reason="missing_vectors_config",
            )

        size = vectors_cfg.get("size")
        distance = vectors_cfg.get("distance")

        if isinstance(size, bool) or not isinstance(size, int):
            raise QdrantConfigMismatchError(
                "Qdrant collection vector size is missing or not an integer",
                reason="invalid_existing_vector_size",
            )
        if size != self.dimension:
            raise QdrantConfigMismatchError(
                f"Existing Qdrant collection vector size {size} != expected {self.dimension}",
                reason="collection_dimension_mismatch",
            )
        if not isinstance(distance, str) or distance.strip().lower() != "cosine":
            raise QdrantConfigMismatchError(
                "Existing Qdrant collection distance metric is not Cosine",
                reason="collection_distance_mismatch",
            )

    @staticmethod
    def _validate_mutation_response(
        status: int, body: dict[str, Any] | None, op_name: str
    ) -> None:
        """Defensively verify 2xx status and operation-specific Qdrant result for write calls.

        - `collection creation`: requires literal `result is True`.
        - `point upsert` / `point delete`: requires `result` dict with explicit
          `result["status"] == "completed"` and no error field. Missing status,
          `"acknowledged"`, `"ok"`, booleans, `{}`, `None`, or errors fail closed.
        """
        if (
            isinstance(status, bool)
            or not isinstance(status, int)
            or status < 200
            or status >= 300
        ):
            raise QdrantBackendError(
                f"Qdrant {op_name} failed with HTTP status {status}",
                reason=f"http_status_{status}",
            )
        if not isinstance(body, dict):
            raise QdrantBackendError(
                f"Qdrant {op_name} returned empty or non-JSON body",
                reason="empty_response_body",
            )
        if body.get("status") != "ok":
            raise QdrantBackendError(
                f"Qdrant {op_name} returned non-ok status field",
                reason="qdrant_status_not_ok",
            )
        if body.get("error") is not None or body.get("status_error") is not None:
            raise QdrantBackendError(
                f"Qdrant {op_name} response contained error field",
                reason="qdrant_response_error",
            )
        if "result" not in body:
            raise QdrantBackendError(
                f"Qdrant {op_name} response missing result field",
                reason="missing_result_field",
            )

        result = body["result"]
        if result is None or result is False:
            raise QdrantBackendError(
                f"Qdrant {op_name} reported unsuccessful result",
                reason="qdrant_result_false",
            )

        if op_name == "collection creation":
            if result is not True:
                raise QdrantBackendError(
                    f"Qdrant {op_name} requires literal result True",
                    reason="unexpected_result_type",
                )
            return

        if op_name in ("point upsert", "point delete"):
            if not isinstance(result, dict):
                raise QdrantBackendError(
                    f"Qdrant {op_name} requires result object",
                    reason="unexpected_result_type",
                )
            if result.get("error") is not None:
                raise QdrantBackendError(
                    f"Qdrant {op_name} result contained error field",
                    reason="qdrant_operation_failed",
                )
            op_status = result.get("status")
            if not isinstance(op_status, str) or op_status != "completed":
                raise QdrantBackendError(
                    f"Qdrant {op_name} did not report completed status",
                    reason="qdrant_operation_not_completed",
                )
            return

        raise QdrantBackendError(
            f"Unsupported Qdrant mutation operation: {op_name}",
            reason="unsupported_mutation_operation",
        )

    def ensure_collection(
        self, dimension: int | None = None, model_id: str | None = None
    ) -> None:
        """Lazily verify or create the dedicated model-bound Qdrant collection."""
        if dimension is not None:
            checked_dim = _validate_dimension(dimension, "dimension")
            if checked_dim != self.dimension:
                raise QdrantConfigMismatchError(
                    f"Requested dimension {checked_dim} does not match backend dimension {self.dimension}",
                    reason="dimension_mismatch",
                )
        if model_id is not None:
            checked_model = _validate_id(model_id, "model_id")
            if checked_model != self.model_id:
                raise QdrantConfigMismatchError(
                    "Requested model_id does not match backend model_id",
                    reason="model_id_mismatch",
                )

        if self._collection_ensured:
            return

        collection_path = f"/collections/{self.collection_name}"
        status, body = self._request_json("GET", collection_path, None)

        if 200 <= status < 300:
            self._validate_existing_collection_body(body)
            self._collection_ensured = True
            return

        if status == 404:
            create_payload = {
                "vectors": {
                    "size": self.dimension,
                    "distance": "Cosine",
                }
            }
            put_status, put_body = self._request_json(
                "PUT", collection_path, create_payload
            )
            self._validate_mutation_response(
                put_status, put_body, "collection creation"
            )
            self._collection_ensured = True
            return

        raise QdrantBackendError(
            f"Unexpected HTTP status {status} when checking Qdrant collection",
            reason=f"http_status_{status}",
        )

    def upsert(
        self, memory_id: str, vector: Sequence[float], payload: dict[str, Any]
    ) -> None:
        """Validate and upsert a single memory vector point with content-free metadata."""
        valid_id = _validate_uuid(memory_id, "memory_id")
        valid_vector = _validate_vector(vector, self.dimension)
        valid_payload = _validate_index_payload(
            payload,
            expected_memory_id=valid_id,
            expected_model_id=self.model_id,
        )

        if not self._collection_ensured:
            self.ensure_collection(self.dimension, self.model_id)

        path = f"/collections/{self.collection_name}/points?wait=true"
        request_body = {
            "points": [
                {
                    "id": valid_id,
                    "vector": valid_vector,
                    "payload": valid_payload,
                }
            ]
        }
        status, resp_body = self._request_json("PUT", path, request_body)
        self._validate_mutation_response(status, resp_body, "point upsert")

    def delete(self, memory_id: str) -> None:
        """Delete a single memory point by canonical UUID with wait=true.

        If point delete returns HTTP 404, issues `GET /collections/{collection_name}`
        on the exact collection and confirms deletion success only when that
        collection GET is also a valid 404 response. Does not create the collection
        just to delete.
        """
        valid_id = _validate_uuid(memory_id, "memory_id")
        path = f"/collections/{self.collection_name}/points/delete?wait=true"
        request_body = {"points": [valid_id]}
        status, resp_body = self._request_json("POST", path, request_body)
        if status == 404:
            collection_path = f"/collections/{self.collection_name}"
            col_status, col_body = self._request_json("GET", collection_path, None)
            if (
                col_status == 404
                and isinstance(resp_body, dict)
                and resp_body.get("status") != "ok"
                and isinstance(col_body, dict)
                and col_body.get("status") != "ok"
            ):
                self._collection_ensured = False
                return
            raise QdrantBackendError(
                f"Qdrant point delete returned 404, but collection GET returned HTTP {col_status}",
                reason="delete_404_collection_not_missing",
            )
        self._validate_mutation_response(status, resp_body, "point delete")
