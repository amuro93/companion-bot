"""Real extraction and indexing callback for deferred companion memory worker (P2o-5)."""

from __future__ import annotations

from typing import Any

from memory_auto import record_user_message_keyed
from memory_core import _validate_id


class DeferredMemoryProcessorError(ValueError, TypeError):
    """Raised when processor configuration arguments are invalid."""


def _has_error_in_recorder_outcome(outcome: Any, max_nodes: int = 500) -> bool:
    """Iteratively check if recorder results or trace indicate an error."""
    if not isinstance(outcome, dict):
        return True
    if not isinstance(outcome.get("message_id"), str) or not isinstance(outcome.get("memory_ids"), list):
        return True
    results, trace = outcome.get("results"), outcome.get("trace")
    if not isinstance(results, list) or not isinstance(trace, list):
        return True

    stack = [results, trace]
    seen_ids: set[int] = set()
    visited = 0
    while stack:
        if visited >= max_nodes:
            return True
        visited += 1
        item = stack.pop()
        item_id = id(item)
        if item_id in seen_ids:
            continue
        seen_ids.add(item_id)

        if isinstance(item, dict):
            if item.get("action") == "error" or item.get("how") == "error":
                return True
            for val in item.values():
                if isinstance(val, (dict, list, tuple)):
                    stack.append(val)
        elif isinstance(item, (list, tuple)):
            for val in item:
                if isinstance(val, (dict, list, tuple)):
                    stack.append(val)
    return False


class DeferredMemoryProcessor:
    """Invokes canonical extraction and drains vector indexing for deferred messages."""

    def __init__(
        self,
        store: Any,
        owner_id: str,
        character_id: str,
        keyer: Any,
        indexer: Any,
        max_memories: int = 2,
        max_index_batches: int = 10,
    ) -> None:
        self._owner_id = _validate_id(owner_id, "owner_id")
        self._character_id = _validate_id(character_id, "character_id")

        if store is None or not callable(getattr(store, "pending_jobs", None)):
            raise DeferredMemoryProcessorError("store must provide a callable pending_jobs method")
        self._store = store

        if keyer is None or not callable(getattr(keyer, "decide", None)):
            raise DeferredMemoryProcessorError("keyer must provide a callable decide method")
        self._keyer = keyer

        if indexer is None or not callable(getattr(indexer, "run_once", None)):
            raise DeferredMemoryProcessorError("indexer must provide a callable run_once method")
        self._indexer = indexer

        if isinstance(max_memories, bool) or not isinstance(max_memories, int) or not (1 <= max_memories <= 5):
            raise DeferredMemoryProcessorError("max_memories must be an integer between 1 and 5")
        self._max_memories = max_memories

        if isinstance(max_index_batches, bool) or not isinstance(max_index_batches, int) or not (1 <= max_index_batches <= 100):
            raise DeferredMemoryProcessorError("max_index_batches must be an integer between 1 and 100")
        self._max_index_batches = max_index_batches

    def __call__(self, content: str) -> dict[str, Any]:
        """Execute canonical extraction followed by bounded index draining."""
        extraction_failed = False

        if not isinstance(content, str):
            extraction_failed = True
        else:
            try:
                outcome = record_user_message_keyed(
                    self._store,
                    self._owner_id,
                    self._character_id,
                    content,
                    max_memories=self._max_memories,
                    keyer=self._keyer,
                )
                if _has_error_in_recorder_outcome(outcome):
                    extraction_failed = True
            except Exception:
                extraction_failed = True

        index_failed = False
        indexing_confirmed = False
        batches = 0

        while batches < self._max_index_batches:
            batches += 1
            try:
                report = self._indexer.run_once(self._owner_id, self._character_id, limit=100)
            except Exception:
                index_failed = True
                break

            if not isinstance(report, dict) or report.get("status") != "ok":
                index_failed = True
                break

            if report.get("owner_id") != self._owner_id or report.get("character_id") != self._character_id:
                index_failed = True
                break

            failed_count = report.get("failed_count")
            if isinstance(failed_count, bool) or not isinstance(failed_count, int) or failed_count != 0:
                index_failed = True
                break

            if report.get("pending_count_exact") is not True or report.get("pending_count_status") != "exact":
                index_failed = True
                break

            pending_count = report.get("pending_count")
            if isinstance(pending_count, bool) or not isinstance(pending_count, int) or pending_count < 0:
                index_failed = True
                break

            if pending_count == 0:
                try:
                    remaining = self._store.pending_jobs(self._owner_id, self._character_id, limit=1)
                except Exception:
                    index_failed = True
                    break

                if not isinstance(remaining, list):
                    index_failed = True
                    break

                if len(remaining) == 0:
                    indexing_confirmed = True
                    break

        if not indexing_confirmed:
            index_failed = True

        if index_failed:
            return {"status": "failed", "error_category": "index_failed"}
        if extraction_failed:
            return {"status": "failed", "error_category": "processor_failed"}
        return {"status": "ok"}
