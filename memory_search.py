"""Fail-closed, SQLite-authorized semantic search over a model-bound Qdrant index."""
from __future__ import annotations

import math
from typing import Any

from memory_core import MAX_QUERY_LENGTH, VALID_DOMAINS, _validate_id, _validate_limit, _validate_text
from memory_index import QdrantHTTPBackend, QdrantBackendError, _validate_uuid, _validate_vector


class SemanticSearchError(Exception):
    """Sanitized search failure; no query, credentials, or backend response included."""


class SemanticSearcher:
    """Caller must supply authenticated owner and reader character IDs."""

    def __init__(self, store: Any, embedder: Any, backend: QdrantHTTPBackend) -> None:
        if not callable(getattr(store, "get_memory", None)):
            raise ValueError("store must provide get_memory")
        self.store, self.embedder, self.backend = store, embedder, backend

    def search(self, owner_id: str, character_id: str, query: str,
               limit: int = 5, domain: str | None = None) -> list[dict[str, Any]]:
        owner_id = _validate_id(owner_id, "owner_id")
        character_id = _validate_id(character_id, "character_id")
        query = _validate_text(query, "query", MAX_QUERY_LENGTH)
        limit = _validate_limit(limit)
        if domain is not None and domain not in VALID_DOMAINS:
            raise ValueError("invalid domain")
        if self.embedder is None or self.backend is None:
            raise SemanticSearchError("search dependency unavailable")
        if not callable(getattr(self.embedder, "embed", None)):
            raise SemanticSearchError("embedder unavailable")
        try:
            if self.embedder.model_id != self.backend.model_id or self.embedder.dimension != self.backend.dimension:
                raise SemanticSearchError("embedding configuration mismatch")
            vector = _validate_vector(self.embedder.embed(query), self.backend.dimension)
        except SemanticSearchError:
            raise
        except Exception:
            raise SemanticSearchError("embedding failed") from None

        # Bounded overfetch to compensate for stale index entries without unbounded reads.
        candidate_limit = min(1000, max(limit, limit * 4))
        must = [
            {"key": "owner_id", "match": {"value": owner_id}},
            {"key": "visible_to", "match": {"value": character_id}},
            {"key": "embedding_model_id", "match": {"value": self.backend.model_id}},
        ]
        if domain is not None:
            must.append({"key": "domain", "match": {"value": domain}})
        request = {
            "query": vector, "filter": {"must": must}, "limit": candidate_limit,
            "with_payload": ["memory_id", "owner_id", "creator_character_id", "visible_to", "domain", "embedding_model_id"],
            "with_vector": False,
        }
        try:
            status, body = self.backend._request_json(
                "POST", f"/collections/{self.backend.collection_name}/points/query", request
            )
            if not 200 <= status < 300 or not isinstance(body, dict) or body.get("status") != "ok" or body.get("error") is not None:
                raise SemanticSearchError("vector search failed")
            result = body.get("result")
            if not isinstance(result, dict) or not isinstance(result.get("points"), list) or len(result["points"]) > candidate_limit:
                raise SemanticSearchError("invalid vector response")
            points = result["points"]
            # Validate the entire response before returning any canonical data.
            for point in points:
                if not isinstance(point, dict) or not isinstance(point.get("payload"), dict):
                    raise SemanticSearchError("invalid vector response")
                _validate_uuid(point.get("id"))
                score = point.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                    raise SemanticSearchError("invalid vector response")
        except SemanticSearchError:
            raise
        except Exception:
            raise SemanticSearchError("vector search failed") from None

        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        try:
            for point in points:
                mid, payload = point["id"], point["payload"]
                if mid in seen:
                    continue
                seen.add(mid)
                visible = payload.get("visible_to")
                if (payload.get("memory_id") != mid or payload.get("owner_id") != owner_id
                    or payload.get("embedding_model_id") != self.backend.model_id
                    or not isinstance(visible, list) or character_id not in visible
                    or payload.get("domain") not in VALID_DOMAINS
                    or (domain is not None and payload.get("domain") != domain)):
                    continue
                record = self.store.get_memory(owner_id, character_id, mid)
                if (record is None or record.get("memory_id") != mid or record.get("owner_id") != owner_id
                    or record.get("domain") != payload["domain"]
                    or character_id not in record.get("visible_to", [])
                    or record.get("creator_character_id") != payload.get("creator_character_id")):
                    continue
                records.append({**record, "score": float(point["score"])})
                if len(records) == limit:
                    break
        except Exception:
            raise SemanticSearchError("canonical search verification failed") from None
        return records


def semantic_search(store: Any, embedder: Any, backend: QdrantHTTPBackend,
                    owner_id: str, character_id: str, query: str,
                    limit: int = 5, domain: str | None = None) -> list[dict[str, Any]]:
    return SemanticSearcher(store, embedder, backend).search(owner_id, character_id, query, limit, domain)
