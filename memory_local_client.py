"""In-process memory client replacing loopback HTTP server with identical auth boundaries."""
from __future__ import annotations

import hmac
import math
from typing import Any, Dict, List, Mapping, Optional

from companion_chat import MemoryUnavailable
from memory_core import (
    MAX_QUERY_LENGTH,
    VALID_DOMAINS,
    _validate_id,
    _validate_limit,
    _validate_text,
)
from memory_http import Scope


class InProcessMemoryClient:
    """Authenticates callers against an immutable token->Scope snapshot in-process."""

    def __init__(self, searcher: Any, credentials: Mapping[str, Scope]) -> None:
        if not credentials or not isinstance(credentials, Mapping):
            raise ValueError("credentials required")
        snapshot = []
        scopes = set()
        for token, scope in credentials.items():
            if (not isinstance(token, str) or len(token) < 32 or not token.isascii()
                    or not token.isprintable() or any(c.isspace() for c in token)):
                raise ValueError("invalid token")
            if not isinstance(scope, Scope) or type(scope.enabled) is not bool:
                raise ValueError("invalid scope")
            _validate_id(scope.owner_id, "owner_id")
            _validate_id(scope.character_id, "character_id")
            pair = (scope.owner_id, scope.character_id)
            if pair in scopes:
                raise ValueError("duplicate scope")
            scopes.add(pair)
            snapshot.append((token, Scope(owner_id=scope.owner_id, character_id=scope.character_id, enabled=scope.enabled)))
        if not callable(getattr(searcher, "search", None)):
            raise ValueError("search dependency required")
        self._searcher = searcher
        self._snapshot = tuple(snapshot)

    def search(
        self,
        token: str,
        query: str,
        limit: int = 5,
        domain: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Authenticate token and perform bounded search against injected searcher."""
        # Authenticate token safely against snapshot
        if (not isinstance(token, str) or len(token) < 32 or len(token) > 1024
                or not token.isascii() or not token.isprintable() or any(c.isspace() for c in token)):
            raise MemoryUnavailable("memory search unavailable")

        matched_scope: Optional[Scope] = None
        for snap_token, candidate in self._snapshot:
            if hmac.compare_digest(token, snap_token):
                matched_scope = candidate

        if matched_scope is None or not matched_scope.enabled:
            raise MemoryUnavailable("memory search unavailable")

        # Validate query, limit, domain
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH or "\x00" in query:
            raise MemoryUnavailable("memory search unavailable")

        if isinstance(limit, bool) or not isinstance(limit, int):
            raise MemoryUnavailable("memory search unavailable")
        try:
            _validate_limit(limit)
        except Exception:
            raise MemoryUnavailable("memory search unavailable") from None

        if domain is not None and (not isinstance(domain, str) or domain not in VALID_DOMAINS):
            raise MemoryUnavailable("memory search unavailable")

        # Execute searcher call safely without leaking query or credentials in errors
        try:
            raw_hits = self._searcher.search(
                matched_scope.owner_id,
                matched_scope.character_id,
                query,
                limit,
                domain,
            )
        except Exception:
            raise MemoryUnavailable("memory search unavailable") from None

        if not isinstance(raw_hits, list) or len(raw_hits) > limit:
            raise MemoryUnavailable("memory search unavailable")

        # Validate all records before returning any
        results = []
        try:
            for hit in raw_hits:
                if not isinstance(hit, dict):
                    raise MemoryUnavailable("memory search unavailable")

                # Owner isolation
                if hit.get("owner_id") != matched_scope.owner_id:
                    raise MemoryUnavailable("memory search unavailable")

                # Character visibility
                visible_to = hit.get("visible_to")
                if not isinstance(visible_to, list) or matched_scope.character_id not in visible_to:
                    raise MemoryUnavailable("memory search unavailable")

                # ID validation
                mem_id = hit.get("memory_id")
                _validate_id(mem_id, "memory_id")

                # Text validation
                text = hit.get("text")
                _validate_text(text, "text")

                # Domain validation
                hit_domain = hit.get("domain")
                if not isinstance(hit_domain, str) or hit_domain not in VALID_DOMAINS:
                    raise MemoryUnavailable("memory search unavailable")
                if domain is not None and hit_domain != domain:
                    raise MemoryUnavailable("memory search unavailable")

                # Score validation
                score = hit.get("score")
                if isinstance(score, bool) or not isinstance(score, (int, float)):
                    raise MemoryUnavailable("memory search unavailable")
                try:
                    if not math.isfinite(score):
                        raise MemoryUnavailable("memory search unavailable")
                except OverflowError:
                    raise MemoryUnavailable("memory search unavailable") from None

                # Source IDs validation
                source_ids = hit.get("source_ids")
                if not isinstance(source_ids, list) or not source_ids:
                    raise MemoryUnavailable("memory search unavailable")
                clean_sources = []
                for sid in source_ids:
                    _validate_id(sid, "source_id")
                    clean_sources.append(sid)

                results.append({
                    "id": mem_id,
                    "text": text,
                    "domain": hit_domain,
                    "score": score,
                    "source_ids": clean_sources,
                })
        except MemoryUnavailable:
            raise
        except Exception:
            raise MemoryUnavailable("memory search unavailable") from None

        return results
