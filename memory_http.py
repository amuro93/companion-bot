"""Isolated loopback HTTP adapter; credentials are explicit test/service configuration."""
from __future__ import annotations

import hmac
import json
import math
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Mapping

from memory_core import MAX_QUERY_LENGTH, VALID_DOMAINS, _validate_id, _validate_limit

MAX_BODY_BYTES = 4096


@dataclass(frozen=True)
class Scope:
    owner_id: str
    character_id: str
    enabled: bool = True


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate field")
        result[key] = value
    return result


def create_handler(searcher, credentials: Mapping[str, Scope]):
    """Create a handler with a frozen token-to-scope snapshot; never accept identity from HTTP."""
    if not credentials or not isinstance(credentials, Mapping):
        raise ValueError("credentials required")
    snapshot = []
    scopes = set()
    for token, scope in credentials.items():
        if not isinstance(token, str) or len(token) < 32 or not token.isascii() or not token.isprintable() or any(c.isspace() for c in token):
            raise ValueError("invalid token")
        if not isinstance(scope, Scope) or type(scope.enabled) is not bool:
            raise ValueError("invalid scope")
        _validate_id(scope.owner_id, "owner_id")
        _validate_id(scope.character_id, "character_id")
        pair = (scope.owner_id, scope.character_id)
        if pair in scopes:
            raise ValueError("duplicate scope")
        scopes.add(pair)
        snapshot.append((token, scope))
    if not callable(getattr(searcher, "search", None)):
        raise ValueError("search dependency required")

    class MemoryHandler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Never log query, authorization, or URL.

        def _reply(self, status, payload):
            data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            if self.path != "/memory/search":
                return self._reply(404, {"error": "not found"})
            auth = self.headers.get_all("Authorization", [])
            supplied = auth[0][7:] if len(auth) == 1 and auth[0].startswith("Bearer ") else ""
            if not supplied or len(supplied) > 1024:
                return self._reply(401, {"error": "unauthorized"})
            scope = None
            for token, candidate in snapshot:
                if hmac.compare_digest(supplied, token):
                    scope = candidate
            if scope is None:
                return self._reply(401, {"error": "unauthorized"})
            if not scope.enabled:
                return self._reply(403, {"error": "forbidden"})
            if self.headers.get_content_type() != "application/json" or len(self.headers.get_all("Content-Type", [])) != 1:
                return self._reply(415, {"error": "unsupported media type"})
            lengths = self.headers.get_all("Content-Length", [])
            if len(lengths) != 1 or not lengths[0].isdigit():
                return self._reply(400, {"error": "invalid request"})
            length = int(lengths[0])
            if length < 1 or length > MAX_BODY_BYTES or self.headers.get("Transfer-Encoding"):
                return self._reply(413, {"error": "invalid body size"})
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8"), object_pairs_hook=_unique_pairs,
                                  parse_constant=lambda _: (_ for _ in ()).throw(ValueError("invalid number")))
                if not isinstance(body, dict) or not set(body) <= {"query", "limit", "domain"}:
                    raise ValueError("invalid fields")
                query = body.get("query")
                if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_LENGTH:
                    raise ValueError("invalid query")
                limit = _validate_limit(body.get("limit", 5))
                domain = body.get("domain")
                if domain is not None and (not isinstance(domain, str) or domain not in VALID_DOMAINS):
                    raise ValueError("invalid domain")
            except (ValueError, UnicodeError):
                return self._reply(400, {"error": "invalid request"})
            try:
                hits = searcher.search(scope.owner_id, scope.character_id, query, limit, domain)
                if not isinstance(hits, list):
                    raise ValueError("invalid results")
                results = []
                for hit in hits:
                    if (not isinstance(hit, dict) or hit.get("owner_id") != scope.owner_id
                        or not isinstance(hit.get("visible_to"), list)
                        or scope.character_id not in hit["visible_to"]):
                        raise ValueError("search returned out-of-scope result")
                    result = {"id": hit["memory_id"], "text": hit["text"], "domain": hit["domain"],
                              "score": hit["score"], "source_ids": hit["source_ids"]}
                    if (not isinstance(result["id"], str) or not isinstance(result["text"], str)
                        or result["domain"] not in VALID_DOMAINS
                        or isinstance(result["score"], bool) or not isinstance(result["score"], (int, float))
                        or not math.isfinite(result["score"]) or not isinstance(result["source_ids"], list)):
                        raise ValueError("invalid result")
                    results.append(result)
                if len(results) > limit:
                    raise ValueError("too many results")
                return self._reply(200, {"results": results})
            except Exception:
                return self._reply(503, {"error": "search unavailable"})

        def do_GET(self):
            self._reply(404, {"error": "not found"})

    return MemoryHandler


def create_server(searcher, credentials: Mapping[str, Scope], host="127.0.0.1", port=0):
    """Bind loopback by default; a non-loopback interface requires an explicit host."""
    return ThreadingHTTPServer((host, port), create_handler(searcher, credentials))
