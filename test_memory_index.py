"""Unit tests for P1b scoped index-job processor and optional Qdrant HTTP adapter.

Uses standard-library unittest with honest fake embedders, fake vector backends,
and injected HTTP transport callables. Performs raw SQLite assertions, closes all
SQLite connections explicitly, and confines temporary directories to environment
TMPDIR with safe fallback to workspace `test-temp`.
"""

from __future__ import annotations

import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Any, Callable, Sequence
import unittest
import urllib.error
import uuid
from memory_core import MAX_LIMIT

from memory_core import MemoryStore
from memory_index import (
    ALLOWED_PAYLOAD_KEYS,
    Embedder,
    IndexWorker,
    QdrantBackendError,
    QdrantConfigMismatchError,
    QdrantHTTPBackend,
    VectorBackend,
    derive_collection_name,
)

_WORKSPACE_ROOT = Path(__file__).resolve().parent


def _resolve_temp_base_dir() -> tuple[Path, bool]:
    """Return `(base_dir, created_fallback_dir)` under env TMPDIR or workspace test-temp."""
    env_tmpdir = os.environ.get("TMPDIR", "").strip()
    if env_tmpdir:
        base = Path(env_tmpdir).resolve()
        base.mkdir(parents=True, exist_ok=True)
        return base, False
    fallback = _WORKSPACE_ROOT / "test-temp"
    existed = fallback.exists()
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback, not existed


class FakeEmbedder:
    """Deterministic fake embedder for offline unit testing."""

    def __init__(
        self,
        model_id: str = "fake-embed-v1",
        dimension: int = 4,
        *,
        override_vector: Sequence[Any] | None = None,
        fail_with: Exception | None = None,
    ) -> None:
        self.model_id = model_id
        self.dimension = dimension
        self.override_vector = override_vector
        self.fail_with = fail_with
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        if self.fail_with is not None:
            raise self.fail_with
        if self.override_vector is not None:
            return list(self.override_vector)  # type: ignore[arg-type]

        digest = hashlib.sha256(f"{self.model_id}:{text}".encode("utf-8")).digest()
        return [
            round(((digest[i % len(digest)] / 255.0) * 2.0) - 1.0, 6)
            for i in range(self.dimension)
        ]


class FakeVectorBackend:
    """In-memory fake vector backend recording operations and supporting hooks."""

    def __init__(
        self,
        *,
        dimension: int | None = None,
        model_id: str | None = None,
        fail_on_ensure: Exception | None = None,
        fail_on_upsert_ids: set[str] | None = None,
        fail_on_delete_ids: set[str] | None = None,
        upsert_error: Exception | None = None,
        delete_error: Exception | None = None,
        on_upsert_hook: Callable[[str, list[float], dict[str, Any]], None] | None = None,
    ) -> None:
        self.dimension = dimension
        self.model_id = model_id
        self.fail_on_ensure = fail_on_ensure
        self.fail_on_upsert_ids = set(fail_on_upsert_ids or ())
        self.fail_on_delete_ids = set(fail_on_delete_ids or ())
        self.upsert_error = upsert_error or RuntimeError(
            "Backend upsert failed: secret_token=tok_live_999"
        )
        self.delete_error = delete_error or RuntimeError(
            "Backend delete failed: secret_token=tok_live_999"
        )
        self.on_upsert_hook = on_upsert_hook

        self.ensured: list[tuple[int, str]] = []
        self.points: dict[str, dict[str, Any]] = {}
        self.operations: list[tuple[str, str]] = []

    def ensure_collection(self, dimension: int, model_id: str) -> None:
        if self.fail_on_ensure is not None:
            raise self.fail_on_ensure
        if self.dimension is not None and dimension != self.dimension:
            raise ValueError(
                f"Backend dimension mismatch: {dimension} != {self.dimension}"
            )
        if self.model_id is not None and model_id != self.model_id:
            raise ValueError(
                f"Backend model_id mismatch: {model_id} != {self.model_id}"
            )
        self.ensured.append((dimension, model_id))

    def upsert(
        self, memory_id: str, vector: list[float], payload: dict[str, Any]
    ) -> None:
        if memory_id in self.fail_on_upsert_ids:
            raise self.upsert_error
        self.points[memory_id] = {
            "id": memory_id,
            "vector": list(vector),
            "payload": dict(payload),
        }
        self.operations.append(("upsert", memory_id))
        if self.on_upsert_hook is not None:
            self.on_upsert_hook(memory_id, vector, payload)

    def delete(self, memory_id: str) -> None:
        if memory_id in self.fail_on_delete_ids:
            raise self.delete_error
        self.points.pop(memory_id, None)
        self.operations.append(("delete", memory_id))


class FakeHTTPTransport:
    """Programmable HTTP transport callable capturing requests for QdrantHTTPBackend tests."""

    def __init__(
        self,
        responder: Callable[
            [str, str, dict[str, Any] | None, dict[str, str], float],
            tuple[int, Any],
        ],
    ) -> None:
        self._responder = responder
        self.requests: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        body: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> tuple[int, Any]:
        parsed_body: dict[str, Any] | None = None
        raw_text: str | None = None
        if body is not None:
            raw_text = body.decode("utf-8")
            parsed_body = json.loads(raw_text)

        record = {
            "method": method,
            "url": url,
            "raw_body": raw_text,
            "json": parsed_body,
            "headers": dict(headers),
            "timeout": timeout,
        }
        self.requests.append(record)
        return self._responder(method, url, parsed_body, dict(headers), timeout)


class TestMemoryIndex(unittest.TestCase):
    """Verification suite for P1b indexing helpers, IndexWorker, and QdrantHTTPBackend."""

    def setUp(self) -> None:
        self._temp_base, self._created_fallback_base = _resolve_temp_base_dir()
        self._tmpdir = tempfile.TemporaryDirectory(dir=str(self._temp_base))
        self.db_path = Path(self._tmpdir.name) / "companion_memory_p1b.db"
        self.store = MemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self._tmpdir.cleanup()
        if self._created_fallback_base and self._temp_base.exists():
            try:
                next(self._temp_base.iterdir())
            except StopIteration:
                try:
                    self._temp_base.rmdir()
                except OSError:
                    pass

    def _raw_job_rows(
        self, owner_id: str | None = None, character_id: str | None = None
    ) -> list[dict[str, Any]]:
        conn = sqlite3.connect(self.db_path)
        try:
            conn.row_factory = sqlite3.Row
            if owner_id is not None and character_id is not None:
                rows = conn.execute(
                    """
                    SELECT seq, job_id, memory_id, action, owner_id, character_id, created_at
                    FROM index_jobs
                    WHERE owner_id = ? AND character_id = ?
                    ORDER BY seq ASC
                    """,
                    (owner_id, character_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT seq, job_id, memory_id, action, owner_id, character_id, created_at
                    FROM index_jobs
                    ORDER BY seq ASC
                    """
                ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def _assert_no_private_strings(
        self, obj: Any, forbidden_substrings: Sequence[str]
    ) -> None:
        serialized = json.dumps(obj, ensure_ascii=False, sort_keys=True)
        for secret in forbidden_substrings:
            self.assertNotIn(secret, serialized)

    def test_pending_report_bounded_and_failed_recheck(self) -> None:
        class PendingStore:
            def __init__(self, fail: bool = False) -> None:
                self.fail = fail
                self.calls = 0

            def pending_jobs(self, owner_id, character_id, limit=100):
                self.calls += 1
                if self.fail and self.calls > 1:
                    raise OSError("private detail")
                count = 2 if self.fail else limit
                return [{"job_id": str(i), "memory_id": str(uuid.uuid4()),
                         "action": "delete"} for i in range(count)]

            def get_index_record(self, *args):
                return None

            def ack_index_job(self, *args):
                return False

        for fail in (False, True):
            report = IndexWorker(PendingStore(fail), None, FakeVectorBackend()).run_once(
                "owner", "char", limit=1
            )
            self.assertEqual(report["status"], "failed")
            self.assertIsNone(report["pending_count"])
            self.assertFalse(report["pending_count_exact"])
            self.assertEqual(report["pending_count_lower_bound"], 0 if fail else MAX_LIMIT)
            self.assertEqual(report["pending_count_status"], "unknown" if fail else "lower_bound")

    def test_default_transport_rejects_redirect_with_and_without_key(self) -> None:
        seen = []

        class Destination(BaseHTTPRequestHandler):
            def do_GET(self):
                seen.append(dict(self.headers))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"status":"ok"}')

            def log_message(self, *args):
                pass

        destination = ThreadingHTTPServer(("127.0.0.1", 0), Destination)

        class Redirect(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(302)
                self.send_header("Location", f"http://127.0.0.1:{destination.server_port}/stolen")
                self.end_headers()

            def log_message(self, *args):
                pass

        redirect = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        from threading import Thread
        threads = [Thread(target=server.serve_forever, daemon=True)
                   for server in (destination, redirect)]
        for thread in threads:
            thread.start()
        try:
            for key in (None, "test-secret-not-real"):
                backend = QdrantHTTPBackend(
                    f"http://127.0.0.1:{redirect.server_port}", "mem", 4, "model", api_key=key
                )
                with self.assertRaises(QdrantBackendError):
                    backend.ensure_collection()
            self.assertEqual(seen, [])
        finally:
            for server in (redirect, destination):
                server.shutdown()
                server.server_close()
            for thread in threads:
                thread.join()

    def test_protocols_and_core_index_helpers(self) -> None:
        embedder = FakeEmbedder()
        backend = FakeVectorBackend()
        self.assertIsInstance(embedder, Embedder)
        self.assertIsInstance(backend, VectorBackend)

        s1 = self.store.append_message(
            "owner_1", "char_a", "m1", "user", "Source message 1"
        )
        mem_id = self.store.save_memory(
            owner_id="owner_1",
            character_id="char_a",
            external_id="mem_1",
            kind="preference",
            text="Creator canonical memory text",
            source_ids=[s1],
            domain="roleplay",
            visible_to=["char_c", "char_b"],
        )

        # Creator can fetch index record with sorted visible_to, domain, and text
        rec = self.store.get_index_record("owner_1", "char_a", mem_id)
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["memory_id"], mem_id)
        self.assertEqual(rec["owner_id"], "owner_1")
        self.assertEqual(rec["creator_character_id"], "char_a")
        self.assertEqual(rec["domain"], "roleplay")
        self.assertEqual(rec["text"], "Creator canonical memory text")
        self.assertEqual(rec["visible_to"], ["char_a", "char_b", "char_c"])

        # Shared reader char_b CAN read via public get_memory, but NOT via get_index_record
        self.assertIsNotNone(self.store.get_memory("owner_1", "char_b", mem_id))
        self.assertIsNone(self.store.get_index_record("owner_1", "char_b", mem_id))
        self.assertIsNone(self.store.get_index_record("owner_2", "char_a", mem_id))
        self.assertIsNone(
            self.store.get_index_record("owner_1", "char_a", str(uuid.uuid4()))
        )

        # Scoped ack_index_job only deletes exact matching (owner_id, character_id, job_id)
        jobs = self.store.pending_jobs("owner_1", "char_a")
        self.assertEqual(len(jobs), 1)
        job_id = jobs[0]["job_id"]

        self.assertFalse(self.store.ack_index_job("owner_1", "char_b", job_id))
        self.assertFalse(self.store.ack_index_job("owner_2", "char_a", job_id))
        self.assertFalse(
            self.store.ack_index_job("owner_1", "char_a", str(uuid.uuid4()))
        )
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        self.assertTrue(self.store.ack_index_job("owner_1", "char_a", job_id))
        self.assertFalse(self.store.ack_index_job("owner_1", "char_a", job_id))
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

    def test_scoped_indexing_ack_and_raw_sqlite_verification(self) -> None:
        secret_1a = "PRIVATE_TEXT_OWNER1_CHAR_A_DO_NOT_LEAK"
        secret_1b = "PRIVATE_TEXT_OWNER1_CHAR_B_DO_NOT_LEAK"
        secret_2a = "PRIVATE_TEXT_OWNER2_CHAR_A_DO_NOT_LEAK"

        s_1a = self.store.append_message("owner_1", "char_a", "m1", "user", "msg 1a")
        s_1b = self.store.append_message("owner_1", "char_b", "m1", "user", "msg 1b")
        s_2a = self.store.append_message("owner_2", "char_a", "m1", "user", "msg 2a")

        m_1a = self.store.save_memory(
            "owner_1", "char_a", "e1", "fact", secret_1a, [s_1a]
        )
        m_1b = self.store.save_memory(
            "owner_1", "char_b", "e1", "fact", secret_1b, [s_1b]
        )
        m_2a = self.store.save_memory(
            "owner_2", "char_a", "e1", "fact", secret_2a, [s_2a]
        )

        self.assertEqual(len(self._raw_job_rows()), 3)

        embedder = FakeEmbedder(model_id="embed-model-alpha", dimension=4)
        backend = FakeVectorBackend(dimension=4, model_id="embed-model-alpha")
        worker = IndexWorker(self.store, embedder, backend)

        report = worker.run_once("owner_1", "char_a")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["processed_count"], 1)
        self.assertEqual(report["upserted_count"], 1)
        self.assertEqual(report["deleted_count"], 0)
        self.assertEqual(report["failed_count"], 0)
        self.assertIsNone(report["failed"])
        self.assertEqual(report["pending_count"], 0)
        self._assert_no_private_strings(
            report, [secret_1a, secret_1b, secret_2a, "msg 1a"]
        )

        # Only m_1a was indexed and ACKed; m_1b and m_2a remain untouched in SQLite
        self.assertEqual(set(backend.points.keys()), {m_1a})
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_b")), 1)
        self.assertEqual(len(self._raw_job_rows("owner_2", "char_a")), 1)
        self.assertEqual(
            self._raw_job_rows("owner_1", "char_b")[0]["memory_id"], m_1b
        )
        self.assertEqual(
            self._raw_job_rows("owner_2", "char_a")[0]["memory_id"], m_2a
        )

    def test_shared_acl_payload_and_owner_separation(self) -> None:
        secret_text = "SHARED_SECRET_MEMORY_TEXT_NEVER_IN_PAYLOAD"
        s1 = self.store.append_message(
            "owner_alpha", "char_creator", "m1", "user", "Original source content"
        )
        s2 = self.store.append_message(
            "owner_beta", "char_creator", "m1", "user", "Other owner source"
        )

        mem_alpha = self.store.save_memory(
            owner_id="owner_alpha",
            character_id="char_creator",
            external_id="ext_shared",
            kind="promise",
            text=secret_text,
            source_ids=[s1],
            domain="roleplay",
            visible_to=["char_z", "char_b"],
        )
        mem_beta = self.store.save_memory(
            owner_id="owner_beta",
            character_id="char_creator",
            external_id="ext_shared",
            kind="fact",
            text="Beta owner private fact",
            source_ids=[s2],
            domain="real",
            visible_to=None,
        )

        embedder = FakeEmbedder(model_id="bge-small-test", dimension=4)
        backend = FakeVectorBackend()
        worker = IndexWorker(self.store, embedder, backend)

        worker.run_once("owner_alpha", "char_creator")
        worker.run_once("owner_beta", "char_creator")

        self.assertIn(mem_alpha, backend.points)
        self.assertIn(mem_beta, backend.points)

        alpha_payload = backend.points[mem_alpha]["payload"]
        beta_payload = backend.points[mem_beta]["payload"]

        self.assertEqual(set(alpha_payload.keys()), set(ALLOWED_PAYLOAD_KEYS))
        self.assertEqual(
            alpha_payload,
            {
                "memory_id": mem_alpha,
                "owner_id": "owner_alpha",
                "creator_character_id": "char_creator",
                "visible_to": ["char_b", "char_creator", "char_z"],
                "domain": "roleplay",
                "embedding_model_id": "bge-small-test",
            },
        )
        self.assertEqual(
            beta_payload,
            {
                "memory_id": mem_beta,
                "owner_id": "owner_beta",
                "creator_character_id": "char_creator",
                "visible_to": ["char_creator"],
                "domain": "real",
                "embedding_model_id": "bge-small-test",
            },
        )
        self._assert_no_private_strings(
            backend.points,
            [secret_text, "Beta owner private fact", "Original source content", s1, s2],
        )

    def test_model_and_dimension_mismatch_retains_job(self) -> None:
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "source")
        mem_id = self.store.save_memory(
            "owner_1", "char_a", "e1", "fact", "Memory to test mismatch", [s1]
        )

        # 1. Embedder returns vector of wrong length (3 != 4)
        bad_len_embedder = FakeEmbedder(
            model_id="m-v1", dimension=4, override_vector=[0.1, 0.2, 0.3]
        )
        backend = FakeVectorBackend(dimension=4, model_id="m-v1")
        worker_1 = IndexWorker(self.store, bad_len_embedder, backend)
        res_1 = worker_1.run_once("owner_1", "char_a")
        self.assertEqual(res_1["status"], "failed")
        self.assertEqual(res_1["processed_count"], 0)
        self.assertEqual(res_1["failed_count"], 1)
        self.assertEqual(res_1["failed"]["error_category"], "validation_error")
        self.assertEqual(backend.points, {})
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # 2. Embedder dimension != Backend dimension (4 != 8)
        valid_embedder = FakeEmbedder(model_id="m-v1", dimension=4)
        wrong_dim_backend = FakeVectorBackend(dimension=8, model_id="m-v1")
        worker_2 = IndexWorker(self.store, valid_embedder, wrong_dim_backend)
        res_2 = worker_2.run_once("owner_1", "char_a")
        self.assertEqual(res_2["status"], "failed")
        self.assertEqual(res_2["failed"]["error_category"], "validation_error")
        self.assertEqual(wrong_dim_backend.points, {})
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # 3. Embedder model_id != Backend model_id
        wrong_model_backend = FakeVectorBackend(dimension=4, model_id="m-v2")
        worker_3 = IndexWorker(self.store, valid_embedder, wrong_model_backend)
        res_3 = worker_3.run_once("owner_1", "char_a")
        self.assertEqual(res_3["status"], "failed")
        self.assertEqual(res_3["failed"]["error_category"], "validation_error")
        self.assertEqual(wrong_model_backend.points, {})
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # 4. Invalid model metadata (empty model_id or boolean dimension)
        for bad_embedder in (
            FakeEmbedder(model_id="   ", dimension=4),
            FakeEmbedder(model_id="m-v1", dimension=True),  # type: ignore[arg-type]
            FakeEmbedder(model_id="m-v1", dimension=0),
        ):
            w = IndexWorker(self.store, bad_embedder, FakeVectorBackend())
            r = w.run_once("owner_1", "char_a")
            self.assertEqual(r["status"], "failed")
            self.assertEqual(r["failed"]["error_category"], "validation_error")
            self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # Finally succeed with matching embedder & backend
        good_worker = IndexWorker(self.store, valid_embedder, backend)
        res_ok = good_worker.run_once("owner_1", "char_a")
        self.assertEqual(res_ok["status"], "ok")
        self.assertEqual(res_ok["processed_count"], 1)
        self.assertIn(mem_id, backend.points)
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

    def test_nan_inf_and_boolean_vector_values_rejected(self) -> None:
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "source")
        self.store.save_memory(
            "owner_1", "char_a", "e1", "fact", "Numeric validation memory", [s1]
        )
        backend = FakeVectorBackend(dimension=4, model_id="m-v1")

        invalid_vectors: list[list[Any]] = [
            [0.1, float("nan"), 0.3, 0.4],
            [0.1, float("inf"), 0.3, 0.4],
            [0.1, float("-inf"), 0.3, 0.4],
            [True, 0.2, 0.3, 0.4],
            [False, 0.2, 0.3, 0.4],
            [0.1, "0.2", 0.3, 0.4],
        ]

        for bad_vec in invalid_vectors:
            embedder = FakeEmbedder(
                model_id="m-v1", dimension=4, override_vector=bad_vec
            )
            worker = IndexWorker(self.store, embedder, backend)
            report = worker.run_once("owner_1", "char_a")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["failed_count"], 1)
            self.assertEqual(report["failed"]["error_category"], "validation_error")
            self.assertEqual(backend.points, {})
            self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

    def test_embedding_failure_retains_job_and_sanitizes_report(self) -> None:
        secret_mem = "TOP_SECRET_USER_DIARY_ENTRY_999"
        secret_cred = "Bearer sk-secret-credential-xyz"
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "source")
        self.store.save_memory("owner_1", "char_a", "e1", "fact", secret_mem, [s1])

        failing_embedder = FakeEmbedder(
            model_id="m-v1",
            dimension=4,
            fail_with=RuntimeError(f"Failed embedding {secret_mem} with {secret_cred}"),
        )
        backend = FakeVectorBackend()
        worker = IndexWorker(self.store, failing_embedder, backend)

        report = worker.run_once("owner_1", "char_a")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["processed_count"], 0)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["pending_count"], 1)
        self.assertEqual(report["failed"]["error_category"], "embedding_error")
        self.assertEqual(report["failed"]["error_type"], "RuntimeError")
        self._assert_no_private_strings(report, [secret_mem, secret_cred])
        self.assertEqual(backend.points, {})
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

    def test_backend_failure_stops_batch_and_preserves_fifo_ordering(self) -> None:
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "source")
        m1 = self.store.save_memory("owner_1", "char_a", "e1", "fact", "First", [s1])
        m2 = self.store.save_memory("owner_1", "char_a", "e2", "fact", "Second", [s1])
        m3 = self.store.save_memory("owner_1", "char_a", "e3", "fact", "Third", [s1])

        embedder = FakeEmbedder(model_id="m-v1", dimension=4)
        backend = FakeVectorBackend(
            dimension=4,
            model_id="m-v1",
            fail_on_upsert_ids={m2},
            upsert_error=ConnectionError("Qdrant down at http://admin:pw123@internal:6333"),
        )
        worker = IndexWorker(self.store, embedder, backend)

        report = worker.run_once("owner_1", "char_a")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["processed_count"], 1)
        self.assertEqual(report["failed_count"], 1)
        self.assertEqual(report["pending_count"], 2)
        self.assertEqual(report["failed"]["memory_id"], m2)
        self.assertEqual(report["failed"]["error_category"], "backend_error")
        self.assertEqual(report["failed"]["error_type"], "ConnectionError")
        self._assert_no_private_strings(
            report, ["pw123", "admin", "First", "Second", "Third"]
        )

        # m1 succeeded and was ACKed; m2 failed and stopped batch so m3 was never attempted
        self.assertEqual(set(backend.points.keys()), {m1})
        self.assertEqual(embedder.calls, ["First", "Second"])
        remaining = self._raw_job_rows("owner_1", "char_a")
        self.assertEqual([r["memory_id"] for r in remaining], [m2, m3])

        # Clearing backend failure and re-running processes m2 and m3 in order
        backend.fail_on_upsert_ids.clear()
        report_2 = worker.run_once("owner_1", "char_a")
        self.assertEqual(report_2["status"], "ok")
        self.assertEqual(report_2["processed_count"], 2)
        self.assertEqual(report_2["pending_count"], 0)
        self.assertEqual(set(backend.points.keys()), {m1, m2, m3})
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

    def test_delete_jobs_succeed_with_failing_or_missing_embedder(self) -> None:
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "source")
        m1 = self.store.save_memory("owner_1", "char_a", "e1", "fact", "To delete 1", [s1])
        m2 = self.store.save_memory("owner_1", "char_a", "e2", "fact", "To delete 2", [s1])

        good_embedder = FakeEmbedder(model_id="m-v1", dimension=4)
        backend = FakeVectorBackend(dimension=4, model_id="m-v1")
        worker_init = IndexWorker(self.store, good_embedder, backend)
        worker_init.run_once("owner_1", "char_a")
        self.assertEqual(set(backend.points.keys()), {m1, m2})

        # Delete both memories in SQLite -> queues two delete jobs
        self.assertTrue(self.store.delete_memory("owner_1", "char_a", m1))
        self.assertTrue(self.store.delete_memory("owner_1", "char_a", m2))
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 2)

        # Case 1: Embedder always raises RuntimeError
        broken_embedder = FakeEmbedder(
            model_id="m-v1",
            dimension=4,
            fail_with=RuntimeError("Embedding service offline"),
        )
        worker_broken_emb = IndexWorker(self.store, broken_embedder, backend)
        res_1 = worker_broken_emb.run_once("owner_1", "char_a", limit=1)
        self.assertEqual(res_1["status"], "ok")
        self.assertEqual(res_1["deleted_count"], 1)
        self.assertEqual(broken_embedder.calls, [])
        self.assertNotIn(m1, backend.points)
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # Case 2: Embedder is None
        worker_no_emb = IndexWorker(self.store, None, backend)
        res_2 = worker_no_emb.run_once("owner_1", "char_a", limit=1)
        self.assertEqual(res_2["status"], "ok")
        self.assertEqual(res_2["deleted_count"], 1)
        self.assertEqual(backend.points, {})
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

    def test_crash_or_ack_failure_and_idempotent_replay(self) -> None:
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "source")
        m1 = self.store.save_memory(
            "owner_1", "char_a", "e1", "fact", "Idempotent replay memory", [s1]
        )

        embedder = FakeEmbedder(model_id="m-v1", dimension=4)
        backend = FakeVectorBackend(dimension=4, model_id="m-v1")
        worker = IndexWorker(self.store, embedder, backend)

        orig_ack = self.store.ack_index_job

        # Simulate ACK raising an operational error after remote upsert succeeded
        def failing_ack_raise(owner_id: str, character_id: str, job_id: str) -> bool:
            raise sqlite3.OperationalError("simulated disk full during ack")

        self.store.ack_index_job = failing_ack_raise  # type: ignore[method-assign]
        try:
            res_fail_1 = worker.run_once("owner_1", "char_a")
        finally:
            self.store.ack_index_job = orig_ack  # type: ignore[method-assign]

        self.assertEqual(res_fail_1["status"], "failed")
        self.assertEqual(res_fail_1["failed"]["error_category"], "ack_error")
        self.assertEqual(res_fail_1["failed"]["error_type"], "OperationalError")
        self._assert_no_private_strings(res_fail_1, ["simulated disk full"])
        # Remote point exists, but SQLite job is still pending
        self.assertIn(m1, backend.points)
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # Simulate ACK returning False
        self.store.ack_index_job = lambda o, c, j: False  # type: ignore[method-assign]
        try:
            res_fail_2 = worker.run_once("owner_1", "char_a")
        finally:
            self.store.ack_index_job = orig_ack  # type: ignore[method-assign]

        self.assertEqual(res_fail_2["status"], "failed")
        self.assertEqual(res_fail_2["failed"]["error_category"], "ack_error")
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # Replay with working ACK succeeds idempotently and removes the SQLite job
        res_ok = worker.run_once("owner_1", "char_a")
        self.assertEqual(res_ok["status"], "ok")
        self.assertEqual(res_ok["processed_count"], 1)
        self.assertEqual(res_ok["pending_count"], 0)
        self.assertEqual(set(backend.points.keys()), {m1})
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

    def test_stale_upsert_after_correction_and_deletion_never_resurrects(self) -> None:
        old_secret = "SUPERSEDED_OLD_TEXT_MUST_NEVER_BE_INDEXED"
        new_secret = "ACTIVE_CORRECTED_TEXT_V2"

        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "src1")
        s2 = self.store.append_message("owner_1", "char_a", "m2", "user", "src2")

        m1 = self.store.save_memory(
            "owner_1", "char_a", "e1", "preference", old_secret, [s1]
        )
        m2 = self.store.correct_memory(
            "owner_1", "char_a", m1, "e2", new_secret, [s1, s2]
        )

        # Queue has: upsert(m1), delete(m1), upsert(m2)
        raw_before = self._raw_job_rows("owner_1", "char_a")
        self.assertEqual(
            [(r["memory_id"], r["action"]) for r in raw_before],
            [(m1, "upsert"), (m1, "delete"), (m2, "upsert")],
        )

        embedder = FakeEmbedder(model_id="m-v1", dimension=4)
        backend = FakeVectorBackend(dimension=4, model_id="m-v1")
        worker = IndexWorker(self.store, embedder, backend)

        report = worker.run_once("owner_1", "char_a")
        self.assertEqual(report["status"], "ok")
        self.assertEqual(report["processed_count"], 3)
        # First job (stale upsert of m1) was converted to delete, second was delete(m1), third was upsert(m2)
        self.assertEqual(report["deleted_count"], 2)
        self.assertEqual(report["upserted_count"], 1)

        # old_secret was NEVER even passed to embedder, and m1 was never upserted to backend
        self.assertEqual(embedder.calls, [new_secret])
        self.assertEqual(
            backend.operations,
            [("delete", m1), ("delete", m1), ("upsert", m2)],
        )
        self.assertEqual(set(backend.points.keys()), {m2})
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

        # Also verify a manually leftover stale upsert job for a deleted memory converts to delete
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """
                INSERT INTO index_jobs (job_id, memory_id, action, owner_id, character_id, created_at)
                VALUES (?, ?, 'upsert', 'owner_1', 'char_a', '2026-10-19T00:00:00.000000Z')
                """,
                (str(uuid.uuid4()), m1),
            )
            conn.commit()
        finally:
            conn.close()

        report_stale = worker.run_once("owner_1", "char_a")
        self.assertEqual(report_stale["status"], "ok")
        self.assertEqual(report_stale["deleted_count"], 1)
        self.assertEqual(report_stale["upserted_count"], 0)
        self.assertNotIn(m1, backend.points)
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

    def test_mutation_injected_during_backend_upsert_followed_by_final_delete(
        self,
    ) -> None:
        # Subtest 1: delete_memory injected while backend.upsert is executing
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "src1")
        m1 = self.store.save_memory(
            "owner_1", "char_a", "e1", "fact", "Delete during upsert", [s1]
        )

        def delete_during_upsert(
            mem_id: str, vec: list[float], payload: dict[str, Any]
        ) -> None:
            if mem_id == m1:
                self.store.delete_memory("owner_1", "char_a", m1)

        embedder = FakeEmbedder(model_id="m-v1", dimension=4)
        backend = FakeVectorBackend(
            dimension=4, model_id="m-v1", on_upsert_hook=delete_during_upsert
        )
        worker = IndexWorker(self.store, embedder, backend)

        report_1 = worker.run_once("owner_1", "char_a")
        self.assertEqual(report_1["status"], "ok")
        # Post-upsert recheck saw m1 gone and immediately executed compensating delete
        self.assertEqual(backend.operations, [("upsert", m1), ("delete", m1)])
        self.assertNotIn(m1, backend.points)
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

        # Subtest 2: in-place canonical ACL/text mutation injected during backend.upsert,
        # preventing stale ACK, followed by delete_memory and final delete run_once
        m2 = self.store.save_memory(
            "owner_1",
            "char_a",
            "e2",
            "fact",
            "Initial text before concurrent edit",
            [self.store.append_message("owner_1", "char_a", "m2", "user", "src2")],
            visible_to=["char_b"],
        )

        mutated_once = False

        def mutate_acl_during_upsert(
            mem_id: str, vec: list[float], payload: dict[str, Any]
        ) -> None:
            nonlocal mutated_once
            if mem_id == m2 and not mutated_once:
                mutated_once = True
                conn = sqlite3.connect(self.db_path)
                try:
                    conn.execute(
                        "DELETE FROM memory_acl WHERE memory_id = ? AND character_id = 'char_b'",
                        (m2,),
                    )
                    conn.commit()
                finally:
                    conn.close()

        backend_2 = FakeVectorBackend(
            dimension=4, model_id="m-v1", on_upsert_hook=mutate_acl_during_upsert
        )
        worker_2 = IndexWorker(self.store, embedder, backend_2)

        # First run_once detects that canonical ACL changed during upsert and refuses to ACK
        report_2 = worker_2.run_once("owner_1", "char_a")
        self.assertEqual(report_2["status"], "failed")
        self.assertEqual(
            report_2["failed"]["error_category"], "concurrent_modification"
        )
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # Now delete the memory and run_once -> executes final delete and clears queue
        self.assertTrue(self.store.delete_memory("owner_1", "char_a", m2))
        report_3 = worker_2.run_once("owner_1", "char_a")
        self.assertEqual(report_3["status"], "ok")
        self.assertEqual(report_3["deleted_count"], 1)
        self.assertNotIn(m2, backend_2.points)
        self.assertEqual(
            backend_2.operations, [("upsert", m2), ("delete", m2)]
        )
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

        # Subtest 3: correct_memory injected during backend.upsert, followed by final delete
        s3 = self.store.append_message("owner_1", "char_a", "m3", "user", "src3")
        m3 = self.store.save_memory(
            "owner_1", "char_a", "e3", "fact", "V1 before concurrent correction", [s3]
        )
        corrected_ids: list[str] = []

        def correct_during_upsert(
            mem_id: str, vec: list[float], payload: dict[str, Any]
        ) -> None:
            if mem_id == m3 and not corrected_ids:
                m4_id = self.store.correct_memory(
                    "owner_1",
                    "char_a",
                    m3,
                    "e3_corr",
                    "V2 corrected during upsert",
                    [s3],
                )
                corrected_ids.append(m4_id)

        backend_3 = FakeVectorBackend(
            dimension=4, model_id="m-v1", on_upsert_hook=correct_during_upsert
        )
        worker_3 = IndexWorker(self.store, embedder, backend_3)

        # First run_once processes initial upsert(m3), detects m3 became superseded during upsert,
        # and immediately deletes m3 from backend
        report_4 = worker_3.run_once("owner_1", "char_a", limit=1)
        self.assertEqual(report_4["status"], "ok")
        self.assertNotIn(m3, backend_3.points)
        self.assertEqual(backend_3.operations, [("upsert", m3), ("delete", m3)])
        self.assertEqual(len(corrected_ids), 1)
        m4 = corrected_ids[0]

        # Delete the lineage and drain remaining jobs -> neither m3 nor m4 exists in backend
        self.assertTrue(self.store.delete_memory("owner_1", "char_a", m4))
        report_5 = worker_3.run_once("owner_1", "char_a")
        self.assertEqual(report_5["status"], "ok")
        self.assertEqual(backend_3.points, {})
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

    def test_tombstone_non_resurrection_and_original_message_retained(self) -> None:
        orig_msg = "KEEP_ORIGINAL_DIALOGUE_MESSAGE_FOREVER"
        derived_secret = "DERIVED_MEMORY_TO_BE_WIPED_FROM_SQLITE_AND_VECTOR_INDEX"

        s1 = self.store.append_message("owner_1", "char_a", "m_orig", "user", orig_msg)
        m1 = self.store.save_memory(
            "owner_1", "char_a", "ext_tomb", "fact", derived_secret, [s1]
        )

        embedder = FakeEmbedder(model_id="m-v1", dimension=4)
        backend = FakeVectorBackend(dimension=4, model_id="m-v1")
        worker = IndexWorker(self.store, embedder, backend)

        worker.run_once("owner_1", "char_a")
        self.assertIn(m1, backend.points)

        self.assertTrue(self.store.delete_memory("owner_1", "char_a", m1))
        del_report = worker.run_once("owner_1", "char_a")
        self.assertEqual(del_report["status"], "ok")
        self.assertEqual(del_report["deleted_count"], 1)
        self.assertNotIn(m1, backend.points)
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

        # Tombstone blocks replay of deleted external_id and queues no new jobs
        with self.assertRaises(ValueError):
            self.store.save_memory(
                "owner_1", "char_a", "ext_tomb", "fact", derived_secret, [s1]
            )
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

        # Verify original message is still in SQLite while derived secret is absent
        conn = sqlite3.connect(self.db_path)
        try:
            msg_row = conn.execute(
                "SELECT content FROM messages WHERE message_id = ?", (s1,)
            ).fetchone()
            self.assertIsNotNone(msg_row)
            self.assertEqual(msg_row[0], orig_msg)

            mem_count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
            self.assertEqual(mem_count, 0)
        finally:
            conn.close()

    def test_qdrant_http_backend_paths_methods_payload_and_no_private_text(
        self,
    ) -> None:
        secret_text = "ULTRA_PRIVATE_TEXT_MUST_NEVER_CROSS_HTTP"
        secret_api_key = "qdrant-secret-api-key-98765"

        s1 = self.store.append_message(
            "owner_1", "char_a", "m1", "user", "Source dialogue text"
        )
        m1 = self.store.save_memory(
            owner_id="owner_1",
            character_id="char_a",
            external_id="ext_http_1",
            kind="preference",
            text=secret_text,
            source_ids=[s1],
            domain="real",
            visible_to=["char_b"],
        )

        collection_created = False

        def responder(
            method: str,
            url: str,
            body: dict[str, Any] | None,
            headers: dict[str, str],
            timeout: float,
        ) -> tuple[int, dict[str, Any]]:
            nonlocal collection_created
            if method == "GET" and "/collections/" in url:
                if not collection_created:
                    return 404, {"status": {"error": "Not found"}}
                return 200, {
                    "status": "ok",
                    "result": {
                        "config": {
                            "params": {
                                "vectors": {"size": 4, "distance": "Cosine"}
                            }
                        }
                    },
                }
            if method == "PUT" and url.endswith(f"/collections/{expected_collection}"):
                collection_created = True
                return 200, {"status": "ok", "result": True}
            if method == "PUT" and url.endswith(
                f"/collections/{expected_collection}/points?wait=true"
            ):
                return 200, {
                    "status": "ok",
                    "result": {"operation_id": 1, "status": "completed"},
                }
            if method == "POST" and url.endswith(
                f"/collections/{expected_collection}/points/delete?wait=true"
            ):
                return 200, {
                    "status": "ok",
                    "result": {"operation_id": 2, "status": "completed"},
                }
            raise AssertionError(f"Unexpected request: {method} {url}")

        transport = FakeHTTPTransport(responder)
        expected_collection = derive_collection_name("companion_mem", "bge-m3", 4)

        # Constructor performs zero network calls
        qdrant = QdrantHTTPBackend(
            base_url="https://qdrant.internal:6333/",
            collection="companion_mem",
            dimension=4,
            model_id="bge-m3",
            api_key=secret_api_key,
            timeout=5.0,
            transport=transport,
        )
        self.assertEqual(transport.requests, [])
        self.assertEqual(qdrant.collection_name, expected_collection)
        self.assertNotIn(secret_api_key, repr(qdrant))

        embedder = FakeEmbedder(model_id="bge-m3", dimension=4)
        worker = IndexWorker(self.store, embedder, qdrant)

        # Run upsert job
        up_report = worker.run_once("owner_1", "char_a")
        self.assertEqual(up_report["status"], "ok")
        self.assertEqual(len(transport.requests), 3)

        # Request 1: GET /collections/{collection_name}
        req_get = transport.requests[0]
        self.assertEqual(req_get["method"], "GET")
        self.assertEqual(
            req_get["url"],
            f"https://qdrant.internal:6333/collections/{expected_collection}",
        )
        self.assertIsNone(req_get["json"])
        self.assertEqual(req_get["headers"]["api-key"], secret_api_key)

        # Request 2: PUT /collections/{collection_name} (after 404)
        req_create = transport.requests[1]
        self.assertEqual(req_create["method"], "PUT")
        self.assertEqual(
            req_create["url"],
            f"https://qdrant.internal:6333/collections/{expected_collection}",
        )
        self.assertEqual(
            req_create["json"],
            {"vectors": {"size": 4, "distance": "Cosine"}},
        )

        # Request 3: PUT /collections/{collection_name}/points?wait=true
        req_upsert = transport.requests[2]
        self.assertEqual(req_upsert["method"], "PUT")
        self.assertEqual(
            req_upsert["url"],
            f"https://qdrant.internal:6333/collections/{expected_collection}/points?wait=true",
        )
        points = req_upsert["json"]["points"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0]["id"], m1)
        self.assertEqual(len(points[0]["vector"]), 4)
        self.assertEqual(
            points[0]["payload"],
            {
                "memory_id": m1,
                "owner_id": "owner_1",
                "creator_character_id": "char_a",
                "visible_to": ["char_a", "char_b"],
                "domain": "real",
                "embedding_model_id": "bge-m3",
            },
        )

        # Delete memory and run delete job
        self.store.delete_memory("owner_1", "char_a", m1)
        del_report = worker.run_once("owner_1", "char_a")
        self.assertEqual(del_report["status"], "ok")
        self.assertEqual(len(transport.requests), 4)

        req_del = transport.requests[3]
        self.assertEqual(req_del["method"], "POST")
        self.assertEqual(
            req_del["url"],
            f"https://qdrant.internal:6333/collections/{expected_collection}/points/delete?wait=true",
        )
        self.assertEqual(req_del["json"], {"points": [m1]})

        # Verify no private text or source content ever appeared in any HTTP request
        for req in transport.requests:
            self._assert_no_private_strings(
                req["json"], [secret_text, "Source dialogue text", s1]
            )

    def test_qdrant_existing_collection_validation_and_defensive_responses(
        self,
    ) -> None:
        # 1. Existing collection with matching size & Cosine succeeds without PUT create
        matching_transport = FakeHTTPTransport(
            lambda method, url, body, headers, timeout: (
                200,
                {
                    "status": "ok",
                    "result": {
                        "config": {
                            "params": {
                                "vectors": {"size": 4, "distance": "Cosine"}
                            }
                        }
                    },
                },
            )
        )
        b_ok = QdrantHTTPBackend(
            "http://localhost:6333",
            "mem_prefix",
            4,
            "model-1",
            transport=matching_transport,
        )
        b_ok.ensure_collection(4, "model-1")
        self.assertEqual(len(matching_transport.requests), 1)
        self.assertEqual(matching_transport.requests[0]["method"], "GET")

        # 2. Existing collection with wrong vector size rejects
        wrong_size_transport = FakeHTTPTransport(
            lambda method, url, body, headers, timeout: (
                200,
                {
                    "status": "ok",
                    "result": {
                        "config": {
                            "params": {
                                "vectors": {"size": 768, "distance": "Cosine"}
                            }
                        }
                    },
                },
            )
        )
        b_bad_size = QdrantHTTPBackend(
            "http://localhost:6333",
            "mem_prefix",
            4,
            "model-1",
            transport=wrong_size_transport,
        )
        with self.assertRaises(QdrantConfigMismatchError):
            b_bad_size.ensure_collection()

        # 3. Existing collection with wrong distance metric rejects
        wrong_dist_transport = FakeHTTPTransport(
            lambda method, url, body, headers, timeout: (
                200,
                {
                    "status": "ok",
                    "result": {
                        "config": {
                            "params": {
                                "vectors": {"size": 4, "distance": "Euclid"}
                            }
                        }
                    },
                },
            )
        )
        b_bad_dist = QdrantHTTPBackend(
            "http://localhost:6333",
            "mem_prefix",
            4,
            "model-1",
            transport=wrong_dist_transport,
        )
        with self.assertRaises(QdrantConfigMismatchError):
            b_bad_dist.ensure_collection()

        # 4. Defensive rejection of HTTP 200 with failure result or malformed bodies
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "src")
        self.store.save_memory(
            "owner_1", "char_a", "e1", "fact", "Test defensive HTTP parsing", [s1]
        )
        embedder = FakeEmbedder(model_id="model-1", dimension=4)

        bad_upsert_responses: list[tuple[int, Any]] = [
            (200, {"status": "error", "status_error": "internal error"}),
            (200, {"status": "ok", "result": {"status": "failed"}}),
            (200, {"status": "ok", "result": {"status": "acknowledged"}}),
            (200, {"status": "ok", "result": {"status": "ok"}}),
            (200, {"status": "ok", "result": {}}),
            (200, {"status": "ok", "result": True}),
            (200, {"status": "ok", "result": False}),
            (200, {"status": "ok", "result": None}),
            (200, {"status": "ok"}),
            (200, b"{malformed json"),
            (500, {"status": "error"}),
        ]

        for status_code, bad_resp in bad_upsert_responses:

            def make_responder(
                sc: int, br: Any
            ) -> Callable[..., tuple[int, Any]]:
                def _resp(
                    method: str,
                    url: str,
                    body: Any,
                    headers: Any,
                    timeout: float,
                ) -> tuple[int, Any]:
                    if method == "GET":
                        return 200, {
                            "status": "ok",
                            "result": {
                                "config": {
                                    "params": {
                                        "vectors": {
                                            "size": 4,
                                            "distance": "Cosine",
                                        }
                                    }
                                }
                            },
                        }
                    return sc, br

                return _resp

            backend = QdrantHTTPBackend(
                "http://localhost:6333",
                "mem_prefix",
                4,
                "model-1",
                transport=FakeHTTPTransport(make_responder(status_code, bad_resp)),
            )
            worker = IndexWorker(self.store, embedder, backend)
            rep = worker.run_once("owner_1", "char_a")
            self.assertEqual(rep["status"], "failed")
            self.assertEqual(rep["failed"]["error_category"], "backend_error")
            # Job must remain in SQLite (never blindly ACKed!)
            self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # 5. Constructor validation of URL, collection prefix, timeout, api_key
        for bad_url in (
            "",
            "ftp://localhost:6333",
            "http://user:password@localhost:6333",
            "https://token@localhost:6333",
            "http://localhost:6333?foo=bar",
            "http://localhost:6333#frag",
            "http://",
        ):
            with self.assertRaises(ValueError):
                QdrantHTTPBackend(bad_url, "valid_prefix", 4, "model-1")

        for bad_prefix in ("", "  ", "../escape", "bad/slash", "bad space", "-bad"):
            with self.assertRaises(ValueError):
                QdrantHTTPBackend("http://localhost:6333", bad_prefix, 4, "model-1")

        for bad_timeout in (0, -1.0, float("inf"), float("nan"), True, "5"):
            with self.assertRaises(ValueError):
                QdrantHTTPBackend(
                    "http://localhost:6333",
                    "valid_prefix",
                    4,
                    "model-1",
                    timeout=bad_timeout,  # type: ignore[arg-type]
                )

    def test_qdrant_fail_closed_mutation_acknowledgement_through_worker(
        self,
    ) -> None:
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "src")
        m1 = self.store.save_memory(
            "owner_1", "char_a", "e_mut_1", "fact", "Wait=true strict completion", [s1]
        )
        embedder = FakeEmbedder(model_id="model-strict", dimension=4)
        expected_col = derive_collection_name("strict_mem", "model-strict", 4)

        valid_collection_info = {
            "status": "ok",
            "result": {
                "config": {
                    "params": {
                        "vectors": {"size": 4, "distance": "Cosine"}
                    }
                }
            },
        }

        # 1. Collection creation requires literal result True; dict/acknowledged/completed/empty/bool/int fail closed
        non_true_collection_results: list[Any] = [
            {},
            {"status": "completed"},
            {"status": "acknowledged"},
            {"status": "ok"},
            False,
            None,
            1,
            "true",
        ]
        for bad_col_result in non_true_collection_results:
            col_transport = FakeHTTPTransport(
                lambda method, url, body, headers, timeout, res=bad_col_result: (
                    (404, {"status": {"error": "Not found"}})
                    if method == "GET"
                    else (200, {"status": "ok", "result": res})
                )
            )
            col_backend = QdrantHTTPBackend(
                "http://localhost:6333",
                "strict_mem",
                4,
                "model-strict",
                transport=col_transport,
            )
            col_worker = IndexWorker(self.store, embedder, col_backend)
            col_rep = col_worker.run_once("owner_1", "char_a")
            self.assertEqual(col_rep["status"], "failed")
            self.assertEqual(col_rep["processed_count"], 0)
            self.assertEqual(col_rep["failed"]["error_category"], "backend_error")
            self.assertFalse(col_backend._collection_ensured)
            self.assertEqual(len(col_transport.requests), 2)
            self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # 2. Point upsert requires result dict with explicit status == 'completed';
        # {}, {'status': 'acknowledged'}, True, missing status, etc. must fail closed and retain job
        incomplete_point_results: list[Any] = [
            {},
            {"operation_id": 10},
            {"status": "acknowledged"},
            {"operation_id": 10, "status": "acknowledged"},
            {"status": "ok"},
            {"status": "failed"},
            {"status": None},
            {"status": True},
            {"status": "completed", "error": "partial write error"},
            True,
            False,
            None,
            "completed",
            ["completed"],
        ]
        for bad_pt_result in incomplete_point_results:

            def upsert_responder(
                method: str,
                url: str,
                body: Any,
                headers: Any,
                timeout: float,
                res: Any = bad_pt_result,
            ) -> tuple[int, Any]:
                if method == "GET" and url.endswith(f"/collections/{expected_col}"):
                    return 200, valid_collection_info
                if method == "PUT" and url.endswith(
                    f"/collections/{expected_col}/points?wait=true"
                ):
                    return 200, {"status": "ok", "result": res}
                raise AssertionError(f"Unexpected request: {method} {url}")

            pt_transport = FakeHTTPTransport(upsert_responder)
            pt_backend = QdrantHTTPBackend(
                "http://localhost:6333",
                "strict_mem",
                4,
                "model-strict",
                transport=pt_transport,
            )
            pt_worker = IndexWorker(self.store, embedder, pt_backend)
            pt_rep = pt_worker.run_once("owner_1", "char_a")
            self.assertEqual(pt_rep["status"], "failed")
            self.assertEqual(pt_rep["processed_count"], 0)
            self.assertEqual(pt_rep["failed_count"], 1)
            self.assertEqual(pt_rep["failed"]["error_category"], "backend_error")
            self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # Truthful completed upsert succeeds and removes the job
        ok_upsert_transport = FakeHTTPTransport(
            lambda method, url, body, headers, timeout: (
                (200, valid_collection_info)
                if method == "GET"
                else (
                    200,
                    {
                        "status": "ok",
                        "result": {"operation_id": 11, "status": "completed"},
                    },
                )
            )
        )
        ok_backend = QdrantHTTPBackend(
            "http://localhost:6333",
            "strict_mem",
            4,
            "model-strict",
            transport=ok_upsert_transport,
        )
        ok_worker = IndexWorker(self.store, embedder, ok_backend)
        ok_rep = ok_worker.run_once("owner_1", "char_a")
        self.assertEqual(ok_rep["status"], "ok")
        self.assertEqual(ok_rep["upserted_count"], 1)
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

        # 3. Point delete also requires result dict with explicit status == 'completed'
        self.assertTrue(self.store.delete_memory("owner_1", "char_a", m1))
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        for bad_del_result in incomplete_point_results:
            del_transport = FakeHTTPTransport(
                lambda method, url, body, headers, timeout, res=bad_del_result: (
                    200,
                    {"status": "ok", "result": res},
                )
            )
            del_backend = QdrantHTTPBackend(
                "http://localhost:6333",
                "strict_mem",
                4,
                "model-strict",
                transport=del_transport,
            )
            del_worker = IndexWorker(self.store, embedder, del_backend)
            del_rep = del_worker.run_once("owner_1", "char_a")
            self.assertEqual(del_rep["status"], "failed")
            self.assertEqual(del_rep["processed_count"], 0)
            self.assertEqual(del_rep["failed"]["error_category"], "backend_error")
            self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # Truthful completed delete succeeds and removes the delete job
        ok_del_transport = FakeHTTPTransport(
            lambda method, url, body, headers, timeout: (
                200,
                {
                    "status": "ok",
                    "result": {"operation_id": 12, "status": "completed"},
                },
            )
        )
        ok_del_backend = QdrantHTTPBackend(
            "http://localhost:6333",
            "strict_mem",
            4,
            "model-strict",
            transport=ok_del_transport,
        )
        ok_del_worker = IndexWorker(self.store, embedder, ok_del_backend)
        ok_del_rep = ok_del_worker.run_once("owner_1", "char_a")
        self.assertEqual(ok_del_rep["status"], "ok")
        self.assertEqual(ok_del_rep["deleted_count"], 1)
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

        # 4. Direct helper probe on _validate_mutation_response
        QdrantHTTPBackend._validate_mutation_response(
            200, {"status": "ok", "result": True}, "collection creation"
        )
        QdrantHTTPBackend._validate_mutation_response(
            200,
            {"status": "ok", "result": {"status": "completed"}},
            "point upsert",
        )
        QdrantHTTPBackend._validate_mutation_response(
            200,
            {"status": "ok", "result": {"status": "completed"}},
            "point delete",
        )
        for bad_res in ({}, {"status": "acknowledged"}, True):
            with self.assertRaises(QdrantBackendError):
                QdrantHTTPBackend._validate_mutation_response(
                    200, {"status": "ok", "result": bad_res}, "point upsert"
                )
            with self.assertRaises(QdrantBackendError):
                QdrantHTTPBackend._validate_mutation_response(
                    200, {"status": "ok", "result": bad_res}, "point delete"
                )
        with self.assertRaises(QdrantBackendError):
            QdrantHTTPBackend._validate_mutation_response(
                200,
                {"status": "ok", "result": {"status": "completed"}},
                "collection creation",
            )
        with self.assertRaises(QdrantBackendError):
            QdrantHTTPBackend._validate_mutation_response(
                200,
                {"status": "ok", "result": {"status": "completed"}},
                "unknown op",
            )

    def test_qdrant_delete_missing_collection_idempotency_vs_arbitrary_404(
        self,
    ) -> None:
        # Memory created and deleted before ever being indexed -> only a delete job is queued
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "src")
        m1 = self.store.save_memory(
            "owner_1", "char_a", "e_del_404", "fact", "Deleted before indexing", [s1]
        )
        self.assertTrue(self.store.delete_memory("owner_1", "char_a", m1))
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)
        self.assertEqual(self._raw_job_rows("owner_1", "char_a")[0]["action"], "delete")

        expected_col = derive_collection_name("companion_mem", "m-v1", 4)
        delete_url = (
            f"http://localhost:6333/collections/{expected_col}/points/delete?wait=true"
        )
        collection_url = f"http://localhost:6333/collections/{expected_col}"

        # 1. Arbitrary 404 on point delete where follow-up GET on exact collection does NOT
        # confirm a missing collection (exists, malformed, empty, auth failure, 500, timeout)
        unconfirmed_get_outcomes: list[Any] = [
            (
                200,
                {
                    "status": "ok",
                    "result": {
                        "config": {
                            "params": {
                                "vectors": {"size": 4, "distance": "Cosine"}
                            }
                        }
                    },
                },
            ),
            (200, {"status": "ok", "result": {}}),
            (404, b"{malformed json"),
            (404, None),
            (404, b""),
            (404, {"status": "ok"}),
            (401, {"status": {"error": "Unauthorized"}}),
            (403, {"status": {"error": "Forbidden"}}),
            (500, {"status": {"error": "Internal server error"}}),
            TimeoutError("collection GET timed out"),
        ]

        for outcome in unconfirmed_get_outcomes:

            def make_unconfirmed_responder(
                get_outcome: Any,
            ) -> Callable[..., tuple[int, Any]]:
                def _resp(
                    method: str,
                    url: str,
                    body: Any,
                    headers: Any,
                    timeout: float,
                ) -> tuple[int, Any]:
                    if method == "POST" and url == delete_url:
                        return 404, {"status": {"error": "Not found"}}
                    if method == "GET" and url == collection_url:
                        if isinstance(get_outcome, Exception):
                            raise get_outcome
                        return get_outcome
                    raise AssertionError(f"Unexpected request: {method} {url}")

                return _resp

            transport = FakeHTTPTransport(make_unconfirmed_responder(outcome))
            qdrant = QdrantHTTPBackend(
                "http://localhost:6333",
                "companion_mem",
                4,
                "m-v1",
                transport=transport,
            )
            worker = IndexWorker(self.store, None, qdrant)
            report = worker.run_once("owner_1", "char_a")
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["processed_count"], 0)
            self.assertEqual(report["failed"]["error_category"], "backend_error")
            self.assertEqual(
                [r["method"] for r in transport.requests], ["POST", "GET"]
            )
            self.assertEqual(transport.requests[0]["url"], delete_url)
            self.assertEqual(transport.requests[1]["url"], collection_url)
            # Delete job MUST remain pending in SQLite
            self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # 2. Confirmed missing collection: point delete returns 404 AND exact collection GET returns 404
        def confirmed_missing_responder(
            method: str,
            url: str,
            body: Any,
            headers: Any,
            timeout: float,
        ) -> tuple[int, Any]:
            if method == "POST" and url == delete_url:
                return 404, {
                    "status": {
                        "error": f"Not found: Collection `{expected_col}` doesn't exist!"
                    }
                }
            if method == "GET" and url == collection_url:
                return 404, {
                    "status": {
                        "error": f"Not found: Collection `{expected_col}` doesn't exist!"
                    }
                }
            raise AssertionError(f"Unexpected request: {method} {url}")

        confirmed_transport = FakeHTTPTransport(confirmed_missing_responder)
        confirmed_qdrant = QdrantHTTPBackend(
            "http://localhost:6333",
            "companion_mem",
            4,
            "m-v1",
            transport=confirmed_transport,
        )
        confirmed_worker = IndexWorker(self.store, None, confirmed_qdrant)
        ok_report = confirmed_worker.run_once("owner_1", "char_a")
        self.assertEqual(ok_report["status"], "ok")
        self.assertEqual(ok_report["processed_count"], 1)
        self.assertEqual(ok_report["deleted_count"], 1)
        self.assertEqual(ok_report["failed_count"], 0)
        # Only POST delete and GET collection were called; collection was NOT created via PUT
        self.assertEqual(
            [(r["method"], r["url"]) for r in confirmed_transport.requests],
            [("POST", delete_url), ("GET", collection_url)],
        )
        self.assertFalse(confirmed_qdrant._collection_ensured)
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])

    def test_same_dimension_different_model_collection_separation(self) -> None:
        b_model_a = QdrantHTTPBackend(
            "http://localhost:6333",
            "companion_mem",
            dimension=384,
            model_id="multilingual-e5-small",
            transport=lambda *args: (200, {}),
        )
        b_model_b = QdrantHTTPBackend(
            "http://localhost:6333",
            "companion_mem",
            dimension=384,
            model_id="bge-small-zh-v1.5",
            transport=lambda *args: (200, {}),
        )
        b_model_a_repeat = QdrantHTTPBackend(
            "http://localhost:6333",
            "companion_mem",
            dimension=384,
            model_id="multilingual-e5-small",
            transport=lambda *args: (200, {}),
        )

        self.assertNotEqual(b_model_a.collection_name, b_model_b.collection_name)
        self.assertEqual(b_model_a.collection_name, b_model_a_repeat.collection_name)
        self.assertTrue(b_model_a.collection_name.startswith("companion_mem_d384_"))
        self.assertTrue(b_model_b.collection_name.startswith("companion_mem_d384_"))

    def test_network_failure_sanitized_and_overlapping_run_once_rejected(
        self,
    ) -> None:
        secret_url_leak = "http://secret-internal-host.local:6333/ leak_key=sec_999"
        s1 = self.store.append_message("owner_1", "char_a", "m1", "user", "src")
        m1 = self.store.save_memory(
            "owner_1", "char_a", "e1", "fact", "Secret memory for network test", [s1]
        )

        def failing_transport(
            method: str, url: str, body: Any, headers: Any, timeout: float
        ) -> tuple[int, Any]:
            raise urllib.error.URLError(secret_url_leak)

        qdrant = QdrantHTTPBackend(
            "http://localhost:6333",
            "companion_mem",
            4,
            "m-v1",
            api_key="super-secret-api-key-val",
            transport=failing_transport,
        )
        embedder = FakeEmbedder(model_id="m-v1", dimension=4)
        worker = IndexWorker(self.store, embedder, qdrant)

        report = worker.run_once("owner_1", "char_a")
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed"]["error_category"], "backend_error")
        self.assertEqual(report["failed"]["error_type"], "QdrantBackendError")
        self._assert_no_private_strings(
            report,
            [
                secret_url_leak,
                "sec_999",
                "super-secret-api-key-val",
                "Secret memory for network test",
            ],
        )
        self.assertEqual(len(self._raw_job_rows("owner_1", "char_a")), 1)

        # Verify per-instance lock rejects overlapping/re-entrant run_once
        overlap_error: Exception | None = None

        def reentrant_hook(
            mem_id: str, vec: list[float], payload: dict[str, Any]
        ) -> None:
            nonlocal overlap_error
            try:
                reentrant_worker.run_once("owner_1", "char_a")
            except Exception as exc:
                overlap_error = exc

        reentrant_backend = FakeVectorBackend(
            dimension=4, model_id="m-v1", on_upsert_hook=reentrant_hook
        )
        reentrant_worker = IndexWorker(self.store, embedder, reentrant_backend)
        rep_ok = reentrant_worker.run_once("owner_1", "char_a")
        self.assertEqual(rep_ok["status"], "ok")
        self.assertIsInstance(overlap_error, RuntimeError)
        self.assertIn(m1, reentrant_backend.points)
        self.assertEqual(self._raw_job_rows("owner_1", "char_a"), [])


if __name__ == "__main__":
    unittest.main()
