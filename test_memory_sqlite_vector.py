"""Unit tests for memory_sqlite_vector (stdlib unittest, file-backed temp DBs).

All fixtures are fictional. `ToyEmbedder` is a deterministic keyword-count
embedder used only to exercise plumbing and ranking mechanics; it makes no claim
about real semantic quality.
"""

from __future__ import annotations

import itertools
import math
import os
import re
import struct
import tempfile
import unittest
import uuid
from unittest import mock

from memory_core import MemoryStore
from memory_index import IndexWorker
from memory_search import SemanticSearchError
from memory_sqlite_vector import (
    SQLiteSemanticSearcher,
    SQLiteVectorAuthorityError,
    SQLiteVectorBackend,
    SQLiteVectorBackendError,
    SQLiteVectorConfigMismatchError,
    SQLiteVectorError,
)

MODEL_ID = "toy-keyword-v1"
VOCAB = ("apple", "banana", "cherry", "dragon", "engine", "forest", "garden", "harbor")
DIM = len(VOCAB) + 1

OWNER_A = "owner-a"
OWNER_B = "owner-b"
CHAR_A = "char-a"
CHAR_B = "char-b"
CHAR_C = "char-c"
CHAR_X = "char-x"


class ToyEmbedder:
    """Deterministic keyword-count embedder (NOT semantic); never all-zero."""

    model_id = MODEL_ID
    dimension = DIM

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> list[float]:
        self.calls.append(text)
        words = re.findall(r"[a-z]+", text.lower())
        vector = [float(words.count(word)) for word in VOCAB]
        vector.append(0.01)
        return vector


class FnEmbedder:
    """Embedder whose behavior is injected, for fault tests."""

    def __init__(self, fn, model_id: str = MODEL_ID, dimension: int = DIM) -> None:
        self.fn = fn
        self.model_id = model_id
        self.dimension = dimension

    def embed(self, text: str):
        return self.fn(text)


class FailingUpsertBackend(SQLiteVectorBackend):
    def upsert(self, memory_id, vector, payload):  # noqa: D401
        raise RuntimeError("SECRET-BACKEND-BODY")


class SQLiteVectorTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.db_path = os.path.join(self._tmp.name, "memory.db")
        self.store = MemoryStore(self.db_path)
        self.embedder = ToyEmbedder()
        self.backend = SQLiteVectorBackend(self.store, DIM, MODEL_ID)
        self.worker = IndexWorker(self.store, self.embedder, self.backend)
        self.searcher = SQLiteSemanticSearcher(self.store, self.embedder, self.backend)
        self._counter = itertools.count(1)

    # -- fixtures ----------------------------------------------------------

    def remember(self, owner, char, text, domain="real", visible_to=None, sources=1):
        n = next(self._counter)
        message_ids = [
            self.store.append_message(owner, char, f"msg-{n}-{i}", "user", f"chat {n} {i}")
            for i in range(sources)
        ]
        return self.store.save_memory(
            owner, char, f"mem-{n}", "fact", text, message_ids,
            domain=domain, visible_to=visible_to,
        )

    def index(self, owner, char):
        report = self.worker.run_once(owner, char, limit=100)
        self.assertEqual(report["status"], "ok", report)
        self.assertEqual(self.store.pending_jobs(owner, char), [])
        return report

    def sql(self, statement, params=()):
        with self.store._connect() as conn:
            return conn.execute(statement, params).fetchall()

    def vector_count(self):
        return self.sql("SELECT COUNT(*) FROM memory_vectors")[0][0]

    def vector_blob(self, memory_id):
        rows = self.sql("SELECT vector FROM memory_vectors WHERE memory_id = ?", (memory_id,))
        return rows[0][0] if rows else None

    def payload(self, memory_id, owner, char, **overrides):
        record = self.store.get_index_record(owner, char, memory_id)
        payload = {
            "memory_id": memory_id,
            "owner_id": owner,
            "creator_character_id": char,
            "visible_to": list(record["visible_to"]),
            "domain": record["domain"],
            "embedding_model_id": MODEL_ID,
        }
        payload.update(overrides)
        return payload

    def good_vector(self, text="apple"):
        return ToyEmbedder().embed(text)


class BackendConfigTests(SQLiteVectorTestCase):
    def test_tables_created_with_single_binding(self):
        names = {r[0] for r in self.sql("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertIn("memory_vector_config", names)
        self.assertIn("memory_vectors", names)
        config = self.sql("SELECT id, model_id, dimension FROM memory_vector_config")
        self.assertEqual([tuple(r) for r in config], [(1, MODEL_ID, DIM)])
        cols = [r[1] for r in self.sql("PRAGMA table_info(memory_vectors)")]
        self.assertEqual(
            set(cols), {"memory_id", "model_id", "dimension", "vector", "updated_at"}
        )
        self.assertNotIn("text", cols)

    def test_properties_expose_binding(self):
        self.assertEqual(self.backend.model_id, MODEL_ID)
        self.assertEqual(self.backend.dimension, DIM)
        self.assertIs(self.backend.store, self.store)

    def test_reconstruct_is_idempotent_and_keeps_data(self):
        mid = self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        again = SQLiteVectorBackend(self.store, DIM, MODEL_ID)
        again.ensure_collection(DIM, MODEL_ID)
        self.assertEqual(self.vector_count(), 1)
        self.assertIsNotNone(self.vector_blob(mid))
        self.assertEqual(len(self.sql("SELECT * FROM memory_vector_config")), 1)

    def test_constructor_mismatch_fails_closed_without_dropping(self):
        self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        before = self.sql("SELECT memory_id, vector FROM memory_vectors")
        with self.assertRaises(SQLiteVectorConfigMismatchError):
            SQLiteVectorBackend(self.store, DIM, "other-model")
        with self.assertRaises(SQLiteVectorConfigMismatchError):
            SQLiteVectorBackend(self.store, DIM + 1, MODEL_ID)
        after = self.sql("SELECT memory_id, vector FROM memory_vectors")
        self.assertEqual([tuple(r) for r in before], [tuple(r) for r in after])
        config = self.sql("SELECT model_id, dimension FROM memory_vector_config")
        self.assertEqual([tuple(r) for r in config], [(MODEL_ID, DIM)])
        self.assertEqual(len(self.searcher.search(OWNER_A, CHAR_A, "apple")), 1)

    def test_config_mismatch_is_value_error(self):
        with self.assertRaises(ValueError):
            SQLiteVectorBackend(self.store, DIM, "other-model")

    def test_constructor_rejects_bad_arguments(self):
        cases = [
            (None, DIM, MODEL_ID),
            (object(), DIM, MODEL_ID),
            (self.store, True, MODEL_ID),
            (self.store, 0, MODEL_ID),
            (self.store, -1, MODEL_ID),
            (self.store, "9", MODEL_ID),
            (self.store, 9.0, MODEL_ID),
            (self.store, 10**9, MODEL_ID),
            (self.store, DIM, ""),
            (self.store, DIM, " model"),
            (self.store, DIM, None),
            (self.store, DIM, 5),
        ]
        for store, dim, model in cases:
            with self.subTest(dim=dim, model=model):
                with self.assertRaises(ValueError):
                    SQLiteVectorBackend(store, dim, model)

    def test_ensure_collection_verifies_both_sides(self):
        self.assertIsNone(self.backend.ensure_collection(DIM, MODEL_ID))
        for dim, model in [(DIM + 1, MODEL_ID), (DIM, "other-model"), (True, MODEL_ID), (DIM, "")]:
            with self.subTest(dim=dim, model=model):
                with self.assertRaises(SQLiteVectorError):
                    self.backend.ensure_collection(dim, model)

    def test_tampered_persisted_config_detected(self):
        self.sql("UPDATE memory_vector_config SET model_id = 'tampered'")
        with self.assertRaises(SQLiteVectorConfigMismatchError):
            self.backend.ensure_collection(DIM, MODEL_ID)
        with self.assertRaises(SQLiteVectorConfigMismatchError):
            SQLiteVectorBackend(self.store, DIM, MODEL_ID)

    def test_partial_tables_fail_closed_not_rebuilt(self):
        self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        self.sql("DROP TABLE memory_vectors")
        with self.assertRaises(SQLiteVectorBackendError):
            SQLiteVectorBackend(self.store, DIM, MODEL_ID)
        names = {r[0] for r in self.sql("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertNotIn("memory_vectors", names)

    def test_missing_config_with_existing_vectors_fails_closed(self):
        self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        self.sql("DELETE FROM memory_vector_config")
        with self.assertRaises(SQLiteVectorBackendError):
            SQLiteVectorBackend(self.store, DIM, MODEL_ID)
        self.assertEqual(self.vector_count(), 1)

    def test_in_memory_store_roundtrip(self):
        with MemoryStore(":memory:") as mem_store:
            backend = SQLiteVectorBackend(mem_store, DIM, MODEL_ID)
            msg = mem_store.append_message(OWNER_A, CHAR_A, "m1", "user", "chat")
            mid = mem_store.save_memory(OWNER_A, CHAR_A, "e1", "fact", "apple", [msg])
            record = mem_store.get_index_record(OWNER_A, CHAR_A, mid)
            payload = {
                "memory_id": mid, "owner_id": OWNER_A, "creator_character_id": CHAR_A,
                "visible_to": record["visible_to"], "domain": "real",
                "embedding_model_id": MODEL_ID,
            }
            backend.upsert(mid, self.good_vector("apple"), payload)
            results = SQLiteSemanticSearcher(mem_store, ToyEmbedder(), backend).search(
                OWNER_A, CHAR_A, "apple"
            )
            self.assertEqual([r["memory_id"] for r in results], [mid])


class WorkerIntegrationTests(SQLiteVectorTestCase):
    def test_worker_writes_vectors_and_acks(self):
        m1 = self.remember(OWNER_A, CHAR_A, "apple orchard")
        m2 = self.remember(OWNER_A, CHAR_A, "banana harbor")
        report = self.index(OWNER_A, CHAR_A)
        self.assertEqual(report["upserted_count"], 2)
        self.assertEqual(self.vector_count(), 2)
        for mid, text in ((m1, "apple orchard"), (m2, "banana harbor")):
            blob = self.vector_blob(mid)
            self.assertEqual(len(blob), 4 * DIM)
            expected = struct.pack(f"<{DIM}f", *self.good_vector(text))
            self.assertEqual(blob, expected)
        row = self.sql("SELECT model_id, dimension FROM memory_vectors LIMIT 1")[0]
        self.assertEqual((row[0], row[1]), (MODEL_ID, DIM))

    def test_no_canonical_text_duplicated_in_vector_tables(self):
        self.remember(OWNER_A, CHAR_A, "apple orchard unique-fictional-sentence")
        self.index(OWNER_A, CHAR_A)
        dump = repr([tuple(r) for r in self.sql("SELECT * FROM memory_vectors")])
        dump += repr([tuple(r) for r in self.sql("SELECT * FROM memory_vector_config")])
        self.assertNotIn("unique-fictional-sentence", dump)

    def test_restart_persistence(self):
        mid = self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        store2 = MemoryStore(self.db_path)
        backend2 = SQLiteVectorBackend(store2, DIM, MODEL_ID)
        searcher2 = SQLiteSemanticSearcher(store2, ToyEmbedder(), backend2)
        results = searcher2.search(OWNER_A, CHAR_A, "apple")
        self.assertEqual([r["memory_id"] for r in results], [mid])

    def test_failed_upsert_is_not_acked_and_leaks_nothing(self):
        mid = self.remember(OWNER_A, CHAR_A, "apple orchard")
        bad_backend = FailingUpsertBackend(self.store, DIM, MODEL_ID)
        worker = IndexWorker(self.store, self.embedder, bad_backend)
        report = worker.run_once(OWNER_A, CHAR_A)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(report["failed"]["error_category"], "backend_error")
        self.assertNotIn("SECRET-BACKEND-BODY", repr(report))
        self.assertEqual(len(self.store.pending_jobs(OWNER_A, CHAR_A)), 1)
        self.assertEqual(self.vector_count(), 0)
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_A, CHAR_A, "apple")
        # A healthy retry succeeds and acks.
        self.index(OWNER_A, CHAR_A)
        self.assertEqual(self.vector_count(), 1)
        self.assertIsNotNone(self.vector_blob(mid))

    def test_worker_delete_job_removes_vector_idempotently(self):
        mid = self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        self.assertTrue(self.store.delete_memory(OWNER_A, CHAR_A, mid))
        self.index(OWNER_A, CHAR_A)
        self.assertEqual(self.vector_count(), 0)
        self.assertFalse(self.store.delete_memory(OWNER_A, CHAR_A, mid))

    def test_worker_with_mismatched_embedder_fails_without_writing(self):
        self.remember(OWNER_A, CHAR_A, "apple orchard")
        other = FnEmbedder(lambda t: [1.0] * 3, model_id=MODEL_ID, dimension=3)
        worker = IndexWorker(self.store, other, self.backend)
        report = worker.run_once(OWNER_A, CHAR_A)
        self.assertEqual(report["status"], "failed")
        self.assertEqual(self.vector_count(), 0)
        self.assertEqual(len(self.store.pending_jobs(OWNER_A, CHAR_A)), 1)


class BackendUpsertValidationTests(SQLiteVectorTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.mid = self.remember(OWNER_A, CHAR_A, "apple orchard")

    def test_valid_upsert_then_overwrite(self):
        self.backend.upsert(self.mid, self.good_vector("apple"), self.payload(self.mid, OWNER_A, CHAR_A))
        self.assertEqual(self.vector_count(), 1)
        self.backend.upsert(self.mid, self.good_vector("banana"), self.payload(self.mid, OWNER_A, CHAR_A))
        self.assertEqual(self.vector_count(), 1)
        self.assertEqual(
            self.vector_blob(self.mid), struct.pack(f"<{DIM}f", *self.good_vector("banana"))
        )

    def test_integer_elements_accepted(self):
        vec = [1] + [0] * (DIM - 1)
        self.backend.upsert(self.mid, vec, self.payload(self.mid, OWNER_A, CHAR_A))
        self.assertEqual(self.vector_count(), 1)

    def test_malformed_memory_id_rejected(self):
        for bad in ["not-a-uuid", "ABCDEF12-3456-4789-ABCD-EF1234567890", 123, None, "", " " + self.mid, b"x"]:
            with self.subTest(bad=bad):
                with self.assertRaises(SQLiteVectorError):
                    self.backend.upsert(bad, self.good_vector(), self.payload(self.mid, OWNER_A, CHAR_A))
                with self.assertRaises(SQLiteVectorError):
                    self.backend.delete(bad)
        self.assertEqual(self.vector_count(), 0)

    def test_payload_must_be_strict_and_consistent(self):
        base = self.payload(self.mid, OWNER_A, CHAR_A)
        extra = dict(base, text="SECRETTEXT")
        missing = {k: v for k, v in base.items() if k != "domain"}
        cases = {
            "not_dict": ["x"],
            "none": None,
            "extra_key": extra,
            "missing_key": missing,
            "other_memory_id": dict(base, memory_id=str(uuid.uuid4())),
            "other_model": dict(base, embedding_model_id="other-model"),
            "bad_domain": dict(base, domain="elsewhere"),
            "empty_visible": dict(base, visible_to=[]),
            "unsorted_visible": dict(base, visible_to=[CHAR_B, CHAR_A]),
            "dup_visible": dict(base, visible_to=[CHAR_A, CHAR_A]),
            "visible_missing_creator": dict(base, visible_to=[CHAR_B]),
            "str_visible": dict(base, visible_to=CHAR_A),
        }
        for name, payload in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(SQLiteVectorError) as ctx:
                    self.backend.upsert(self.mid, self.good_vector(), payload)
                self.assertNotIn("SECRETTEXT", str(ctx.exception))
        self.assertEqual(self.vector_count(), 0)

    def test_invalid_vectors_rejected(self):
        good = self.good_vector()

        def with_first(value):
            return [value] + good[1:]

        cases = {
            "bool": with_first(True),
            "string_element": with_first("SECRETVEC"),
            "none_element": with_first(None),
            "nan": with_first(float("nan")),
            "inf": with_first(float("inf")),
            "neg_inf": with_first(float("-inf")),
            "zero": [0.0] * DIM,
            "int_zero": [0] * DIM,
            "short": good[:-1],
            "long": good + [1.0],
            "empty": [],
            "none": None,
            "string": "SECRETVEC",
            "dict": {"a": 1},
            "bytes": b"\x00" * DIM,
            "overflow": with_first(1e39),
            "neg_overflow": with_first(-1e39),
            "huge_int": with_first(10**400),
            "underflow_only": [1e-50] + [0.0] * (DIM - 1),
            "underflow_element": [1e-50, 1.0] + [0.0] * (DIM - 2),
        }
        for name, vec in cases.items():
            with self.subTest(name=name):
                with self.assertRaises(SQLiteVectorError) as ctx:
                    self.backend.upsert(self.mid, vec, self.payload(self.mid, OWNER_A, CHAR_A))
                self.assertNotIn("SECRETVEC", str(ctx.exception))
        self.assertEqual(self.vector_count(), 0)

    def test_float32_boundary_values_accepted(self):
        vec = [3.4028234e38] + [1e-38] + [0.0] * (DIM - 2)
        self.backend.upsert(self.mid, vec, self.payload(self.mid, OWNER_A, CHAR_A))
        self.assertEqual(self.vector_count(), 1)

    def test_forged_metadata_rejected(self):
        forged = {
            "owner": self.payload(self.mid, OWNER_A, CHAR_A, owner_id="SECRET-owner"),
            "creator": self.payload(self.mid, OWNER_A, CHAR_A, creator_character_id=CHAR_B,
                                    visible_to=[CHAR_B]),
            "domain": self.payload(self.mid, OWNER_A, CHAR_A, domain="roleplay"),
            "extra_reader": self.payload(self.mid, OWNER_A, CHAR_A, visible_to=[CHAR_A, CHAR_C]),
        }
        for name, payload in forged.items():
            with self.subTest(name=name):
                with self.assertRaises(SQLiteVectorAuthorityError) as ctx:
                    self.backend.upsert(self.mid, self.good_vector(), payload)
                self.assertNotIn("SECRET", str(ctx.exception))
        self.assertEqual(self.vector_count(), 0)

    def test_cross_owner_rejected(self):
        payload = self.payload(self.mid, OWNER_A, CHAR_A, owner_id=OWNER_B)
        with self.assertRaises(SQLiteVectorAuthorityError):
            self.backend.upsert(self.mid, self.good_vector(), payload)
        other_mid = self.remember(OWNER_B, CHAR_B, "apple harbor")
        spoof = self.payload(other_mid, OWNER_B, CHAR_B, owner_id=OWNER_A, creator_character_id=CHAR_A,
                             visible_to=[CHAR_A])
        with self.assertRaises(SQLiteVectorAuthorityError):
            self.backend.upsert(other_mid, self.good_vector(), spoof)
        self.assertEqual(self.vector_count(), 0)

    def test_shared_reader_cannot_spoof_write(self):
        shared = self.remember(OWNER_A, CHAR_A, "banana harbor", visible_to=[CHAR_B])
        # Reader claims to be the creator.
        spoof_creator = self.payload(shared, OWNER_A, CHAR_A, creator_character_id=CHAR_B)
        with self.assertRaises(SQLiteVectorAuthorityError):
            self.backend.upsert(shared, self.good_vector(), spoof_creator)
        # Reader claims canonical creator but widens the readers.
        spoof_readers = self.payload(shared, OWNER_A, CHAR_A, visible_to=[CHAR_A, CHAR_B, CHAR_C])
        with self.assertRaises(SQLiteVectorAuthorityError):
            self.backend.upsert(shared, self.good_vector(), spoof_readers)
        self.assertEqual(self.vector_count(), 0)

    def test_inactive_memory_rejected(self):
        corrected = self.store.correct_memory(
            OWNER_A, CHAR_A, self.mid, "ext-correct", "banana harbor",
            [self.store.append_message(OWNER_A, CHAR_A, "src-corr", "user", "chat")],
        )
        old_payload = {
            "memory_id": self.mid, "owner_id": OWNER_A, "creator_character_id": CHAR_A,
            "visible_to": [CHAR_A], "domain": "real", "embedding_model_id": MODEL_ID,
        }
        with self.assertRaises(SQLiteVectorAuthorityError):
            self.backend.upsert(self.mid, self.good_vector(), old_payload)
        self.assertTrue(self.store.delete_memory(OWNER_A, CHAR_A, corrected))
        new_payload = dict(old_payload, memory_id=corrected)
        with self.assertRaises(SQLiteVectorAuthorityError):
            self.backend.upsert(corrected, self.good_vector(), new_payload)
        self.assertEqual(self.vector_count(), 0)

    def test_unknown_memory_rejected(self):
        ghost = str(uuid.uuid4())
        payload = {
            "memory_id": ghost, "owner_id": OWNER_A, "creator_character_id": CHAR_A,
            "visible_to": [CHAR_A], "domain": "real", "embedding_model_id": MODEL_ID,
        }
        with self.assertRaises(SQLiteVectorAuthorityError):
            self.backend.upsert(ghost, self.good_vector(), payload)

    def test_delete_is_idempotent_and_never_deletes_source_memory(self):
        self.backend.upsert(self.mid, self.good_vector(), self.payload(self.mid, OWNER_A, CHAR_A))
        self.backend.delete(self.mid)
        self.backend.delete(self.mid)
        self.backend.delete(str(uuid.uuid4()))
        self.assertEqual(self.vector_count(), 0)
        self.assertIsNotNone(self.store.get_memory(OWNER_A, CHAR_A, self.mid))

    def test_failed_validation_leaves_existing_vector_unchanged(self):
        self.backend.upsert(self.mid, self.good_vector("apple"), self.payload(self.mid, OWNER_A, CHAR_A))
        before = self.vector_blob(self.mid)
        with self.assertRaises(SQLiteVectorError):
            self.backend.upsert(self.mid, [float("nan")] * DIM, self.payload(self.mid, OWNER_A, CHAR_A))
        self.assertEqual(self.vector_blob(self.mid), before)

    def test_operations_fail_loudly_when_table_missing(self):
        self.sql("ALTER TABLE memory_vectors RENAME TO memory_vectors_old")
        with self.assertRaises(SQLiteVectorBackendError):
            self.backend.upsert(self.mid, self.good_vector(), self.payload(self.mid, OWNER_A, CHAR_A))
        with self.assertRaises(SQLiteVectorBackendError):
            self.backend.delete(self.mid)
        with self.assertRaises(SQLiteVectorBackendError):
            self.backend.ensure_collection(DIM, MODEL_ID)


class SearchBehaviorTests(SQLiteVectorTestCase):
    def test_ranking_scores_and_canonical_fields(self):
        m1 = self.remember(OWNER_A, CHAR_A, "apple apple apple")
        m2 = self.remember(OWNER_A, CHAR_A, "apple banana")
        m3 = self.remember(OWNER_A, CHAR_A, "cherry dragon")
        self.index(OWNER_A, CHAR_A)
        results = self.searcher.search(OWNER_A, CHAR_A, "apple")
        self.assertEqual([r["memory_id"] for r in results], [m1, m2, m3])
        scores = [r["score"] for r in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertGreater(scores[0], 0.99)
        self.assertAlmostEqual(scores[1], 0.7071, delta=0.01)
        for r in results:
            self.assertIsInstance(r["score"], float)
            self.assertTrue(math.isfinite(r["score"]))
            self.assertTrue(-1.0 <= r["score"] <= 1.0)
        self.assertEqual(results[0]["text"], "apple apple apple")
        self.assertEqual(results[0]["owner_id"], OWNER_A)

    def test_source_ids_preserved(self):
        mid = self.remember(OWNER_A, CHAR_A, "apple garden", sources=3)
        self.index(OWNER_A, CHAR_A)
        canonical = self.store.get_memory(OWNER_A, CHAR_A, mid)
        results = self.searcher.search(OWNER_A, CHAR_A, "apple")
        self.assertEqual(len(results[0]["source_ids"]), 3)
        self.assertEqual(results[0]["source_ids"], canonical["source_ids"])
        self.assertEqual(results[0]["visible_to"], [CHAR_A])

    def test_top_k_never_exceeds_limit(self):
        for count in range(1, 7):
            self.remember(OWNER_A, CHAR_A, " ".join(["apple"] * count) + " banana")
        self.index(OWNER_A, CHAR_A)
        for limit in (1, 3, 6, 10):
            with self.subTest(limit=limit):
                results = self.searcher.search(OWNER_A, CHAR_A, "apple", limit=limit)
                self.assertEqual(len(results), min(limit, 6))
        self.assertEqual(len(self.searcher.search(OWNER_A, CHAR_A, "apple")), 5)

    def test_ties_are_deterministic(self):
        ids = [self.remember(OWNER_A, CHAR_A, "apple banana") for _ in range(4)]
        self.index(OWNER_A, CHAR_A)
        first = self.searcher.search(OWNER_A, CHAR_A, "apple", limit=4)
        second = self.searcher.search(OWNER_A, CHAR_A, "apple", limit=4)
        self.assertEqual(len({r["score"] for r in first}), 1)
        self.assertEqual([r["memory_id"] for r in first], sorted(ids))
        self.assertEqual([r["memory_id"] for r in first], [r["memory_id"] for r in second])

    def test_valid_empty_database_returns_empty_list(self):
        self.assertEqual(self.searcher.search(OWNER_A, CHAR_A, "apple"), [])

    def test_invalid_requests_rejected(self):
        bad_calls = [
            dict(owner_id="", character_id=CHAR_A, query="apple"),
            dict(owner_id=OWNER_A, character_id=" x", query="apple"),
            dict(owner_id=None, character_id=CHAR_A, query="apple"),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query=""),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query="   "),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query=None),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query="x" * 1025),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query="a\x00b"),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query="apple", limit=0),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query="apple", limit=True),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query="apple", limit=1001),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query="apple", limit="5"),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query="apple", domain="elsewhere"),
            dict(owner_id=OWNER_A, character_id=CHAR_A, query="apple", domain=5),
        ]
        for kwargs in bad_calls:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    self.searcher.search(**kwargs)

    def test_cross_owner_isolation(self):
        mine = self.remember(OWNER_A, CHAR_A, "apple harbor")
        theirs = self.remember(OWNER_B, CHAR_A, "apple harbor")
        self.index(OWNER_A, CHAR_A)
        self.index(OWNER_B, CHAR_A)
        self.assertEqual(
            [r["memory_id"] for r in self.searcher.search(OWNER_A, CHAR_A, "apple")], [mine]
        )
        self.assertEqual(
            [r["memory_id"] for r in self.searcher.search(OWNER_B, CHAR_A, "apple")], [theirs]
        )

    def test_domain_filter_applied_before_ranking(self):
        real = self.remember(OWNER_A, CHAR_A, "apple", domain="real")
        play = self.remember(OWNER_A, CHAR_A, "apple apple banana", domain="roleplay")
        self.index(OWNER_A, CHAR_A)
        both = {r["memory_id"] for r in self.searcher.search(OWNER_A, CHAR_A, "apple")}
        self.assertEqual(both, {real, play})
        only_real = self.searcher.search(OWNER_A, CHAR_A, "apple", domain="real")
        self.assertEqual([r["memory_id"] for r in only_real], [real])
        only_play = self.searcher.search(OWNER_A, CHAR_A, "apple", domain="roleplay")
        self.assertEqual([r["memory_id"] for r in only_play], [play])

    def test_private_memory_hidden_from_other_characters(self):
        private = self.remember(OWNER_A, CHAR_A, "apple vault")
        self.index(OWNER_A, CHAR_A)
        self.assertEqual(
            [r["memory_id"] for r in self.searcher.search(OWNER_A, CHAR_A, "apple")], [private]
        )
        self.assertEqual(self.searcher.search(OWNER_A, CHAR_B, "apple"), [])

    def test_shared_reader_allowed_others_denied(self):
        shared = self.remember(OWNER_A, CHAR_A, "apple vault", visible_to=[CHAR_B])
        self.index(OWNER_A, CHAR_A)
        got = self.searcher.search(OWNER_A, CHAR_B, "apple")
        self.assertEqual([r["memory_id"] for r in got], [shared])
        self.assertEqual(got[0]["visible_to"], [CHAR_A, CHAR_B])
        self.assertEqual(got[0]["creator_character_id"], CHAR_A)
        self.assertEqual(self.searcher.search(OWNER_A, CHAR_C, "apple"), [])

    def test_forged_reader_cannot_gain_access_through_payload(self):
        private = self.remember(OWNER_A, CHAR_A, "apple vault")
        forged = self.payload(private, OWNER_A, CHAR_A, visible_to=[CHAR_A, CHAR_C])
        with self.assertRaises(SQLiteVectorAuthorityError):
            self.backend.upsert(private, self.good_vector("apple"), forged)
        self.index(OWNER_A, CHAR_A)
        self.assertEqual(self.searcher.search(OWNER_A, CHAR_C, "apple"), [])

    def test_unauthorized_high_scores_do_not_starve_authorized(self):
        for _ in range(30):
            self.remember(OWNER_A, CHAR_X, "apple apple apple apple")
        for _ in range(10):
            self.remember(OWNER_B, CHAR_A, "apple apple apple apple")
        low1 = self.remember(OWNER_A, CHAR_A, "apple banana banana banana")
        low2 = self.remember(OWNER_A, CHAR_A, "apple cherry cherry cherry")
        self.index(OWNER_A, CHAR_X)
        self.index(OWNER_B, CHAR_A)
        self.index(OWNER_A, CHAR_A)
        self.assertGreater(self.vector_count(), 40)
        results = self.searcher.search(OWNER_A, CHAR_A, "apple", limit=2)
        self.assertEqual({r["memory_id"] for r in results}, {low1, low2})

    def test_superseded_old_version_excluded_before_worker_deletes_it(self):
        old = self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        src = self.store.append_message(OWNER_A, CHAR_A, "src-fix", "user", "fix chat")
        new = self.store.correct_memory(OWNER_A, CHAR_A, old, "ext-fix", "banana orchard", [src])
        # Worker has not run: old vector still physically present, new active vector missing.
        self.assertIsNotNone(self.vector_blob(old))
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_A, CHAR_A, "apple")
        self.index(OWNER_A, CHAR_A)
        self.assertIsNone(self.vector_blob(old))
        self.assertIsNotNone(self.vector_blob(new))
        results = self.searcher.search(OWNER_A, CHAR_A, "banana")
        self.assertEqual([r["memory_id"] for r in results], [new])
        self.assertEqual(results[0]["text"], "banana orchard")

    def test_deleted_memory_excluded_before_worker_deletes_vector(self):
        mid = self.remember(OWNER_A, CHAR_A, "apple orchard")
        keep = self.remember(OWNER_A, CHAR_A, "apple banana")
        self.index(OWNER_A, CHAR_A)
        self.assertTrue(self.store.delete_memory(OWNER_A, CHAR_A, mid))
        self.assertIsNotNone(self.vector_blob(mid))
        results = self.searcher.search(OWNER_A, CHAR_A, "apple")
        self.assertEqual([r["memory_id"] for r in results], [keep])
        self.index(OWNER_A, CHAR_A)
        self.assertIsNone(self.vector_blob(mid))

    def test_acl_revoked_after_indexing_not_served(self):
        shared = self.remember(OWNER_A, CHAR_A, "apple vault", visible_to=[CHAR_B])
        self.index(OWNER_A, CHAR_A)
        self.assertEqual(len(self.searcher.search(OWNER_A, CHAR_B, "apple")), 1)
        self.sql(
            "DELETE FROM memory_acl WHERE memory_id = ? AND character_id = ?", (shared, CHAR_B)
        )
        self.assertEqual(self.searcher.search(OWNER_A, CHAR_B, "apple"), [])
        self.assertEqual(
            [r["memory_id"] for r in self.searcher.search(OWNER_A, CHAR_A, "apple")], [shared]
        )

    def test_acl_granted_after_indexing_is_honored_from_canonical_state(self):
        mid = self.remember(OWNER_A, CHAR_A, "apple vault")
        self.index(OWNER_A, CHAR_A)
        self.assertEqual(self.searcher.search(OWNER_A, CHAR_B, "apple"), [])
        self.sql(
            "INSERT INTO memory_acl (memory_id, owner_id, character_id) VALUES (?, ?, ?)",
            (mid, OWNER_A, CHAR_B),
        )
        self.assertEqual(
            [r["memory_id"] for r in self.searcher.search(OWNER_A, CHAR_B, "apple")], [mid]
        )

    def test_canonical_get_memory_is_rechecked_for_every_result(self):
        mids = [self.remember(OWNER_A, CHAR_A, "apple " * n + "banana") for n in (3, 2, 1)]
        self.index(OWNER_A, CHAR_A)
        real = self.store.get_memory
        seen: list[str] = []

        def spy(owner, char, mid):
            seen.append(mid)
            return real(owner, char, mid)

        with mock.patch.object(self.store, "get_memory", side_effect=spy):
            results = self.searcher.search(OWNER_A, CHAR_A, "apple", limit=2)
        self.assertEqual(seen, [r["memory_id"] for r in results])
        self.assertEqual(len(results), 2)
        self.assertEqual([r["memory_id"] for r in results], mids[:2])

    def test_revocation_between_ranking_and_return_is_skipped(self):
        mids = [self.remember(OWNER_A, CHAR_A, "apple " * n + "banana") for n in (3, 2, 1)]
        self.index(OWNER_A, CHAR_A)
        real = self.store.get_memory
        state = {"first": True}

        def racing(owner, char, mid):
            if state["first"]:
                state["first"] = False
                self.store.delete_memory(owner, char, mid)
            return real(owner, char, mid)

        with mock.patch.object(self.store, "get_memory", side_effect=racing):
            results = self.searcher.search(OWNER_A, CHAR_A, "apple", limit=2)
        self.assertEqual([r["memory_id"] for r in results], mids[1:])

    def test_get_memory_failure_or_mismatch_handled_closed(self):
        mid = self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        with mock.patch.object(self.store, "get_memory", side_effect=RuntimeError("SECRET-DB")):
            with self.assertRaises(SemanticSearchError) as ctx:
                self.searcher.search(OWNER_A, CHAR_A, "apple")
        self.assertNotIn("SECRET-DB", str(ctx.exception))
        wrong_owner = {"memory_id": mid, "owner_id": OWNER_B, "domain": "real",
                       "creator_character_id": CHAR_A, "visible_to": [CHAR_A]}
        with mock.patch.object(self.store, "get_memory", return_value=wrong_owner):
            self.assertEqual(self.searcher.search(OWNER_A, CHAR_A, "apple"), [])

    def test_returned_records_are_independent_copies(self):
        self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        first = self.searcher.search(OWNER_A, CHAR_A, "apple")
        first[0]["text"] = "mutated"
        self.assertEqual(self.searcher.search(OWNER_A, CHAR_A, "apple")[0]["text"], "apple orchard")


class SearchFailClosedTests(SQLiteVectorTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.top = self.remember(OWNER_A, CHAR_A, "apple apple apple")
        self.low = self.remember(OWNER_A, CHAR_A, "cherry dragon")
        self.index(OWNER_A, CHAR_A)
        self.assertEqual(len(self.searcher.search(OWNER_A, CHAR_A, "apple")), 2)

    def test_corrupt_blob_raises_and_no_partial_results(self):
        original = self.vector_blob(self.low)
        corruptions = {
            "short": b"\x00\x01",
            "long": original + b"\x00\x00\x00\x00",
            "empty": b"",
            "nan": struct.pack(f"<{DIM}f", *([float("nan")] + [1.0] * (DIM - 1))),
            "inf": struct.pack(f"<{DIM}f", *([float("inf")] + [1.0] * (DIM - 1))),
            "zero": struct.pack(f"<{DIM}f", *([0.0] * DIM)),
            "text": "not-a-blob",
            "integer": 7,
        }
        for name, value in corruptions.items():
            with self.subTest(name=name):
                self.sql("UPDATE memory_vectors SET vector = ? WHERE memory_id = ?", (value, self.low))
                with self.assertRaises(SemanticSearchError):
                    self.searcher.search(OWNER_A, CHAR_A, "apple")
        self.sql("UPDATE memory_vectors SET vector = ? WHERE memory_id = ?", (original, self.low))
        self.assertEqual(len(self.searcher.search(OWNER_A, CHAR_A, "apple")), 2)

    def test_corruption_in_unauthorized_rows_does_not_affect_other_readers(self):
        other = self.remember(OWNER_B, CHAR_B, "apple harbor")
        self.index(OWNER_B, CHAR_B)
        self.sql("UPDATE memory_vectors SET vector = x'00' WHERE memory_id = ?", (other,))
        self.assertEqual(len(self.searcher.search(OWNER_A, CHAR_A, "apple")), 2)
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_B, CHAR_B, "apple")

    def test_row_model_or_dimension_mismatch_raises(self):
        for column, value in (("model_id", "other-model"), ("dimension", DIM + 1)):
            with self.subTest(column=column):
                original = self.sql(
                    f"SELECT {column} FROM memory_vectors WHERE memory_id = ?", (self.low,)
                )[0][0]
                self.sql(f"UPDATE memory_vectors SET {column} = ? WHERE memory_id = ?", (value, self.low))
                with self.assertRaises(SemanticSearchError):
                    self.searcher.search(OWNER_A, CHAR_A, "apple")
                self.sql(f"UPDATE memory_vectors SET {column} = ? WHERE memory_id = ?", (original, self.low))
        self.assertEqual(len(self.searcher.search(OWNER_A, CHAR_A, "apple")), 2)

    def test_orphaned_vector_rows_are_ignored(self):
        self.sql(
            "INSERT INTO memory_vectors (memory_id, model_id, dimension, vector, updated_at) "
            "VALUES ('not-a-uuid', ?, ?, ?, ?)",
            (MODEL_ID, DIM, self.vector_blob(self.low), "2026-11-12T00:00:00Z"),
        )
        # Orphaned rows do not join to canonical memories, so they are never candidates.
        self.assertEqual(len(self.searcher.search(OWNER_A, CHAR_A, "apple")), 2)

    def test_renamed_vector_table_raises_search_error(self):
        self.sql("ALTER TABLE memory_vectors RENAME TO memory_vectors_old")
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_A, CHAR_A, "apple")

    def test_dropped_or_missing_config_raises_search_error(self):
        self.sql("DELETE FROM memory_vector_config")
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_A, CHAR_A, "apple")

    def test_dropped_config_table_raises_search_error(self):
        self.sql("DROP TABLE memory_vector_config")
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_A, CHAR_A, "apple")

    def test_tampered_config_raises_search_error(self):
        self.sql("UPDATE memory_vector_config SET dimension = ?", (DIM + 1,))
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_A, CHAR_A, "apple")

    def test_missing_canonical_table_raises_search_error(self):
        self.sql("ALTER TABLE memory_acl RENAME TO memory_acl_old")
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_A, CHAR_A, "apple")

    def test_missing_dependencies_raise_not_empty(self):
        self.assertRaises(
            SemanticSearchError,
            SQLiteSemanticSearcher(self.store, None, self.backend).search,
            OWNER_A, CHAR_A, "apple",
        )
        self.assertRaises(
            SemanticSearchError,
            SQLiteSemanticSearcher(self.store, self.embedder, None).search,
            OWNER_A, CHAR_A, "apple",
        )
        self.assertRaises(
            SemanticSearchError,
            SQLiteSemanticSearcher(self.store, object(), self.backend).search,
            OWNER_A, CHAR_A, "apple",
        )

    def test_embedder_configuration_mismatch_raises(self):
        wrong_model = FnEmbedder(lambda t: ToyEmbedder().embed(t), model_id="other-model")
        wrong_dim = FnEmbedder(lambda t: [1.0, 0.0, 0.0], dimension=3)
        bad_meta = FnEmbedder(lambda t: ToyEmbedder().embed(t), dimension=True)
        bad_model_meta = FnEmbedder(lambda t: ToyEmbedder().embed(t), model_id="")
        for name, embedder in (
            ("model", wrong_model), ("dim", wrong_dim),
            ("bad_dim_meta", bad_meta), ("bad_model_meta", bad_model_meta),
        ):
            with self.subTest(name=name):
                searcher = SQLiteSemanticSearcher(self.store, embedder, self.backend)
                with self.assertRaises(SemanticSearchError):
                    searcher.search(OWNER_A, CHAR_A, "apple")

    def test_embedder_failures_are_sanitized(self):
        def boom(text):
            raise RuntimeError(f"leak:{text}")

        good = ToyEmbedder().embed("apple")
        cases = {
            "raises": boom,
            "nan": lambda t: [float("nan")] + good[1:],
            "inf": lambda t: [float("inf")] + good[1:],
            "zero": lambda t: [0.0] * DIM,
            "short": lambda t: good[:-1],
            "bool": lambda t: [True] + good[1:],
            "none": lambda t: None,
            "string": lambda t: "abc",
            "overflow": lambda t: [1e39] + good[1:],
        }
        for name, fn in cases.items():
            with self.subTest(name=name):
                searcher = SQLiteSemanticSearcher(self.store, FnEmbedder(fn), self.backend)
                with self.assertRaises(SemanticSearchError) as ctx:
                    searcher.search(OWNER_A, CHAR_A, "zzsecretzz apple")
                self.assertNotIn("zzsecretzz", str(ctx.exception))

    def test_search_errors_never_include_query_or_vectors(self):
        self.sql("ALTER TABLE memory_vectors RENAME TO memory_vectors_old")
        with self.assertRaises(SemanticSearchError) as ctx:
            self.searcher.search(OWNER_A, CHAR_A, "zzsecretzz apple")
        self.assertNotIn("zzsecretzz", str(ctx.exception))
        self.assertIsNone(ctx.exception.__cause__)


class SearcherConstructionTests(SQLiteVectorTestCase):
    def test_rejects_invalid_store(self):
        with self.assertRaises(ValueError):
            SQLiteSemanticSearcher(None, self.embedder, self.backend)
        with self.assertRaises(ValueError):
            SQLiteSemanticSearcher(object(), self.embedder, self.backend)

    def test_rejects_backend_for_different_store_or_type(self):
        other_store = MemoryStore(os.path.join(self._tmp.name, "other.db"))
        other_backend = SQLiteVectorBackend(other_store, DIM, MODEL_ID)
        with self.assertRaises(ValueError):
            SQLiteSemanticSearcher(self.store, self.embedder, other_backend)
        with self.assertRaises(ValueError):
            SQLiteSemanticSearcher(self.store, self.embedder, object())


class SearchReadinessTests(SQLiteVectorTestCase):
    def test_pending_missing_vector_raises_semantic_search_error(self):
        # 1 active authorized canonical memory, 1 pending upsert, 0 vectors.
        self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.assertEqual(self.vector_count(), 0)
        self.assertEqual(len(self.store.pending_jobs(OWNER_A, CHAR_A)), 1)
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_A, CHAR_A, "apple")

    def test_acked_missing_vector_raises_semantic_search_error(self):
        # Active canonical memory was ACKed and indexed, but vector row was deleted.
        mid = self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        self.assertEqual(self.vector_count(), 1)
        # Delete the ACKed vector row directly from vector storage.
        self.sql("DELETE FROM memory_vectors WHERE memory_id = ?", (mid,))
        self.assertEqual(self.vector_count(), 0)
        with self.assertRaises(SemanticSearchError):
            self.searcher.search(OWNER_A, CHAR_A, "apple")

    def test_valid_empty_scope_with_unrelated_pending_jobs(self):
        # Unrelated owner and private character have pending unindexed memories.
        self.remember(OWNER_B, CHAR_B, "apple orchard")
        self.remember(OWNER_A, CHAR_B, "apple banana")
        # Searching character has truly zero active memories.
        self.assertEqual(self.searcher.search(OWNER_A, CHAR_A, "apple"), [])

    def test_valid_empty_scope_with_domain_filter(self):
        # Memory exists and is indexed in 'roleplay', but searching in 'real' domain.
        mid = self.remember(OWNER_A, CHAR_A, "apple orchard", domain="roleplay")
        self.index(OWNER_A, CHAR_A)
        self.assertEqual(self.vector_count(), 1)
        # In roleplay domain, it is found.
        self.assertEqual(
            [r["memory_id"] for r in self.searcher.search(OWNER_A, CHAR_A, "apple", domain="roleplay")],
            [mid],
        )
        # In real domain, scope has 0 active memories -> returns healthy [].
        self.assertEqual(self.searcher.search(OWNER_A, CHAR_A, "apple", domain="real"), [])

    def test_unauthorized_only_missing_vectors_do_not_block_authorized_search(self):
        # OWNER_B and CHAR_B have unindexed active memories with missing vectors.
        self.remember(OWNER_B, CHAR_A, "apple pie")
        self.remember(OWNER_A, CHAR_B, "apple tart")
        # Searching character OWNER_A, CHAR_A has an indexed active memory.
        mine = self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        # Unauthorized missing vectors must not block OWNER_A, CHAR_A's search.
        results = self.searcher.search(OWNER_A, CHAR_A, "apple")
        self.assertEqual([r["memory_id"] for r in results], [mine])

    def test_superseded_old_vector_pending_deletion_does_not_block_when_current_vector_exists(self):
        # Old memory is superseded by new memory; new memory is indexed.
        old = self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        src = self.store.append_message(OWNER_A, CHAR_A, "src-s", "user", "update")
        new = self.store.correct_memory(OWNER_A, CHAR_A, old, "ext-s", "apple banana", [src])
        self.index(OWNER_A, CHAR_A)
        # Re-insert old superseded vector row to simulate pending deletion of old vector.
        self.sql(
            "INSERT INTO memory_vectors (memory_id, model_id, dimension, vector, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (old, MODEL_ID, DIM, self.vector_blob(new), "2026-11-12T00:00:00Z"),
        )
        # Superseded vector row does not block search when current active vector exists.
        results = self.searcher.search(OWNER_A, CHAR_A, "apple")
        self.assertEqual([r["memory_id"] for r in results], [new])

    def test_config_check_in_same_connection_snapshot_regression(self):
        # Regression for TOCTOU: verify config check happens in same connection as ranking.
        self.remember(OWNER_A, CHAR_A, "apple orchard")
        self.index(OWNER_A, CHAR_A)
        store2 = MemoryStore(self.db_path)
        orig_embed = self.embedder.embed

        def tampering_embed(text: str) -> list[float]:
            with store2._connect() as conn2:
                conn2.execute("UPDATE memory_vector_config SET dimension = ?", (DIM + 1,))
            return orig_embed(text)

        with mock.patch.object(self.embedder, "embed", side_effect=tampering_embed):
            with self.assertRaises(SemanticSearchError):
                self.searcher.search(OWNER_A, CHAR_A, "apple")

    def test_raced_delete_empty_or_invalid_source_ids_skipped(self):
        m1 = self.remember(OWNER_A, CHAR_A, "apple apple", sources=2)
        m2 = self.remember(OWNER_A, CHAR_A, "apple banana", sources=1)
        self.index(OWNER_A, CHAR_A)
        real_get = self.store.get_memory

        def racing_get(owner, char, mid):
            rec = real_get(owner, char, mid)
            if rec and mid == m1:
                # Simulate race with delete where sources were cleared.
                rec = dict(rec, source_ids=[])
            return rec

        with mock.patch.object(self.store, "get_memory", side_effect=racing_get):
            results = self.searcher.search(OWNER_A, CHAR_A, "apple")
            # m1 was skipped due to empty source_ids; m2 was returned.
            self.assertEqual([r["memory_id"] for r in results], [m2])
            self.assertEqual(results[0]["source_ids"], real_get(OWNER_A, CHAR_A, m2)["source_ids"])

    def test_search_rejects_caller_transaction_without_rollback_regression(self):
        store = MemoryStore(":memory:")
        try:
            embedder = ToyEmbedder()
            backend = SQLiteVectorBackend(store, DIM, MODEL_ID)
            searcher = SQLiteSemanticSearcher(store, embedder, backend)
            with store._connect() as conn:
                conn.execute("BEGIN")
                conn.execute(
                    "INSERT INTO memory_key_blocks (owner_id, character_id, meta_key, created_at) "
                    "VALUES (?, ?, ?, ?)",
                    (OWNER_A, CHAR_A, "fictional-key", "2026-11-12T00:00:00Z"),
                )
                with self.assertRaises(SemanticSearchError):
                    searcher.search(OWNER_A, CHAR_A, "apple")
                self.assertTrue(conn.in_transaction)
                row = conn.execute(
                    "SELECT 1 FROM memory_key_blocks "
                    "WHERE owner_id = ? AND character_id = ? AND meta_key = ?",
                    (OWNER_A, CHAR_A, "fictional-key"),
                ).fetchone()
                self.assertIsNotNone(row)
                conn.execute("ROLLBACK")
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
