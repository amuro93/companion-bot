"""Tests for durable deferred-memory queue storage (TASK A-5a).

Standard-library-only test suite covering lifecycle, concurrency, scope isolation,
idle debounce, retry backoff, fault recovery, and caller transaction preservation.
"""

from __future__ import annotations

import math
import os
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from memory_core import MemoryStore, _validate_id
from memory_deferred_queue import (
    DeferredMemoryQueue,
    DeferredQueueError,
    VALID_ERROR_CATEGORIES,
)
from memory_trigger import message_external_id


class FakeClock:
    """Deterministic injectable clock with zero sleeps."""

    def __init__(self, initial_time: float = 1000.0) -> None:
        self.current_time = float(initial_time)

    def __call__(self) -> float:
        return self.current_time

    def advance(self, seconds: float) -> None:
        self.current_time += float(seconds)


class TestDeferredMemoryQueue(unittest.TestCase):
    def setUp(self) -> None:
        self.owner_id = "user-test-1"
        self.character_id = "char-test-1"
        self.clock = FakeClock(1000.0)

    def test_init_new_scope_no_historical_replay(self) -> None:
        """Historical source messages existing prior to queue registration must NOT be queued."""
        store = MemoryStore(":memory:")
        try:
            # Insert historical messages prior to queue creation
            store.append_message(
                self.owner_id,
                self.character_id,
                "tg-hist-1",
                "user",
                "Historical utterance 1",
            )
            store.append_message(
                self.owner_id,
                self.character_id,
                "tg-hist-2",
                "user",
                "Historical utterance 2",
            )
            store.append_message(
                self.owner_id,
                self.character_id,
                "asst-hist-3",
                "assistant",
                "Historical assistant reply",
            )

            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=180,
                clock=self.clock,
            )

            st = queue.status()
            self.assertEqual(st["pending_count"], 0)
            self.assertEqual(st["processing_count"], 0)
            self.assertEqual(st["done_count"], 0)
            self.assertEqual(st["excluded_count"], 0)

            self.clock.advance(300)
            self.assertIsNone(queue.claim_due())
        finally:
            store.close()

    def test_init_existing_scope_preserves_cursor_and_jobs(self) -> None:
        """Existing scope preserves cursor, activity, and jobs across restarts."""
        store = MemoryStore(":memory:")
        try:
            q1 = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=180,
                clock=self.clock,
            )
            msg1 = q1.capture("First sentence for queue")
            self.clock.advance(200)
            claim1 = q1.claim_due()
            self.assertIsNotNone(claim1)
            self.assertTrue(q1.complete(msg1, claim1["claim_token"]))

            msg2 = q1.capture("Second sentence for queue")
            status1 = q1.status()
            self.assertEqual(status1["pending_count"], 1)
            self.assertEqual(status1["done_count"], 1)

            # Re-initialize on same store and scope
            q2 = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=180,
                clock=self.clock,
            )
            status2 = q2.status()
            self.assertEqual(status2["pending_count"], 1)
            self.assertEqual(status2["done_count"], 1)
            self.assertEqual(status2["last_success"], status1["last_success"])
            self.assertEqual(status2["last_activity"], status1["last_activity"])
        finally:
            store.close()

    def test_capture_and_full_lifecycle(self) -> None:
        """Capture -> claim_due -> complete lifecycle with status counters."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=180,
                clock=self.clock,
            )
            content = "我最喜歡喝無糖烏龍茶"
            msg_id = queue.capture(content)
            self.assertTrue(isinstance(msg_id, str) and len(msg_id) > 0)

            # Immediately after capture, idle_seconds (180s) has not passed
            self.assertIsNone(queue.claim_due(lease_seconds=300))
            self.assertEqual(queue.status()["pending_count"], 1)
            self.assertEqual(queue.status()["processing_count"], 0)

            # Advance clock past idle
            self.clock.advance(181)
            claimed = queue.claim_due(lease_seconds=300)
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["message_id"], msg_id)
            self.assertEqual(claimed["content"], content)
            self.assertEqual(claimed["attempts"], 1)
            self.assertTrue(isinstance(claimed["claim_token"], str))

            # Status reflects processing
            st = queue.status()
            self.assertEqual(st["pending_count"], 1)
            self.assertEqual(st["processing_count"], 1)

            # Complete with claim token
            ok = queue.complete(msg_id, claimed["claim_token"])
            self.assertTrue(ok)

            st = queue.status()
            self.assertEqual(st["pending_count"], 0)
            self.assertEqual(st["processing_count"], 0)
            self.assertEqual(st["done_count"], 1)
            self.assertEqual(st["last_success"], self.clock.current_time)
            self.assertIsNone(st["last_error_category"])
        finally:
            store.close()

    def test_idle_debounce_and_duplicate_capture_postpones_claim(self) -> None:
        """Duplicate capture postpones due claims and does not requeue completed work."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=180,
                clock=self.clock,
            )
            msg1 = queue.capture("我每天都喝黑咖啡")
            self.clock.advance(150)
            # Before 180s elapsed, duplicate capture of identical content
            msg1_again = queue.capture("我每天都喝黑咖啡")
            self.assertEqual(msg1, msg1_again)

            # 40s later (total 190s from first capture, but only 40s from second capture)
            self.clock.advance(40)
            self.assertIsNone(queue.claim_due())

            # Now advance past 180s from second capture (150 + 40 + 150 = 340s)
            self.clock.advance(150)
            claimed = queue.claim_due()
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["message_id"], msg1)
            self.assertTrue(queue.complete(msg1, claimed["claim_token"]))

            # Now that it's done, duplicate capture updates activity but does NOT requeue
            queue.capture("我每天都喝黑咖啡")
            st = queue.status()
            self.assertEqual(st["done_count"], 1)
            self.assertEqual(st["pending_count"], 0)
            self.clock.advance(300)
            self.assertIsNone(queue.claim_due())
        finally:
            store.close()

    def test_assistant_and_manual_sources_filtered(self) -> None:
        """Only user messages with external_id starting with 'tg-' are queued."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )

            # Append assistant message directly
            store.append_message(
                self.owner_id,
                self.character_id,
                "tg-asst-1",
                "assistant",
                "I am an assistant",
            )
            # Append manual user message directly
            store.append_message(
                self.owner_id,
                self.character_id,
                "manual-user-1",
                "user",
                "Manual message",
            )
            # Append valid user message via capture
            valid_id = queue.capture("Valid user message")

            self.clock.advance(100)
            claimed = queue.claim_due()
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["message_id"], valid_id)
            self.assertTrue(queue.complete(valid_id, claimed["claim_token"]))

            # No other jobs should exist
            self.assertIsNone(queue.claim_due())
            st = queue.status()
            self.assertEqual(st["done_count"], 1)
            self.assertEqual(st["pending_count"], 0)
        finally:
            store.close()

    def test_crash_after_canonical_append_recovered_on_restart(self) -> None:
        """If crash occurs after canonical append before jobs, startup/recover finds it."""
        store = MemoryStore(":memory:")
        try:
            q1 = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )

            # Directly append canonical user message simulating a crash before queue insert
            ext_id = message_external_id("Unrecovered crash message")
            msg_id = store.append_message(
                self.owner_id,
                self.character_id,
                ext_id,
                "user",
                "Unrecovered crash message",
            )

            # New queue instance on restart
            q2 = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )
            st = q2.status()
            self.assertEqual(st["pending_count"], 1)

            self.clock.advance(100)
            claimed = q2.claim_due()
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["message_id"], msg_id)
            self.assertEqual(claimed["content"], "Unrecovered crash message")
            self.assertTrue(q2.complete(msg_id, claimed["claim_token"]))
        finally:
            store.close()

    def test_source_exclusion_atomically_marks_excluded(self) -> None:
        """Excluded source message marks job excluded without returning content, proceeding to next."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )
            msg1 = queue.capture("Message that will be excluded")
            msg2 = queue.capture("Message that stays active")

            # Mark msg1 excluded in canonical messages table
            with store._connect() as conn:
                conn.execute(
                    "UPDATE messages SET excluded_from_memory = 1 WHERE message_id = ?",
                    (msg1,),
                )

            self.clock.advance(100)
            # claim_due should atomically mark msg1 excluded and claim msg2
            claimed = queue.claim_due()
            self.assertIsNotNone(claimed)
            self.assertEqual(claimed["message_id"], msg2)
            self.assertEqual(claimed["content"], "Message that stays active")

            st = queue.status()
            self.assertEqual(st["excluded_count"], 1)
            self.assertEqual(st["pending_count"], 1)
            self.assertEqual(st["processing_count"], 1)

            self.assertTrue(queue.complete(msg2, claimed["claim_token"]))
            st = queue.status()
            self.assertEqual(st["done_count"], 1)
            self.assertEqual(st["excluded_count"], 1)
            self.assertEqual(st["pending_count"], 0)
        finally:
            store.close()

    def test_earliest_ordering_and_no_overtake(self) -> None:
        """Head job in processing state with unexpired lease prevents later jobs from overtaking."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )
            msg1 = queue.capture("Head message 1")
            msg2 = queue.capture("Tail message 2")

            self.clock.advance(100)
            claimed1 = queue.claim_due(lease_seconds=300)
            self.assertIsNotNone(claimed1)
            self.assertEqual(claimed1["message_id"], msg1)

            # Second call must NOT overtake msg1 to claim msg2
            self.assertIsNone(queue.claim_due(lease_seconds=300))

            # Complete msg1
            self.assertTrue(queue.complete(msg1, claimed1["claim_token"]))

            # Now msg2 can be claimed
            claimed2 = queue.claim_due(lease_seconds=300)
            self.assertIsNotNone(claimed2)
            self.assertEqual(claimed2["message_id"], msg2)
            self.assertTrue(queue.complete(msg2, claimed2["claim_token"]))
        finally:
            store.close()

    def test_lease_expiry_and_reclaim_with_stale_token_rejection(self) -> None:
        """Expired lease allows reclaim with new token; stale token cannot complete or fail."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )
            msg1 = queue.capture("Leased message")
            self.clock.advance(100)

            claim1 = queue.claim_due(lease_seconds=200)
            self.assertIsNotNone(claim1)
            token1 = claim1["claim_token"]
            self.assertEqual(claim1["attempts"], 1)

            # Lease still active at 150s
            self.clock.advance(150)
            self.assertIsNone(queue.claim_due(lease_seconds=200))

            # Advance past lease expiry (total 250s since claim)
            self.clock.advance(100)
            claim2 = queue.claim_due(lease_seconds=200)
            self.assertIsNotNone(claim2)
            token2 = claim2["claim_token"]
            self.assertNotEqual(token1, token2)
            self.assertEqual(claim2["attempts"], 2)

            # Stale token1 cannot complete or fail
            self.assertFalse(queue.complete(msg1, token1))
            self.assertFalse(queue.fail(msg1, token1, "processor_failed"))

            # Token2 completes successfully
            self.assertTrue(queue.complete(msg1, token2))
            st = queue.status()
            self.assertEqual(st["done_count"], 1)
        finally:
            store.close()

    def test_ack_after_lease_time_before_reclaim_succeeds(self) -> None:
        """ACK after lease duration but before any competing reclaim succeeds."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )
            msg1 = queue.capture("Message to ACK late")
            self.clock.advance(100)
            claim1 = queue.claim_due(lease_seconds=100)
            self.assertIsNotNone(claim1)

            # Advance past lease expiry, but no one reclaims it
            self.clock.advance(200)
            ok = queue.complete(msg1, claim1["claim_token"])
            self.assertTrue(ok)
            self.assertEqual(queue.status()["done_count"], 1)
        finally:
            store.close()

    def test_fail_whitelist_categories_and_retry_backoff(self) -> None:
        """fail() stores only whitelist category and enforces retry_seconds backoff without silent loss."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )
            msg1 = queue.capture("Message that will fail once")
            self.clock.advance(100)

            claim1 = queue.claim_due(lease_seconds=300)
            self.assertIsNotNone(claim1)

            # Invalid category rejected without echoing
            with self.assertRaises(ValueError):
                queue.fail(msg1, claim1["claim_token"], "invalid_custom_category")

            # Valid category fail
            ok = queue.fail(
                msg1,
                claim1["claim_token"],
                error_category="processor_failed",
                retry_seconds=50,
            )
            self.assertTrue(ok)

            st = queue.status()
            self.assertEqual(st["last_error_category"], "processor_failed")
            self.assertEqual(st["pending_count"], 1)
            self.assertEqual(st["processing_count"], 0)

            # Before retry_seconds elapses, claim_due returns None
            self.clock.advance(30)
            self.assertIsNone(queue.claim_due())

            # After retry_seconds elapses, claim succeeds with incremented attempts
            self.clock.advance(25)
            claim2 = queue.claim_due()
            self.assertIsNotNone(claim2)
            self.assertEqual(claim2["attempts"], 2)

            # Complete clears last_error_category
            self.assertTrue(queue.complete(msg1, claim2["claim_token"]))
            st = queue.status()
            self.assertIsNone(st["last_error_category"])
            self.assertEqual(st["done_count"], 1)
        finally:
            store.close()

    def test_caller_transaction_preserved_and_rejected(self) -> None:
        """Caller transaction is rejected before BEGIN and caller data is never rolled back."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )

            with store._connect() as conn:
                conn.execute("BEGIN")
                conn.execute(
                    """
                    INSERT INTO messages (
                        message_id, owner_id, character_id, external_id,
                        role, content, occurred_at, created_at, payload_hash
                    ) VALUES ('caller-msg-1', 'user-test-1', 'char-test-1', 'ext-c-1',
                              'user', 'Caller message', '2026-01-01T00:00:00.000000Z',
                              '2026-01-01T00:00:00.000000Z', 'hash123')
                    """
                )

                # Queue operations must reject with DeferredQueueError
                with self.assertRaises(DeferredQueueError):
                    queue.claim_due()
                with self.assertRaises(DeferredQueueError):
                    queue.capture("Attempted capture during transaction")
                with self.assertRaises(DeferredQueueError):
                    queue.status()
                with self.assertRaises(DeferredQueueError):
                    queue.recover()

                # Caller transaction must still be active and intact!
                self.assertTrue(conn.in_transaction)
                conn.execute("COMMIT")

            # Verify caller data was safely committed
            with store._connect() as conn:
                row = conn.execute(
                    "SELECT content FROM messages WHERE message_id = 'caller-msg-1'"
                ).fetchone()
                self.assertIsNotNone(row)
                self.assertEqual(row["content"], "Caller message")
        finally:
            store.close()

    def test_scope_isolation_between_different_owners_and_characters(self) -> None:
        """Scopes are completely isolated; one scope's work is never visible or claimable by another."""
        store = MemoryStore(":memory:")
        try:
            q_alice = DeferredMemoryQueue(
                store,
                "alice",
                "char-1",
                idle_seconds=60,
                clock=self.clock,
            )
            q_bob = DeferredMemoryQueue(
                store,
                "bob",
                "char-1",
                idle_seconds=60,
                clock=self.clock,
            )
            q_alice_char2 = DeferredMemoryQueue(
                store,
                "alice",
                "char-2",
                idle_seconds=60,
                clock=self.clock,
            )

            msg_alice = q_alice.capture("Alice message")
            self.clock.advance(100)

            # Bob and Alice-char2 see nothing pending
            self.assertEqual(q_bob.status()["pending_count"], 0)
            self.assertEqual(q_alice_char2.status()["pending_count"], 0)
            self.assertIsNone(q_bob.claim_due())
            self.assertIsNone(q_alice_char2.claim_due())

            # Alice claims and completes
            claim_alice = q_alice.claim_due()
            self.assertIsNotNone(claim_alice)
            self.assertEqual(claim_alice["message_id"], msg_alice)

            # Bob cannot complete Alice's message
            self.assertFalse(q_bob.complete(msg_alice, claim_alice["claim_token"]))

            self.assertTrue(q_alice.complete(msg_alice, claim_alice["claim_token"]))
            self.assertEqual(q_alice.status()["done_count"], 1)
            self.assertEqual(q_bob.status()["done_count"], 0)
        finally:
            store.close()

    def test_input_and_clock_validation(self) -> None:
        """Validate parameter boundaries, negative/bool inputs, and clock error handling."""
        store = MemoryStore(":memory:")
        try:
            # Invalid idle_seconds
            with self.assertRaises(ValueError):
                DeferredMemoryQueue(store, self.owner_id, self.character_id, idle_seconds=-1)
            with self.assertRaises(ValueError):
                DeferredMemoryQueue(store, self.owner_id, self.character_id, idle_seconds=True)
            with self.assertRaises(ValueError):
                DeferredMemoryQueue(store, self.owner_id, self.character_id, idle_seconds=100000)

            # Invalid clock
            with self.assertRaises(ValueError):
                DeferredMemoryQueue(store, self.owner_id, self.character_id, clock="not_a_callable")

            # Clock returning non-finite
            def bad_clock():
                return float("nan")

            with self.assertRaises(DeferredQueueError):
                DeferredMemoryQueue(store, self.owner_id, self.character_id, clock=bad_clock)

            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )

            # Invalid capture text
            with self.assertRaises(ValueError):
                queue.capture("")
            with self.assertRaises(ValueError):
                queue.capture("x" * 1025)
            with self.assertRaises(ValueError):
                queue.capture("null\x00byte")

            # Invalid lease_seconds on claim_due
            with self.assertRaises(ValueError):
                queue.claim_due(lease_seconds=0)
            with self.assertRaises(ValueError):
                queue.claim_due(lease_seconds=True)
            with self.assertRaises(ValueError):
                queue.claim_due(lease_seconds=100000)

            # Invalid retry_seconds on fail
            with self.assertRaises(ValueError):
                queue.fail("any-id", "any-token", retry_seconds=-5)
            with self.assertRaises(ValueError):
                queue.fail("any-id", "any-token", retry_seconds=True)
        finally:
            store.close()

    def test_two_independent_store_instances_competing_on_disk(self) -> None:
        """Two independent Store instances pointing to the same disk DB coordinate via SQL locks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_queue.db")
            store1 = MemoryStore(db_path)
            store2 = MemoryStore(db_path)
            try:
                q1 = DeferredMemoryQueue(
                    store1,
                    self.owner_id,
                    self.character_id,
                    idle_seconds=60,
                    clock=self.clock,
                )
                q2 = DeferredMemoryQueue(
                    store2,
                    self.owner_id,
                    self.character_id,
                    idle_seconds=60,
                    clock=self.clock,
                )

                msg = q1.capture("Shared disk message")
                self.clock.advance(100)

                # q1 claims it
                claim1 = q1.claim_due(lease_seconds=300)
                self.assertIsNotNone(claim1)

                # q2 cannot claim the unexpired leased message
                claim2 = q2.claim_due(lease_seconds=300)
                self.assertIsNone(claim2)

                # q1 completes
                self.assertTrue(q1.complete(msg, claim1["claim_token"]))

                # q2 status observes the completion
                st2 = q2.status()
                self.assertEqual(st2["done_count"], 1)
                self.assertEqual(st2["pending_count"], 0)
            finally:
                store1.close()
                store2.close()

    def test_status_sanitization_no_content_or_tokens(self) -> None:
        """status() dictionary exposes only counters and metadata, never raw content or tokens."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )
            queue.capture("Secret message content")
            st = queue.status()

            expected_keys = {
                "pending_count",
                "processing_count",
                "excluded_count",
                "done_count",
                "last_activity",
                "last_success",
                "last_error_category",
            }
            self.assertEqual(set(st.keys()), expected_keys)
            for key, val in st.items():
                self.assertNotIn("Secret message content", str(val))
        finally:
            store.close()

    def test_missing_canonical_message_fails_loudly(self) -> None:
        """If canonical message row is deleted from store, claim_due fails loudly with DeferredQueueError."""
        store = MemoryStore(":memory:")
        try:
            queue = DeferredMemoryQueue(
                store,
                self.owner_id,
                self.character_id,
                idle_seconds=60,
                clock=self.clock,
            )
            msg_id = queue.capture("Corruptible message")

            # Remove canonical message from messages table
            with store._connect() as conn:
                conn.execute("DELETE FROM messages WHERE message_id = ?", (msg_id,))

            self.clock.advance(100)
            with self.assertRaises(DeferredQueueError):
                queue.claim_due()
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()
