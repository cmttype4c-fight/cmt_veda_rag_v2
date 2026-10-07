"""
test_intake_lifecycle.py
-------------------------
Round 4: real, executed tests against SqliteBackend + HashingTfidfEmbeddings
+ NumpyVectorStore + worker.InlineQueueAdapter (the same sandbox-executable
stack test_pipeline_e2e.py uses for round 3). Covers:

  - register_intake -> approve_intake (auto-queue) -> worker -> INDEXED,
    and that the result is actually retrievable (lexical_search).
  - Full-text-only rule: a too-short / abstract-like text is rejected.
  - Source-type allowlist / Newsletter isolation: a disallowed source_type
    is rejected before any document row is created.
  - Metadata PATCH: allowed while pending_approval, rejected after.
  - Retry policy: a pipeline that always fails ends up FAILED after
    exactly MAX_RETRIES attempts, with retry_count recorded, and does NOT
    retry a 6th time (no infinite loop).
  - Bulk register + bulk approve: one bad item among good ones is
    recorded as a per-item failure without blocking or mismarking the
    rest.

Run with: pytest test_intake_lifecycle.py -v
(or `python3 -I test_intake_lifecycle.py` — falls back to a plain runner
at the bottom, same convention as test_pipeline_e2e.py.)
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db import SqliteBackend
from embeddings import HashingTfidfEmbeddings, EmbeddingBackend
from vector_store import NumpyVectorStore
from ingestion_pipeline import IngestionInput
from config import IngestionState, MAX_RETRIES, IllegalStateTransitionError
import intake
from intake import (
    register_intake, approve_intake, patch_intake_metadata, reject_intake,
    bulk_register_intake, bulk_approve_intake,
    InvalidSourceType, FullTextRuleViolation, IntakeConflictError,
)
from worker import InlineQueueAdapter
from db import DuplicateDocumentError


FULL_TEXT = (
    "Abstract\nCharcot-Marie-Tooth (CMT) disease is a group of inherited "
    "peripheral neuropathies. " + ("This sentence pads the document to pass "
    "the full-text-only length floor so the test exercises real pipeline "
    "logic rather than being rejected at the door. " * 30)
)
assert len(FULL_TEXT) > 1000  # sanity-check the fixture itself


def _fresh_env():
    tmpdir = tempfile.mkdtemp(prefix="cmt_veda_intake_")
    db = SqliteBackend(os.path.join(tmpdir, "test.db"))
    embedder = HashingTfidfEmbeddings(n_features=512)
    vs = NumpyVectorStore(os.path.join(tmpdir, "vs", "store"), dim=512)
    adapter = InlineQueueAdapter(db, embedder, vs)
    return tmpdir, db, embedder, vs, adapter


def test_register_approve_process_indexed_and_retrievable():
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        payload = IngestionInput(raw_text=FULL_TEXT, format="text", title="CMT Overview",
                                  doi="10.1/test1")
        doc = register_intake(db, payload, "discovery", actor="discovery_engine")
        assert doc.approval_status == IngestionState.PENDING_APPROVAL
        assert doc.source_type == "discovery"
        assert doc.extracted_text == FULL_TEXT

        doc, job_id = approve_intake(db, doc.document_id, actor="admin1", enqueue_fn=adapter.enqueue_fn)
        assert doc.approval_status == IngestionState.QUEUED  # APPROVED passed through, now QUEUED
        adapter.run_pending()

        final = db.get_document(doc.document_id)
        assert final.approval_status == IngestionState.INDEXED, final.approval_status

        hits = db.lexical_search("Charcot-Marie-Tooth", limit=5)
        assert any(True for _ in hits), "indexed document should be lexically retrievable"

        job = db.get_latest_processing_job(doc.document_id)
        assert job["status"] == "finished"
        print("PASS: register -> approve (auto-queue) -> worker -> INDEXED, retrievable")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_full_text_only_rule_rejects_short_text():
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        # Long enough (>50 chars) to clear ingestion_pipeline's own pre-existing
        # "corrupted content" floor, but still well under MIN_FULL_TEXT_CHARS
        # (1000) — this isolates the round-4 full-text-only check from the
        # round-3 corrupted-content check so the test proves the right thing.
        abstract_only = (
            "Abstract: This is a brief abstract-only summary of the study "
            "with no full body text attached, standing in for an "
            "abstract-only record that must never enter the RAG corpus."
        )
        assert 50 < len(abstract_only) < intake.MIN_FULL_TEXT_CHARS
        payload = IngestionInput(raw_text=abstract_only, format="text", title="Short")
        try:
            register_intake(db, payload, "direct_upload", actor="admin1")
            assert False, "expected FullTextRuleViolation"
        except FullTextRuleViolation:
            pass
        assert db.list_documents_page() == []
        print("PASS: full-text-only rule rejects short/abstract-like text")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_disallowed_source_type_rejected_newsletter_isolation():
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        payload = IngestionInput(raw_text=FULL_TEXT, format="text", title="Should not enter RAG")
        for bad in ("newsletter", "editorial", "gemini_summary", ""):
            try:
                register_intake(db, payload, bad, actor="admin1")
                assert False, f"expected InvalidSourceType for source_type={bad!r}"
            except InvalidSourceType:
                pass
        assert db.list_documents_page() == [], "no document row should exist for a rejected source_type"
        print("PASS: disallowed source_type (incl. newsletter/editorial) rejected before any row is created")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_metadata_patch_allowed_pending_rejected_after_approval():
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        payload = IngestionInput(raw_text=FULL_TEXT, format="text", title="Wrong Title", doi="10.1/test2")
        doc = register_intake(db, payload, "direct_upload", actor="admin1", uploaded_by="alice")

        patched = patch_intake_metadata(db, doc.document_id, {"title": "Corrected Title"})
        assert patched.title == "Corrected Title"

        approve_intake(db, doc.document_id, actor="admin1", enqueue_fn=adapter.enqueue_fn)
        try:
            patch_intake_metadata(db, doc.document_id, {"title": "Too Late"})
            assert False, "expected IntakeConflictError after approval"
        except IntakeConflictError:
            pass
        print("PASS: metadata PATCH allowed pre-approval, locked post-approval")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


class _AlwaysFailsEmbedder(EmbeddingBackend):
    """Deterministic failure injection for the retry-policy test."""
    def dim(self):
        return 512

    def embed(self, texts):
        raise RuntimeError("simulated embedding failure for retry-policy test")


def test_retry_policy_fails_five_times_then_stays_failed():
    tmpdir, db, _, vs, _ = _fresh_env()
    try:
        bad_embedder = _AlwaysFailsEmbedder()
        adapter = InlineQueueAdapter(db, bad_embedder, vs)
        payload = IngestionInput(raw_text=FULL_TEXT, format="text", title="Will Fail", doi="10.1/fail1")
        doc = register_intake(db, payload, "direct_upload", actor="admin1")
        approve_intake(db, doc.document_id, actor="admin1", enqueue_fn=adapter.enqueue_fn)
        adapter.run_pending()  # drains the initial attempt + every retry it self-enqueues

        final = db.get_document(doc.document_id)
        assert final.approval_status == IngestionState.FAILED, final.approval_status
        assert final.retry_count == MAX_RETRIES, final.retry_count

        job = db.get_latest_processing_job(doc.document_id)
        assert job["attempt"] == MAX_RETRIES, job["attempt"]
        assert job["status"] == "failed"
        print(f"PASS: retry policy — FAILED after exactly {MAX_RETRIES} attempts, no further auto-retry")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_bulk_register_one_bad_item_does_not_block_others():
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        items = [
            {"source_type": "direct_upload", "payload": {"raw_text": FULL_TEXT, "format": "text",
                                                           "title": "Good One", "doi": "10.1/good1"}},
            {"source_type": "newsletter", "payload": {"raw_text": FULL_TEXT, "format": "text",
                                                        "title": "Bad: wrong source_type"}},
            {"source_type": "direct_upload", "payload": {"raw_text": "too short", "format": "text",
                                                           "title": "Bad: too short"}},
            {"source_type": "discovery", "payload": {"raw_text": FULL_TEXT, "format": "text",
                                                       "title": "Good Two", "doi": "10.1/good2"}},
        ]
        result = bulk_register_intake(db, items, actor="admin1")
        assert result["item_count"] == 4
        assert result["success_count"] == 2, result
        assert result["failure_count"] == 2, result
        statuses = [r["status"] for r in result["results"]]
        assert statuses == ["success", "failed", "failed", "success"], statuses

        items_in_batch = db.list_batch_items(result["batch_id"])
        assert len(items_in_batch) == 4
        print("PASS: bulk register — per-item success/failure, one bad item never blocks the rest")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_bulk_approve_mixed_results():
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        ids = []
        for i in range(3):
            payload = IngestionInput(raw_text=FULL_TEXT, format="text", title=f"Doc {i}", doi=f"10.1/bulk{i}")
            doc = register_intake(db, payload, "direct_upload", actor="admin1")
            ids.append(doc.document_id)
        ids.append("00000000-0000-0000-0000-000000000000")  # nonexistent -> forces a per-item failure

        result = bulk_approve_intake(db, ids, actor="admin1", enqueue_fn=adapter.enqueue_fn)
        assert result["success_count"] == 3, result
        assert result["failure_count"] == 1, result
        adapter.run_pending()

        for doc_id in ids[:3]:
            final = db.get_document(doc_id)
            assert final.approval_status == IngestionState.INDEXED, (doc_id, final.approval_status)
        print("PASS: bulk approve — mixed success/failure accounted per item, successes still indexed")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_reject_intake_removes_document():
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        payload = IngestionInput(raw_text=FULL_TEXT, format="text", title="To Reject", doi="10.1/reject1")
        doc = register_intake(db, payload, "direct_upload", actor="admin1")
        rejected = reject_intake(db, doc.document_id, actor="admin1", reason="not relevant")
        assert rejected.approval_status == IngestionState.REMOVED
        print("PASS: reject_intake transitions a pending document straight to REMOVED")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_queue_intake_enqueue_failure_does_not_strand_document_in_queued():
    """Proves the fix: if enqueue_fn itself raises (Redis/RQ unreachable,
    as it genuinely is in this sandbox), queue_intake() must not leave the
    document sitting in QUEUED with zero job record -- it should land in
    FAILED (the only other legal exit from QUEUED) with a job row an
    admin can inspect to see why."""
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        def _always_raises(document_id, attempt):
            raise ConnectionError("simulated: Redis unreachable")

        payload = IngestionInput(raw_text=FULL_TEXT, format="text", title="Enqueue Fails", doi="10.1/enq1")
        doc = register_intake(db, payload, "direct_upload", actor="admin1")
        try:
            approve_intake(db, doc.document_id, actor="admin1", enqueue_fn=_always_raises)
            assert False, "expected the enqueue failure to propagate"
        except ConnectionError:
            pass

        final = db.get_document(doc.document_id)
        assert final.approval_status == IngestionState.FAILED, final.approval_status
        assert final.retry_count == 1, final.retry_count

        job = db.get_latest_processing_job(doc.document_id)
        assert job is not None, "an enqueue failure must still leave an inspectable job record"
        assert job["status"] == "failed"
        assert "enqueue_fn failed" in (job["error"] or "")
        print("PASS: an enqueue_fn failure lands the document in FAILED (not stuck in QUEUED), with a job record")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_illegal_transition_is_a_distinct_error_type_not_generic_valueerror():
    """Proves the contract-doc fix: re-approving an already-INDEXED
    document (an illegal transition, not a missing document) raises
    IllegalStateTransitionError specifically -- a ValueError subclass,
    so existing `except ValueError` call sites are unaffected, but
    main.py's _intake_exc_to_http can (and now does) map THIS case to
    409 Conflict instead of the misleading 404 "no such document" a
    bare ValueError would have produced."""
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        payload = IngestionInput(raw_text=FULL_TEXT, format="text", title="Already Indexed", doi="10.1/illegal1")
        doc = register_intake(db, payload, "direct_upload", actor="admin1")
        approve_intake(db, doc.document_id, actor="admin1", enqueue_fn=adapter.enqueue_fn)
        adapter.run_pending()
        assert db.get_document(doc.document_id).approval_status == IngestionState.INDEXED

        try:
            approve_intake(db, doc.document_id, actor="admin1", enqueue_fn=adapter.enqueue_fn)
            assert False, "expected IllegalStateTransitionError re-approving an INDEXED document"
        except IllegalStateTransitionError:
            pass
        except ValueError:
            assert False, "raised a bare ValueError, not the specific IllegalStateTransitionError subclass"

        print("PASS: illegal transition (re-approving INDEXED) raises IllegalStateTransitionError, not a bare ValueError")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_duplicate_detection_still_enforced_at_register():
    tmpdir, db, embedder, vs, adapter = _fresh_env()
    try:
        payload = IngestionInput(raw_text=FULL_TEXT, format="text", title="Original", doi="10.1/dup1")
        register_intake(db, payload, "direct_upload", actor="admin1")
        try:
            register_intake(db, payload, "direct_upload", actor="admin1")
            assert False, "expected DuplicateDocumentError"
        except DuplicateDocumentError:
            pass
        print("PASS: duplicate detection (unchanged from round 3) still enforced in register_intake")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


_TESTS = [
    test_register_approve_process_indexed_and_retrievable,
    test_full_text_only_rule_rejects_short_text,
    test_disallowed_source_type_rejected_newsletter_isolation,
    test_metadata_patch_allowed_pending_rejected_after_approval,
    test_retry_policy_fails_five_times_then_stays_failed,
    test_bulk_register_one_bad_item_does_not_block_others,
    test_bulk_approve_mixed_results,
    test_queue_intake_enqueue_failure_does_not_strand_document_in_queued,
    test_illegal_transition_is_a_distinct_error_type_not_generic_valueerror,
    test_reject_intake_removes_document,
    test_duplicate_detection_still_enforced_at_register,
]

if __name__ == "__main__":
    failures = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e!r}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    sys.exit(1 if failures else 0)
