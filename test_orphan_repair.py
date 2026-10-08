"""
test_orphan_repair.py
---------------------
The live CMT-RAG-000034 failure: a document registered on PostgreSQL with
extracted_text = '' (insert bug), later approved, queued, retried to 6/5 and
stuck FAILED, with a delayed retry still pending in Redis. Verifies:

  * re-sending the source REPAIRS the same row (same document_id/source_id),
    no allow_duplicate, from every non-terminal state;
  * approval is NOT bypassed: it returns to pending_approval, approval fields
    cleared, retry_count reset, audit row written;
  * the stale retry chain is cancelled (Postgres ledger) and a stale job that
    wakes up later does nothing;
  * afterwards pending -> approved -> queued -> processing -> indexed works and
    Ask Veda's retriever returns the source;
  * guards: approve/queue/retry refuse a text-less document (no state change);
    permanent errors are not auto-retried; a backend that fails to persist the
    text is detected at registration (read-back);
  * INDEXED documents and genuine duplicates are never touched.
Run: python3 test_orphan_repair.py   (or pytest)
"""

import os
import shutil
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import intake
from config import IngestionState as S, MAX_RETRIES
from db import SqliteBackend, DuplicateDocumentError
from embeddings import HashingTfidfEmbeddings
from ingestion_pipeline import IngestionInput
from retrieval import HybridRetriever
from vector_store import NumpyVectorStore
from worker import InlineQueueAdapter, process_intake_job

TITLE = ("Cardiovascular Autonomic Neuropathy in Charcot-Marie-Tooth Disease: A Genotype-Stratified "
         "Study of PMP22 Duplication and GJB1 Mutation Carriers")
BODY = (("Cardiovascular autonomic neuropathy was assessed in Charcot-Marie-Tooth disease patients "
         "with PMP22 duplication and GJB1 mutations using heart-rate variability and orthostatic testing. ") * 120)[:20000]
DOI = "10.1234/cmt.cardio.autonomic"


def _env():
    tmp = tempfile.mkdtemp(prefix="cmt_repair_")
    db = SqliteBackend(os.path.join(tmp, "t.db"))
    emb = HashingTfidfEmbeddings(n_features=512)
    vs = NumpyVectorStore(os.path.join(tmp, "vs", "s"), dim=512)
    return tmp, db, emb, vs, InlineQueueAdapter(db, emb, vs)


def _register(db, text=BODY, **kw):
    return intake.register_intake(
        db, IngestionInput(raw_text=text, format="text", title=TITLE, doi=DOI,
                           source_format="xml", source_mime_type="application/xml", **kw),
        "discovery", actor="veda")


def _make_orphan(db, target):
    """Reproduces the buggy row: registered, text lost, then driven to `target`
    the way the live system did (transitions only; the new guards would refuse
    to approve a text-less document, which is the point)."""
    d = _register(db)
    db.set_intake_extras(d.document_id, extracted_text="", source_format="")
    path = {S.PENDING_APPROVAL: [], S.APPROVED: [S.APPROVED], S.QUEUED: [S.APPROVED, S.QUEUED],
            S.PROCESSING: [S.APPROVED, S.QUEUED, S.PROCESSING],
            S.FAILED: [S.APPROVED, S.QUEUED, S.PROCESSING, S.FAILED]}[target]
    for st in path:
        db.transition_document_state(d.document_id, st, actor="veda-admin")
    stale = []
    if target in (S.QUEUED, S.PROCESSING, S.FAILED):
        for attempt in range(1, 4):                       # a retry chain in flight
            rid = f"rq-{uuid.uuid4()}"
            db.create_processing_job(str(uuid.uuid4()), d.document_id, rid, attempt)
            if attempt < 3:
                db.update_processing_job(rid, "failed", "No extracted text stored")
            else:
                stale.append(rid)                          # still 'queued': the delayed retry in Redis
    if target == S.FAILED:
        db.set_retry_count(d.document_id, 6)
    return d, stale


def test_repair_from_every_stuck_state_keeps_ids_and_requires_reapproval():
    for target in (S.PENDING_APPROVAL, S.APPROVED, S.QUEUED, S.PROCESSING, S.FAILED):
        tmp, db, emb, vs, adapter = _env()
        try:
            orphan, stale = _make_orphan(db, target)
            assert db.get_document(orphan.document_id).extracted_text == ""
            fixed = _register(db)                                          # no allow_duplicate
            assert (fixed.document_id, fixed.source_id) == (orphan.document_id, orphan.source_id), target
            assert fixed.approval_status == S.PENDING_APPROVAL, (target, fixed.approval_status)
            assert fixed.approved_by is None and fixed.approved_at is None   # approval NOT carried over
            assert fixed.retry_count == 0
            assert fixed.extracted_text == BODY and len(fixed.extracted_text) == 20000
            assert (fixed.source_format, fixed.source_mime_type) == ("xml", "application/xml")
            assert len(db.list_documents_page()) == 1                       # no duplicate row
            if target != S.PENDING_APPROVAL:
                audit = db._conn.execute(
                    "SELECT actor, to_state FROM ingestion_audit WHERE document_id=? ORDER BY id DESC LIMIT 1",
                    (orphan.document_id,)).fetchone()
                assert audit["to_state"] == "pending_approval" and audit["actor"].startswith("system:repair")
            for rid in stale:
                assert db.get_processing_job(rid)["status"] == "cancelled", (target, rid)
        finally:
            db.close(); shutil.rmtree(tmp, ignore_errors=True)
    print("PASS: repaired in place from pending/approved/queued/processing/failed; ids kept; approval cleared; retries reset; jobs cancelled")


def test_stale_retry_job_does_nothing_after_repair():
    tmp, db, emb, vs, adapter = _env()
    try:
        orphan, stale = _make_orphan(db, S.FAILED)
        _register(db)
        before = db.get_document(orphan.document_id)
        calls = []
        process_intake_job(db, emb, vs, orphan.document_id, stale[0], 4, lambda *a: calls.append(a) or "x")
        after = db.get_document(orphan.document_id)
        assert after.approval_status == S.PENDING_APPROVAL == before.approval_status
        assert after.retry_count == 0 and not calls                         # no new retry enqueued
        print("PASS: a stale delayed retry that wakes up after the repair is a no-op")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_repaired_document_goes_approved_queued_processing_indexed_and_ask_veda_finds_it():
    tmp, db, emb, vs, adapter = _env()
    try:
        orphan, _ = _make_orphan(db, S.FAILED)
        fixed = _register(db)
        assert len(fixed.extracted_text) > 0                                # non-zero BEFORE processing
        doc, job = intake.approve_intake(db, fixed.document_id, "admin", adapter.enqueue_fn)
        assert doc.approval_status == S.QUEUED
        adapter.run_pending()
        final = db.get_document(fixed.document_id)
        assert final.approval_status == S.INDEXED and final.retry_count == 0
        retriever = HybridRetriever(db, emb, vs)
        cands, scope, _ = retriever.retrieve_and_rank(
            "What is known about cardiovascular autonomic neuropathy in CMT with PMP22 duplication?")
        assert any(c.source_id == orphan.source_id for c in cands), [c.source_id for c in cands]
        assert all(c.metadata.get("content_type") == "research_paper" for c in cands if c.source_id == orphan.source_id)
        print(f"PASS: repaired {orphan.source_id} -> approve -> queued -> processing -> indexed; retriever returns it")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_text_less_document_cannot_be_approved_queued_or_retried():
    tmp, db, emb, vs, adapter = _env()
    try:
        d = _register(db)
        db.set_intake_extras(d.document_id, extracted_text="")
        try:
            intake.approve_intake(db, d.document_id, "admin", adapter.enqueue_fn)
            assert False
        except intake.MissingExtractedTextError:
            pass
        assert db.get_document(d.document_id).approval_status == S.PENDING_APPROVAL      # no state change
        assert db.get_latest_processing_job(d.document_id) is None
        failed, _ = _make_orphan(db, S.FAILED) if False else (None, None)
        print("PASS: approve refuses a document without extracted text and leaves its state unchanged")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_admin_retry_of_text_less_failed_document_is_refused():
    tmp, db, emb, vs, adapter = _env()
    try:
        orphan, _ = _make_orphan(db, S.FAILED)
        try:
            intake.queue_intake(db, orphan.document_id, "admin", adapter.enqueue_fn, attempt=1)
            assert False
        except intake.MissingExtractedTextError:
            pass
        assert db.get_document(orphan.document_id).approval_status == S.FAILED
        print("PASS: explicit retry of a text-less FAILED document is refused (409), not looped")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_permanent_errors_are_not_auto_retried_but_transient_ones_still_are():
    tmp, db, emb, vs, adapter = _env()
    try:
        # permanent: text missing -> exactly one attempt, FAILED, no further job
        orphan, _ = _make_orphan(db, S.PENDING_APPROVAL)
        db.transition_document_state(orphan.document_id, S.APPROVED, "a")
        db.transition_document_state(orphan.document_id, S.QUEUED, "a")
        rid = f"rq-{uuid.uuid4()}"
        db.create_processing_job(str(uuid.uuid4()), orphan.document_id, rid, 1)
        enq = []
        process_intake_job(db, emb, vs, orphan.document_id, rid, 1, lambda *a: enq.append(a) or "x")
        got = db.get_document(orphan.document_id)
        assert got.approval_status == S.FAILED and got.retry_count == 1 and not enq, (got.approval_status, got.retry_count, enq)

        # transient: embedder raises -> the full MAX_RETRIES chain still happens
        class Flaky(HashingTfidfEmbeddings):
            def embed(self, texts): raise RuntimeError("embedder temporarily unavailable")
        tmp2 = tempfile.mkdtemp(prefix="cmt_retry_")
        db2 = SqliteBackend(os.path.join(tmp2, "t.db"))
        ad2 = InlineQueueAdapter(db2, Flaky(n_features=512), NumpyVectorStore(os.path.join(tmp2, "v", "s"), dim=512))
        d2 = _register(db2)
        intake.approve_intake(db2, d2.document_id, "admin", ad2.enqueue_fn)
        ad2.run_pending()
        end = db2.get_document(d2.document_id)
        assert end.approval_status == S.FAILED and end.retry_count == MAX_RETRIES, (end.approval_status, end.retry_count)
        db2.close(); shutil.rmtree(tmp2, ignore_errors=True)
        print(f"PASS: no-text failure stops after 1 attempt; transient failure still retries exactly {MAX_RETRIES}x")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_registration_detects_a_backend_that_drops_extracted_text():
    class DropsText(SqliteBackend):                       # emulates the PostgresBackend.create_document bug
        def create_document(self, doc):
            doc.extracted_text = ""
            super().create_document(doc)
    tmp = tempfile.mkdtemp(prefix="cmt_drop_")
    db = DropsText(os.path.join(tmp, "t.db"))
    try:
        try:
            _register(db)
            assert False, "expected IntakePersistenceError"
        except intake.IntakePersistenceError as e:
            assert "extracted_text was not persisted" in str(e)
        (row,) = db.list_documents_page()
        assert row.approval_status == S.DISCOVERED        # not actionable: never reached pending_approval
        print("PASS: a storage layer that drops the text is detected at registration (loud error, row not actionable)")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_discovered_orphan_left_by_a_persistence_failure_is_repaired_on_resend():
    tmp = tempfile.mkdtemp(prefix="cmt_disc_")
    class DropsOnce(SqliteBackend):
        drop = True
        def create_document(self, doc):
            if DropsOnce.drop:
                doc.extracted_text = ""
            super().create_document(doc)
    db = DropsOnce(os.path.join(tmp, "t.db"))
    try:
        try:
            _register(db)
        except intake.IntakePersistenceError:
            pass
        DropsOnce.drop = False
        fixed = _register(db)
        assert fixed.approval_status == S.PENDING_APPROVAL and len(fixed.extracted_text) == 20000
        assert len(db.list_documents_page()) == 1
        print("PASS: re-sending after a persistence failure repairs the discovered row")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_indexed_documents_and_real_duplicates_are_never_touched():
    tmp, db, emb, vs, adapter = _env()
    try:
        d = _register(db)
        intake.approve_intake(db, d.document_id, "admin", adapter.enqueue_fn); adapter.run_pending()
        assert db.get_document(d.document_id).approval_status == S.INDEXED
        for _ in range(2):
            try:
                _register(db)
                assert False
            except DuplicateDocumentError:
                pass
        db.set_intake_extras(d.document_id, extracted_text="")              # INDEXED but no stored text (legacy shape)
        try:
            _register(db)
            assert False, "an INDEXED document must never be 'repaired'"
        except DuplicateDocumentError:
            pass
        assert db.get_document(d.document_id).approval_status == S.INDEXED
        print("PASS: INDEXED documents and genuine duplicates still raise DuplicateDocumentError")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failures = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:
            import traceback
            failures += 1
            print(f"FAIL: {t.__name__}: {e!r}")
            traceback.print_exc(limit=4)
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    sys.exit(1 if failures else 0)
