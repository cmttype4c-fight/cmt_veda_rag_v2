#!/usr/bin/env python3
"""
scripts/postgres_smoke_test.py
--------------------------------
Real verification script for the PostgreSQL persistence path (final
hardening brief, item 1: "Implement and verify the real PostgreSQL
persistence path"). This is the thing to actually run on/against your
staging Postgres instance before trusting it — it exercises `db.py`'s
`PostgresBackend` for real, the same way `tests/test_pipeline_e2e.py`
exercises `SqliteBackend` for real in this sandbox.

**This script has NOT been run anywhere.** There is no Postgres server,
no `psycopg2` package, and no network to get either in the sandbox that
built this repo. Do not treat this script's existence as verification —
run it and read its output before believing the Postgres path works.

Usage:
    pip install psycopg2-binary
    psql "$RAG_POSTGRES_DSN" -f db/schema.sql      # apply schema (idempotent, safe to rerun)
    psql "$RAG_POSTGRES_DSN" -f db/schema.sql      # run it again — should not error (proves idempotency)
    python3 scripts/postgres_smoke_test.py --dsn "$RAG_POSTGRES_DSN"

What it checks, end to end, against the real database:
  1. Connects and confirms the expected tables/types exist.
  2. Creates a document, walks it through the full corrected lifecycle
     (discovered -> pending_approval -> approved -> queued -> processing
     -> indexed), and confirms an illegal transition (indexed -> approved)
     is rejected by the database layer.
  3. Confirms the Source ID survives a fresh connection (closes and
     reopens the connection to simulate a service restart) and that the
     sequence does not reuse or collide.
  4. Confirms `get_active_chunk_ids()` excludes an approved-but-not-yet-
     indexed document's chunks, then includes them once indexed.
  5. Removes the document and confirms its chunks disappear from both
     `get_active_chunk_ids()` and `lexical_search()`.
  6. Cleans up the test rows it created (does not touch anything else in
     the database).

Exit code is 0 only if every check passes.
"""

import argparse
import sys
import uuid

sys.path.insert(0, ".")

from db import PostgresBackend, DocumentRecord, ChunkRecord
from config import IngestionState


def _check(label: str, condition: bool, details: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" — {details}" if details and not condition else ""))
    return condition


def run(dsn: str) -> bool:
    all_ok = True
    db = PostgresBackend(dsn)
    test_doc_id = str(uuid.uuid4())

    try:
        # 1. connectivity + schema presence (implicit: create_document
        #    will fail loudly if the table/type doesn't exist)
        source_id = db.next_source_id()
        all_ok &= _check("Connected and allocated a Source ID from the real sequence",
                          bool(source_id) and source_id.startswith("CMT-RAG-"))

        doc = DocumentRecord(
            document_id=test_doc_id, source_id=source_id,
            title="[SMOKE TEST] delete me", doi="10.9999/smoke-test-do-not-use",
        )
        db.create_document(doc)
        all_ok &= _check("Created document row", db.get_document(test_doc_id) is not None)

        # 2. lifecycle correctness
        db.transition_document_state(test_doc_id, IngestionState.PENDING_APPROVAL, "smoke-test")
        db.transition_document_state(test_doc_id, IngestionState.APPROVED, "smoke-test")
        db.add_chunks([ChunkRecord(chunk_id=str(uuid.uuid4()), document_id=test_doc_id,
                                    section="body", chunk_order=0,
                                    content="smoke test chunk mentioning CMT and SH3TC2")])
        active_before_index = db.get_active_chunk_ids()
        chunk_ids = [c.chunk_id for c in db.get_chunks_for_document(test_doc_id)]
        all_ok &= _check("APPROVED-but-not-INDEXED chunk is NOT in the active set",
                          not any(cid in active_before_index for cid in chunk_ids))

        db.transition_document_state(test_doc_id, IngestionState.QUEUED, "smoke-test")
        db.transition_document_state(test_doc_id, IngestionState.PROCESSING, "smoke-test")
        db.transition_document_state(test_doc_id, IngestionState.INDEXED, "smoke-test")
        active_after_index = db.get_active_chunk_ids()
        all_ok &= _check("INDEXED chunk IS in the active set",
                          all(cid in active_after_index for cid in chunk_ids))

        try:
            db.transition_document_state(test_doc_id, IngestionState.APPROVED, "smoke-test")
            all_ok &= _check("Illegal transition (INDEXED -> APPROVED) rejected", False,
                              "no exception was raised")
        except ValueError:
            all_ok &= _check("Illegal transition (INDEXED -> APPROVED) rejected", True)

        # 3. Source ID survives a fresh connection
        db.close()
        db = PostgresBackend(dsn)
        refetched = db.get_document(test_doc_id)
        all_ok &= _check("Source ID stable across a fresh connection",
                          refetched is not None and refetched.source_id == source_id)
        second_id = db.next_source_id()
        all_ok &= _check("Sequence continues without collision after reconnect",
                          second_id != source_id)

        # 4. lexical search only returns INDEXED content
        hits = db.lexical_search("SH3TC2", limit=10)
        all_ok &= _check("Lexical search finds the indexed chunk",
                          any(cid in chunk_ids for cid, _ in hits))

        # 5. removal actually removes it
        db.transition_document_state(test_doc_id, IngestionState.REMOVED, "smoke-test")
        active_after_removal = db.get_active_chunk_ids()
        hits_after_removal = db.lexical_search("SH3TC2", limit=10)
        all_ok &= _check("Removed document's chunks excluded from active set",
                          not any(cid in active_after_removal for cid in chunk_ids))
        all_ok &= _check("Removed document's chunks excluded from lexical search",
                          not any(cid in chunk_ids for cid, _ in hits_after_removal))

        history = db.audit_history(test_doc_id)
        all_ok &= _check("Full audit trail preserved (6 transitions)", len(history) == 6,
                          f"got {len(history)}")

    finally:
        # 6. cleanup — only removes rows this script itself created
        try:
            with db._cursor() as cur:
                cur.execute("DELETE FROM cmt_veda_rag.documents WHERE document_id = %s", (test_doc_id,))
            print("[INFO] Cleaned up smoke-test rows.")
        except Exception as e:
            print(f"[WARN] Cleanup failed, you may need to manually delete document_id={test_doc_id}: {e}")
        db.close()

    return all_ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dsn", required=True, help="Postgres connection string, e.g. postgresql://user:pass@host:5432/dbname")
    args = ap.parse_args()

    ok = run(args.dsn)
    print("\n" + ("ALL CHECKS PASSED" if ok else "ONE OR MORE CHECKS FAILED"))
    sys.exit(0 if ok else 1)
