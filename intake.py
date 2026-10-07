"""
intake.py
---------
Round 4: splits round 3's fused approve_and_ingest()/ingest_document()
into the four real functions the brief names explicitly:

    register_intake()  DISCOVERED        -> PENDING_APPROVAL
    approve_intake()   PENDING_APPROVAL  -> APPROVED  -> QUEUED  (auto-queues;
                                                                   your confirmed decision #1)
    queue_intake()     APPROVED/FAILED   -> QUEUED            (also called
                                                                 internally by approve_intake,
                                                                 and by the worker on retry)
    process_intake()   QUEUED            -> PROCESSING -> INDEXED/FAILED
                        (worker-side; see worker.py -- never called from an
                         HTTP request, which is the actual fix for round 3's
                         "approval blocks for minutes" problem)

Both source types (discovery, direct_upload) go through the SAME four
functions from here on (brief §2: "Both sources must ultimately follow
the same RAG approval and ingestion lifecycle"). discovery.py's
register_candidate()/reject_candidate() keep doing what they already did
(the discovery_candidates ledger, before any text exists) -- main.py's
legacy /admin/discovery/approve handler now calls register_intake() +
approve_intake() internally instead of the old approve_and_ingest(); see
main.py for exactly how the two are wired together.

Full-text-only + Newsletter isolation (brief §4/§16) is enforced here, at
the backend level, in register_intake() -- not only in whatever UI calls
it -- via config.ALLOWED_SOURCE_TYPES and config.MIN_FULL_TEXT_CHARS. See
those constants' docstrings in config.py for the honest caveats on what
this heuristic does and doesn't catch.
"""

import uuid
from datetime import datetime, timezone
from typing import Callable, Optional

from config import (
    IngestionState, ALLOWED_SOURCE_TYPES, MIN_FULL_TEXT_CHARS, MAX_RETRIES,
)
from db import DBBackend, DocumentRecord, ChunkRecord, DuplicateDocumentError
from ingestion_pipeline import (
    IngestionInput, IngestionValidationError, validate_input, _extract_text,
    auto_extract_metadata,
)
from ingestion import chunk_document
from embeddings import EmbeddingBackend
from vector_store import VectorStore
from url_safety import sanitize_reference_url


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class IntakeError(Exception):
    """Base for any intake-lifecycle rejection that isn't already its own
    exception type (DuplicateDocumentError, IngestionValidationError stay
    as-is and propagate unchanged -- main.py already maps those to 409/422
    from round 3; this is for the NEW rejection reasons round 4 adds)."""


class FullTextRuleViolation(IntakeError):
    pass


class InvalidSourceType(IntakeError):
    pass


class IntakeConflictError(IntakeError):
    """A job is already in flight, or metadata edit attempted post-approval."""


def _enforce_full_text_only(extracted_text: str) -> None:
    """Brief §4: 'A document is eligible for RAG only if genuine full text
    is available' / §16: Newsletter content must never enter RAG. This is
    a minimum-length HEURISTIC (config.MIN_FULL_TEXT_CHARS) — honestly not
    a real abstract-vs-full-text classifier; see that constant's docstring
    in config.py. The actual isolation guarantee against Newsletter
    content is structural, not length-based: nothing in this codebase's
    Newsletter pipeline calls register_intake(), and the source_type
    allowlist below rejects anything that isn't 'discovery' or
    'direct_upload' before a document row is ever created."""
    if len(extracted_text.strip()) < MIN_FULL_TEXT_CHARS:
        raise FullTextRuleViolation(
            f"Document text is {len(extracted_text.strip())} characters, below the "
            f"{MIN_FULL_TEXT_CHARS}-character full-text floor — rejecting as likely "
            f"abstract-only, editorial, or otherwise non-full-text material. "
            f"RAG accepts genuine scientific full text only."
        )


def register_intake(
    db: DBBackend,
    payload: IngestionInput,
    source_type: str,
    actor: str,
    *,
    uploaded_by: Optional[str] = None,
    original_filename: str = "",
    allow_duplicate: bool = False,
) -> DocumentRecord:
    """DISCOVERED -> PENDING_APPROVAL. Does the real validation + text
    extraction NOW (brief §9: 'UPLOAD -> validation -> extraction ->
    metadata -> PENDING_APPROVAL') so an admin reviewing a pending item is
    looking at what will actually be indexed, not a promise to extract it
    later. Does NOT chunk/embed/index — process_intake() does that, on
    the worker, after approve_intake()/queue_intake().
    """
    if source_type not in ALLOWED_SOURCE_TYPES:
        raise InvalidSourceType(
            f"source_type must be one of {sorted(ALLOWED_SOURCE_TYPES)}, got "
            f"{source_type!r}. Rejected before any document row was created."
        )

    validate_input(payload)
    extracted_text = _extract_text(payload)
    if not extracted_text or not extracted_text.strip():
        raise IngestionValidationError("Document text is empty after extraction.")
    if len(extracted_text.strip()) < 50:
        raise IngestionValidationError(
            "Document text is suspiciously short (<50 chars) — rejecting "
            "rather than registering likely-corrupted content."
        )
    _enforce_full_text_only(extracted_text)

    auto_meta = auto_extract_metadata(extracted_text)
    genes = sorted(set(payload.genes) | set(auto_meta["genes"]))
    cmt_subtypes = sorted(set(payload.cmt_subtypes) | set(auto_meta["cmt_subtypes"]))

    if not allow_duplicate:
        dup = db.find_duplicate(doi=payload.doi, pmid=payload.pmid, title=payload.title)
        if dup is not None:
            raise DuplicateDocumentError(
                f"Duplicate of existing document {dup.document_id} "
                f"(source_id={dup.source_id}); pass allow_duplicate=True to override."
            )

    document_id = str(uuid.uuid4())
    source_id = db.next_source_id()
    # ingestion_method keeps its round-3 values ('discovery' / 'manual_upload')
    # — see DocumentRecord.source_type's docstring for why this is a computed
    # alias rather than a physical column rename.
    ingestion_method = "discovery" if source_type == "discovery" else "manual_upload"
    doc = DocumentRecord(
        document_id=document_id, source_id=source_id, title=payload.title,
        authors=payload.authors, journal=payload.journal,
        publication_date=payload.publication_date, doi=payload.doi, pmid=payload.pmid,
        trial_id=payload.trial_id, cmt_subtypes=cmt_subtypes, genes=genes,
        study_type=payload.study_type, source_tier=payload.source_tier,
        source_url=sanitize_reference_url(payload.source_url),
        discovery_candidate_id=payload.discovery_candidate_id,
        ingestion_method=ingestion_method,
        approval_status=IngestionState.DISCOVERED,
        knowledge_version=payload.knowledge_version,
        source_format=payload.format,
        uploaded_by=uploaded_by,
        uploaded_at=_now() if uploaded_by else None,
        original_filename=original_filename,
        extracted_text=extracted_text,
    )
    db.create_document(doc)
    db.transition_document_state(document_id, IngestionState.PENDING_APPROVAL, actor="system")
    return db.get_document(document_id)


def patch_intake_metadata(db: DBBackend, document_id: str, fields: dict) -> DocumentRecord:
    """PATCH /admin/intake/{id} (brief §8's last line): editable ONLY
    while pending_approval, never after — keeps the audit trail honest
    about what was actually reviewed and approved."""
    doc = db.get_document(document_id)
    if doc is None:
        raise ValueError(f"No such document: {document_id}")
    if doc.approval_status != IngestionState.PENDING_APPROVAL:
        raise IntakeConflictError(
            f"Cannot edit metadata: document {document_id} is in state "
            f"'{doc.approval_status.value}', not 'pending_approval'. Metadata is "
            f"locked once a document leaves pending_approval, to keep the audit "
            f"trail honest about what was actually reviewed."
        )
    return db.update_document_metadata(document_id, fields)


def reject_intake(db: DBBackend, document_id: str, actor: str, reason: str = "") -> DocumentRecord:
    """New-path equivalent of discovery.py's reject_candidate(), but
    operating on an already-created document row (register_intake()
    creates the row immediately, unlike the old discovery_candidates
    ledger which held no document row pre-approval). Maps onto the
    existing REMOVED terminal state rather than inventing a new one."""
    return db.transition_document_state(document_id, IngestionState.REMOVED, actor=actor)


def queue_intake(
    db: DBBackend,
    document_id: str,
    actor: str,
    enqueue_fn: Callable[[str, int], str],
    attempt: int = 1,
) -> tuple[DocumentRecord, str]:
    """APPROVED -> QUEUED (first attempt), or FAILED -> QUEUED (an admin-
    triggered retry after MAX_RETRIES was hit, or a worker-driven retry
    under it — see worker.py). `enqueue_fn(document_id, attempt) ->
    rq_job_id` is injected so this function has no direct Redis/RQ
    dependency: worker.py passes the real RQ enqueue; tests pass an
    inline stand-in. Either way, Postgres (processing_jobs) is what
    `queue_intake` itself treats as authoritative.

    Idempotency (brief §12 "duplicate protection"): refuses to double-queue
    a document that already has a non-terminal job.

    If `enqueue_fn` itself raises (e.g. Redis/RQ unreachable), the
    document is deliberately NOT left sitting in QUEUED with no job
    record behind it — that would be a silent lie ("queued" with nothing
    actually in flight, and no error anywhere for an admin to see). QUEUED's
    only legal exits are PROCESSING and FAILED (config._ALLOWED_TRANSITIONS
    — there is no "un-queue back to APPROVED"), so this records a failed
    processing_jobs row (a synthetic id, since no real rq_job_id was ever
    issued) and transitions straight to FAILED, then re-raises so the
    caller (main.py) can surface it as a clear 503 rather than a
    misleading "success".
    """
    existing = db.get_latest_processing_job(document_id)
    if existing is not None and existing["status"] in ("queued", "started"):
        raise IntakeConflictError(
            f"Document {document_id} already has an in-flight job "
            f"({existing['rq_job_id']}, status={existing['status']}); refusing to double-queue."
        )
    doc = db.transition_document_state(document_id, IngestionState.QUEUED, actor=actor)
    try:
        rq_job_id = enqueue_fn(document_id, attempt)
    except Exception as e:
        synthetic_job_id = f"enqueue-failed-{uuid.uuid4()}"
        db.create_processing_job(str(uuid.uuid4()), document_id, synthetic_job_id, attempt)
        db.update_processing_job(synthetic_job_id, status="failed", error=f"enqueue_fn failed: {e}")
        db.transition_document_state(document_id, IngestionState.FAILED, actor="system")
        db.set_retry_count(document_id, (doc.retry_count or 0) + 1)
        raise
    db.create_processing_job(str(uuid.uuid4()), document_id, rq_job_id, attempt)
    return doc, rq_job_id


def approve_intake(
    db: DBBackend,
    document_id: str,
    actor: str,
    enqueue_fn: Callable[[str, int], str],
) -> tuple[DocumentRecord, str]:
    """PENDING_APPROVAL -> APPROVED -> QUEUED, in one call (your confirmed
    decision #1: approval auto-queues; no separate manual queue step in
    the normal admin workflow). APPROVED remains the real, distinct state
    it always was — it's just not exposed as its own long-lived admin
    action anymore. The document becomes retrievable only once the worker
    finishes process_intake() and the state reaches INDEXED."""
    db.transition_document_state(document_id, IngestionState.APPROVED, actor=actor)
    return queue_intake(db, document_id, actor=actor, enqueue_fn=enqueue_fn, attempt=1)


def process_intake(
    db: DBBackend,
    embedder: EmbeddingBackend,
    vector_store: VectorStore,
    document_id: str,
    rq_job_id: str,
) -> DocumentRecord:
    """QUEUED -> PROCESSING -> INDEXED/FAILED. Runs on the worker (see
    worker.py's process_intake_job), NEVER inline in an HTTP request —
    this is the actual fix for round 3's "the approval request blocks
    until indexing finishes" problem (what made the standalone
    laptop-RAG assistant project's requests take 2-5 minutes).

    On failure: records FAILED + the error, and increments
    documents.retry_count (Postgres stays authoritative for this count —
    see config.MAX_RETRIES). Does NOT re-queue itself; that decision
    (re-queue under MAX_RETRIES vs. leave FAILED for an explicit admin
    retry) belongs to the worker's job wrapper (worker.py), which is the
    layer that actually talks to RQ/backoff scheduling — this function
    stays testable without touching a queue at all.
    """
    db.update_processing_job(rq_job_id, status="started")
    doc = db.transition_document_state(document_id, IngestionState.PROCESSING, actor="system")
    try:
        extracted_text = doc.extracted_text
        if not extracted_text or not extracted_text.strip():
            raise IngestionValidationError(
                "No extracted text stored for this document — cannot process. "
                "(This would indicate a bug in register_intake(), which should "
                "never leave a document pending without its extracted text.)"
            )

        chunks = chunk_document(document_id, doc.source_id, extracted_text)
        if not chunks:
            raise IngestionValidationError("Chunker produced zero chunks from the input text.")

        chunk_records = [
            ChunkRecord(
                chunk_id=c["metadata"].chunk_id, document_id=document_id,
                section=c["metadata"].section, chunk_order=c["metadata"].order,
                content=c["text"],
            )
            for c in chunks
        ]
        db.add_chunks(chunk_records)
        vectors = embedder.embed([c["text"] for c in chunks])
        vector_store.add([r.chunk_id for r in chunk_records], vectors)

        doc = db.transition_document_state(document_id, IngestionState.INDEXED, actor="system")
        db.update_processing_job(rq_job_id, status="finished")
        return doc
    except Exception as e:
        db.transition_document_state(document_id, IngestionState.FAILED, actor="system")
        db.update_processing_job(rq_job_id, status="failed", error=str(e))
        db.set_retry_count(document_id, (doc.retry_count or 0) + 1)
        raise


# ---------------------------------------------------------------------
# Bulk operations (brief §10). Each loops with its own try/except PER
# ITEM — never one transaction around the whole batch — because a
# transaction-wrapped batch would roll back everyone's work if item 7 of
# 50 fails, which is exactly what the brief says must NOT happen ("one
# failed document must not incorrectly mark every document in the batch
# as successful").
# ---------------------------------------------------------------------

def bulk_register_intake(db: DBBackend, items: list[dict], actor: str) -> dict:
    """items: [{source_type, payload: {...IngestionInput kwargs...},
    uploaded_by?, original_filename?}, ...]"""
    batch_id = str(uuid.uuid4())
    db.create_intake_batch(batch_id, "intake", actor, len(items))
    results = []
    for item in items:
        try:
            payload = IngestionInput(**item.get("payload", {}))
            source_type = item["source_type"]
            doc = register_intake(
                db, payload, source_type, actor,
                uploaded_by=item.get("uploaded_by"),
                original_filename=item.get("original_filename", ""),
            )
            db.set_intake_extras(doc.document_id, intake_batch_id=batch_id)
            db.record_batch_item(batch_id, doc.document_id, "success", None)
            results.append({"document_id": doc.document_id, "status": "success"})
        except Exception as e:
            db.record_batch_item(batch_id, None, "failed", str(e))
            results.append({"status": "failed", "error": str(e)})
    return {"batch_id": batch_id, "results": results, **(db.get_batch(batch_id) or {})}


def bulk_approve_intake(
    db: DBBackend,
    document_ids: list[str],
    actor: str,
    enqueue_fn: Callable[[str, int], str],
) -> dict:
    """pending_approval -> approved -> queued for each id (approval
    auto-queues, same as the single-item path). One failure doesn't block
    or roll back the rest."""
    batch_id = str(uuid.uuid4())
    db.create_intake_batch(batch_id, "approve", actor, len(document_ids))
    results = []
    for document_id in document_ids:
        try:
            doc, rq_job_id = approve_intake(db, document_id, actor, enqueue_fn)
            db.record_batch_item(batch_id, document_id, "success", None)
            results.append({"document_id": document_id, "status": "success", "job_id": rq_job_id})
        except Exception as e:
            db.record_batch_item(batch_id, document_id, "failed", str(e))
            results.append({"document_id": document_id, "status": "failed", "error": str(e)})
    return {"batch_id": batch_id, "results": results, **(db.get_batch(batch_id) or {})}
