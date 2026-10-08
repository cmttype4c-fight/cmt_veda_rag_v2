"""
worker.py
---------
Round 4: the async processing layer (brief §12). Two things live here:

  1. RQ/Redis wiring (`enqueue_processing`, `run_worker`) — the REAL
     production path. NOT executable in this sandbox: no `redis` or `rq`
     package and no network to install them (same constraint
     DELIVERABLES.md documents for psycopg2 — see that file). Written to
     be run as: `python worker.py` (a separate process/container from the
     FastAPI app).

  2. `InlineQueueAdapter` — a synchronous, in-process stand-in with the
     exact same `enqueue(document_id, attempt) -> job_id` signature
     `intake.queue_intake()` expects. This IS what runs in this sandbox's
     tests (test_intake_lifecycle.py) and is a legitimate fallback for a
     from-scratch dev/CI environment with no Redis — not a hidden shortcut:
     it's named, documented, and never used unless explicitly selected.

Retry/backoff orchestration lives here (in `process_intake_job`), not in
intake.py's `process_intake()` — that function just runs the pipeline and
raises; whether/when to re-queue after a failure is a scheduling decision
that belongs next to the thing that actually talks to a queue.
"""

import logging
import uuid
from typing import Optional

from config import (
    REDIS_URL, RQ_QUEUE_NAME, RQ_JOB_TIMEOUT_SECONDS, MAX_RETRIES,
    retry_backoff_seconds,
)
from db import DBBackend
from embeddings import EmbeddingBackend
from vector_store import VectorStore
import intake
from config import IllegalStateTransitionError
from ingestion_pipeline import IngestionValidationError

logger = logging.getLogger("cmt-veda-ai")

# Errors that will fail identically on every attempt -> never auto-retried.
_NON_RETRYABLE = (IngestionValidationError, intake.IntakeError, IllegalStateTransitionError)


# ---------------------------------------------------------------------
# The job wrapper. Same function is used by both the real RQ worker and
# InlineQueueAdapter, so retry/backoff behavior is identical either way.
# ---------------------------------------------------------------------

def process_intake_job(
    db: DBBackend,
    embedder: EmbeddingBackend,
    vector_store: VectorStore,
    document_id: str,
    rq_job_id: str,
    attempt: int,
    enqueue_fn,
) -> None:
    """The actual job body. Runs intake.process_intake(); on failure,
    decides whether to re-queue (attempt < MAX_RETRIES) or leave the
    document FAILED for an explicit admin retry (brief: 'do not create an
    infinite automatic retry loop'). `enqueue_fn` is threaded through so a
    retry re-enters the SAME queue, real or inline."""
    row = db.get_processing_job(rq_job_id)
    if row is not None and row.get("status") == "cancelled":
        # A stale job (e.g. a delayed retry still in Redis) for a document that
        # was repaired/reset. PostgreSQL is the authoritative ledger: do nothing.
        logger.info("skipping cancelled job %s for document %s", rq_job_id, document_id)
        return
    try:
        intake.process_intake(db, embedder, vector_store, document_id, rq_job_id)
    except _NON_RETRYABLE as exc:
        # Deterministic defects (no extracted text, validation failure, illegal
        # state) fail the same way every time: retrying 5x only burns the retry
        # budget (this is how CMT-RAG-000034 reached 6/5). The document stays
        # FAILED with the reason recorded; an admin fixes the cause, then retries.
        logger.warning("non-retryable failure for document %s: %s", document_id, exc)
    except Exception:
        if attempt < MAX_RETRIES:
            # Real RQ path: schedule the retry `retry_backoff_seconds(attempt)`
            # in the future (rq-scheduler / Queue.enqueue_in) instead of
            # immediately — see enqueue_processing's docstring. The inline
            # adapter (tests) enqueues immediately: there's no real queue
            # to delay on, and tests shouldn't sleep for backoff.
            intake.queue_intake(
                db, document_id, actor="system:retry",
                enqueue_fn=enqueue_fn, attempt=attempt + 1,
            )
        # else: stays FAILED. Nothing further happens automatically.


# ---------------------------------------------------------------------
# 1. Real RQ/Redis wiring — written, NOT executed in this sandbox.
# ---------------------------------------------------------------------

def get_redis_connection():
    import redis  # noqa: local import — only required when this path is used
    return redis.from_url(REDIS_URL)


def get_queue():
    import rq  # noqa: local import
    return rq.Queue(RQ_QUEUE_NAME, connection=get_redis_connection())


def _rq_job_body(document_id: str, attempt: int) -> None:
    """The function RQ actually calls in the worker process. Rebuilds the
    db/embedder/vector_store from config inside the worker process itself
    (a separate OS process from the FastAPI app -- these objects are never
    shared across the process boundary, by design). Fresh objects per job
    also means the FAISS index is re-read from disk each time, so a job
    always appends to the latest persisted state."""
    from runtime_setup import build_db, build_embedder, build_vector_store
    db = build_db()
    try:
        embedder = build_embedder()
        vector_store = build_vector_store(embedder)
        rq_job_id = _current_rq_job_id() or str(uuid.uuid4())
        process_intake_job(db, embedder, vector_store, document_id, rq_job_id, attempt, enqueue_processing)
    finally:
        if hasattr(db, "close"):
            db.close()


def _current_rq_job_id() -> Optional[str]:
    from rq import get_current_job
    job = get_current_job()
    return job.id if job else None


def enqueue_processing(document_id: str, attempt: int) -> str:
    """The real `enqueue_fn` passed to intake.queue_intake() in production.
    A retry (attempt > 1) is scheduled `retry_backoff_seconds(attempt)` in
    the future via rq-scheduler's `Queue.enqueue_in`, rather than
    immediately — this is the actual exponential-backoff mechanism (brief
    §12). The first attempt enqueues immediately."""
    queue = get_queue()
    if attempt <= 1:
        job = queue.enqueue(_rq_job_body, document_id, attempt, job_timeout=RQ_JOB_TIMEOUT_SECONDS)
    else:
        job = queue.enqueue_in(
            __import__("datetime").timedelta(seconds=retry_backoff_seconds(attempt - 1)),
            _rq_job_body, document_id, attempt, job_timeout=RQ_JOB_TIMEOUT_SECONDS,
        )
    return job.id


def run_worker() -> None:
    """CLI entrypoint: `python worker.py`. Runs as a separate process from
    the FastAPI app (`main.py`), listening on RQ_QUEUE_NAME.

    with_scheduler=True is required: retries after attempt 1 are enqueued
    with Queue.enqueue_in() (exponential backoff), and RQ only moves those
    delayed jobs onto the queue when a worker is running its scheduler.
    Run exactly ONE worker process: each job appends to the on-disk vector
    index, and concurrent writers would race."""
    import logging
    import rq
    from runtime_setup import validate_production_config
    logging.basicConfig(level=logging.INFO)
    validate_production_config()
    connection = get_redis_connection()
    queue = rq.Queue(RQ_QUEUE_NAME, connection=connection)
    rq.Worker([queue], connection=connection).work(with_scheduler=True)


# ---------------------------------------------------------------------
# 2. InlineQueueAdapter — the tested substitute (this sandbox, and a
#    from-scratch dev/CI box with no Redis). Named and documented, never
#    silently substituted for the real thing.
# ---------------------------------------------------------------------

class InlineQueueAdapter:
    """Runs `process_intake_job` synchronously, in-process, the moment
    it's enqueued — no Redis, no second process, no real asynchrony.
    Deliberately NOT used unless the caller explicitly passes this
    adapter's `.enqueue_fn` to intake.queue_intake()/approve_intake().
    Retries triggered by a failure also run inline and immediately
    (backoff is skipped — nothing here depends on wall-clock delay).
    """

    def __init__(self, db: DBBackend, embedder: EmbeddingBackend, vector_store: VectorStore):
        self.db = db
        self.embedder = embedder
        self.vector_store = vector_store
        self._pending: list = []

    def enqueue_fn(self, document_id: str, attempt: int) -> str:
        rq_job_id = f"inline-{uuid.uuid4()}"
        # NOTE: intake.queue_intake() calls this BEFORE creating the
        # processing_jobs row for THIS attempt (it needs the job id back
        # first) — so process_intake_job must run after that row exists.
        # queue_intake() creates the row immediately after enqueue_fn
        # returns, so running the job body synchronously here would race
        # ahead of it. To keep ordering correct without changing
        # queue_intake()'s contract, the inline adapter defers the actual
        # run via `run_pending()`, called explicitly by the caller (tests
        # call it right after queue_intake()/approve_intake() returns).
        self._pending.append((document_id, rq_job_id, attempt))
        return rq_job_id

    def run_pending(self) -> None:
        """Drains and runs every job enqueued since the last call —
        including any retry jobs process_intake_job itself enqueues on
        failure, so one run_pending() call fully settles a document to
        its final state (INDEXED, or FAILED after MAX_RETRIES)."""
        while self._pending:
            document_id, rq_job_id, attempt = self._pending.pop(0)
            process_intake_job(
                self.db, self.embedder, self.vector_store,
                document_id, rq_job_id, attempt, self.enqueue_fn,
            )


if __name__ == "__main__":
    run_worker()
