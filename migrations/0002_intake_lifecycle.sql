-- CMT Veda AI RAG v2 — round 4: intake lifecycle migration
--
-- Adds what register_intake() / approve_intake() / queue_intake() /
-- process_intake() need on top of schema.sql's existing documents /
-- discovery_candidates / ingestion_audit tables and ingestion_state enum.
-- Purely additive — no column dropped or renamed, no existing row
-- touched except by an ADD COLUMN default. Idempotent: safe to rerun
-- (IF NOT EXISTS throughout; Postgres supports it on ADD COLUMN, unlike
-- CREATE TYPE — see schema.sql's own note on that).
--
-- DEVIATION FROM THE REVIEWED PROPOSAL DOC: the proposal sketched a new
-- `source_type` enum/column as a physical rename of `ingestion_method`.
-- Implemented here instead as a pure read-time mapping in application
-- code (DocumentRecord.source_type property in db.py, and the
-- source_type filter in list_documents_page()) — zero schema/enum change,
-- same external vocabulary ('discovery' / 'direct_upload') in every
-- round-4 API response, zero risk to round-3's already-passing tests
-- that read/write `ingestion_method` directly. If you'd rather have the
-- physical column (e.g. for a future cross-database report that queries
-- Postgres directly without going through this app), say so and I'll
-- add it as a true migration in a follow-up round.
--
-- Run: psql <dsn> -f migrations/0002_intake_lifecycle.sql
-- (after schema.sql has already been applied once)

ALTER TABLE cmt_veda_rag.documents
    ADD COLUMN IF NOT EXISTS source_format TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS uploaded_by TEXT,
    ADD COLUMN IF NOT EXISTS uploaded_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS original_filename TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS extracted_text TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS retry_count INTEGER NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS intake_batch_id UUID;

-- Discovery provenance (brief §7): preserved beyond discovery_candidate_id
-- (already in schema.sql) with the full-text's own format and a free-form
-- bag for whatever Discovery's internal IDs/metadata look like, so this
-- doesn't need another migration every time Discovery's payload shape
-- changes.
ALTER TABLE cmt_veda_rag.discovery_candidates
    ADD COLUMN IF NOT EXISTS source_format TEXT,
    ADD COLUMN IF NOT EXISTS original_provenance JSONB;

-- Bulk operations (brief §10). One row per bulk call; per-item results in
-- intake_batch_items so "one failed document must not incorrectly mark
-- every document in the batch as successful" is structurally true — the
-- batch's own success_count/failure_count are counters, not a verdict.
CREATE TABLE IF NOT EXISTS cmt_veda_rag.intake_batches (
    batch_id      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    batch_type    TEXT NOT NULL,          -- 'intake' | 'approve'
    created_by    TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    item_count    INTEGER NOT NULL DEFAULT 0,
    success_count INTEGER NOT NULL DEFAULT 0,
    failure_count INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS cmt_veda_rag.intake_batch_items (
    id          BIGSERIAL PRIMARY KEY,
    batch_id    UUID NOT NULL REFERENCES cmt_veda_rag.intake_batches(batch_id) ON DELETE CASCADE,
    document_id UUID REFERENCES cmt_veda_rag.documents(document_id),
    status      TEXT NOT NULL,            -- 'success' | 'failed'
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_batch_items_batch_id ON cmt_veda_rag.intake_batch_items (batch_id);

-- RQ job tracking (brief §12). Postgres is the durable source of truth
-- for retry/attempt/status/error — Redis/RQ itself can lose this on a
-- flush or restart, so nothing here depends on Redis remembering anything.
CREATE TABLE IF NOT EXISTS cmt_veda_rag.processing_jobs (
    job_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id  UUID NOT NULL REFERENCES cmt_veda_rag.documents(document_id) ON DELETE CASCADE,
    rq_job_id    TEXT NOT NULL UNIQUE,
    attempt      INTEGER NOT NULL DEFAULT 1,
    status       TEXT NOT NULL DEFAULT 'queued',  -- 'queued' | 'started' | 'finished' | 'failed'
    enqueued_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at   TIMESTAMPTZ,
    finished_at  TIMESTAMPTZ,
    error        TEXT
);
CREATE INDEX IF NOT EXISTS idx_processing_jobs_document_id ON cmt_veda_rag.processing_jobs (document_id);
