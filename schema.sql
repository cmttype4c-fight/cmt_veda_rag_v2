-- CMT Veda AI RAG v2 — metadata schema (PostgreSQL)
--
-- Scope: scientific-knowledge metadata ONLY. This is deliberately kept in
-- its own schema/namespace, separate from CMT Veda application/user data
-- and Registry patient data (section 30). Vector embeddings themselves can
-- stay in FAISS initially (section 47) — this schema tracks the metadata
-- and lifecycle that sit alongside the vectors, keyed by chunk_id/document_id
-- so FAISS and Postgres never disagree about what's indexed.
--
-- IDEMPOTENCY (final hardening brief item 1: "safe to run repeatedly"):
-- every statement here is written to be rerunnable against a database
-- that already has some or all of this schema applied. CREATE TABLE/
-- INDEX/SCHEMA/SEQUENCE all support IF NOT EXISTS natively. CREATE TYPE
-- (used for the two enums below) does NOT support IF NOT EXISTS in
-- PostgreSQL — the standard idiom is a DO block that catches
-- `duplicate_object`, used below. This file has NOT been run against a
-- real Postgres instance in this sandbox (no server/psycopg2/network
-- available here) — it is written correctly and reviewed for the known
-- idempotency gotchas, but "written correctly" and "verified against a
-- live server" are different claims; see DELIVERABLES.md.
--
-- Run this file (`psql -f db/schema.sql` or equivalent) once per
-- deployment target; rerunning it after a partial/failed prior run, or
-- as part of a repeatable migration/deploy step, should be safe.

-- gen_random_uuid() is built into PostgreSQL core as of v13. On older
-- versions it requires the pgcrypto extension. This line is a no-op if
-- the function already exists in core; it's included so this schema
-- also works unmodified against PG < 13.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE SCHEMA IF NOT EXISTS cmt_veda_rag;

DO $$ BEGIN
    CREATE TYPE cmt_veda_rag.ingestion_state AS ENUM (
        'discovered',
        'pending_approval',
        'approved',
        'queued',
        'processing',
        'indexed',
        'failed',
        'removed'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;  -- type already exists; safe to rerun
END $$;

DO $$ BEGIN
    CREATE TYPE cmt_veda_rag.ingestion_method AS ENUM (
        'discovery',
        'manual_upload'
    );
EXCEPTION
    WHEN duplicate_object THEN NULL;  -- type already exists; safe to rerun
END $$;

-- Atomic, race-free Source ID allocation (a MAX(source_id)+1 approach has
-- a TOCTOU race under concurrent writes — use a real sequence instead).
-- Source IDs must survive restart/reindex (final hardening brief item 6):
-- a Postgres SEQUENCE is durable server-side state, unaffected by this
-- service restarting or the vector index being rebuilt.
CREATE SEQUENCE IF NOT EXISTS cmt_veda_rag.source_id_seq START 1;

-- One row per source document (paper, guideline, etc).
CREATE TABLE IF NOT EXISTS cmt_veda_rag.documents (
    document_id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id               TEXT NOT NULL UNIQUE,   -- e.g. 'CMT-RAG-000427', public-safe
    title                   TEXT,
    authors                 TEXT[],
    journal                 TEXT,
    publication_date        DATE,
    doi                     TEXT,
    pmid                    TEXT,
    trial_id                TEXT,
    cmt_subtypes            TEXT[] DEFAULT '{}',
    genes                   TEXT[] DEFAULT '{}',
    study_type              TEXT,
    source_tier             TEXT DEFAULT 'unspecified',
    source_url              TEXT,                   -- PUBLIC bibliographic link only —
                                                      -- app-layer sanitized via url_safety.py
                                                      -- to reject direct PDF/storage links
                                                      -- (final hardening brief item 5); this
                                                      -- column does not itself enforce that,
                                                      -- the application layer does, on every
                                                      -- write AND every read (defense in depth)
    internal_storage_path   TEXT,                    -- NEVER returned by any API response
    discovery_candidate_id  TEXT,
    ingestion_method        cmt_veda_rag.ingestion_method NOT NULL DEFAULT 'manual_upload',
    approval_status         cmt_veda_rag.ingestion_state NOT NULL DEFAULT 'discovered',
    approved_by             TEXT,
    approved_at             TIMESTAMPTZ,
    knowledge_version       TEXT,
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_documents_genes ON cmt_veda_rag.documents USING GIN (genes);
CREATE INDEX IF NOT EXISTS idx_documents_subtypes ON cmt_veda_rag.documents USING GIN (cmt_subtypes);
CREATE INDEX IF NOT EXISTS idx_documents_pmid ON cmt_veda_rag.documents (pmid);
CREATE INDEX IF NOT EXISTS idx_documents_doi ON cmt_veda_rag.documents (doi);
CREATE INDEX IF NOT EXISTS idx_documents_approval_status ON cmt_veda_rag.documents (approval_status);

-- One row per chunk. Lexical (tsvector) search lives here so Postgres full
-- text search can be the "lexical" side of hybrid retrieval (item 9: keep
-- hybrid retrieval + deterministic reranking, no neural reranker yet).
-- ONLY documents with approval_status = 'indexed' are retrievable — see
-- db.py's PostgresBackend.lexical_search(), which joins against
-- `documents` and filters on this column. This table's rows persisting
-- after a document is REMOVED is fine: removal correctness comes from
-- the JOIN filter, not from deleting chunk rows (final hardening brief
-- item 2's "removed documents must immediately stop contributing to
-- retrieval even if stale vectors remain" — the same principle applies
-- to stale Postgres rows, and the JOIN filter handles it identically).
CREATE TABLE IF NOT EXISTS cmt_veda_rag.chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id     UUID NOT NULL REFERENCES cmt_veda_rag.documents(document_id) ON DELETE CASCADE,
    section         TEXT NOT NULL DEFAULT 'body',   -- abstract/introduction/methods/... /table_caption
    chunk_order     INTEGER NOT NULL DEFAULT 0,
    content         TEXT NOT NULL,
    content_tsv     TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding_id    TEXT,                            -- FAISS row id / external vector store key
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id ON cmt_veda_rag.chunks (document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON cmt_veda_rag.chunks USING GIN (content_tsv);

-- Discovery candidate intake, prior to admin review (section 19).
CREATE TABLE IF NOT EXISTS cmt_veda_rag.discovery_candidates (
    discovery_candidate_id  TEXT PRIMARY KEY,
    proposed_title           TEXT,
    proposed_source_url      TEXT,
    discovered_at             TIMESTAMPTZ NOT NULL DEFAULT now(),
    reviewed_by                TEXT,
    reviewed_at                TIMESTAMPTZ,
    review_decision            TEXT,   -- e.g. 'approved', 'rejected', 'rejected_duplicate', 'rejected_invalid'
    linked_document_id         UUID REFERENCES cmt_veda_rag.documents(document_id)
);

-- Full audit trail of every lifecycle transition (section 42-43). Never
-- delete rows here, even when a document is 'removed'.
CREATE TABLE IF NOT EXISTS cmt_veda_rag.ingestion_audit (
    id              BIGSERIAL PRIMARY KEY,
    document_id     UUID NOT NULL REFERENCES cmt_veda_rag.documents(document_id) ON DELETE CASCADE,
    from_state      cmt_veda_rag.ingestion_state,
    to_state        cmt_veda_rag.ingestion_state NOT NULL,
    actor           TEXT NOT NULL,
    occurred_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_ingestion_audit_document_id ON cmt_veda_rag.ingestion_audit (document_id);

-- Per-answer audit trail (section 42), so any answer can be traced back to
-- exactly what was retrieved/cited under which knowledge version. Do not
-- expose this table's contents to patient/student-facing responses.
CREATE TABLE IF NOT EXISTS cmt_veda_rag.answer_audit (
    id                  BIGSERIAL PRIMARY KEY,
    asked_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    persona             TEXT NOT NULL,
    query_scope         TEXT,
    knowledge_version   TEXT,
    retrieved_chunk_ids UUID[],
    cited_source_ids    TEXT[],
    model_version       TEXT,
    insufficient_evidence BOOLEAN NOT NULL DEFAULT FALSE
);

-- Knowledge version bookkeeping (section 32).
CREATE TABLE IF NOT EXISTS cmt_veda_rag.knowledge_versions (
    knowledge_version   TEXT PRIMARY KEY,
    built_at            TIMESTAMPTZ NOT NULL DEFAULT now(),
    document_count      INTEGER,
    chunk_count         INTEGER,
    notes               TEXT
);

-- Multi-turn conversation support. Conversation history is used ONLY to
-- resolve a follow-up question into a standalone one for RETRIEVAL
-- (see conversation.py) — it is never fed into the generation prompt as
-- fact, which is what keeps every answer grounded in freshly retrieved
-- evidence rather than the model's memory of earlier turns.
--
-- PRIVACY NOTE: patient/caregiver questions may themselves be sensitive
-- even though the scientific RAG corpus stays PHI-free (section 30/AG).
-- This table is a new sensitive-data surface that didn't exist before
-- multi-turn support — there is no automatic retention/expiry policy
-- implemented here; `delete_conversation()` exists so a "clear my chat"
-- action has something real to call, but a scheduled cleanup job (e.g.
-- delete conversations with last_activity_at older than N days) still
-- needs to be added operationally before this goes to production with
-- real patient traffic.
CREATE TABLE IF NOT EXISTS cmt_veda_rag.conversations (
    conversation_id     TEXT PRIMARY KEY,
    user_id             TEXT,   -- TRUSTED value from Veda's authenticated backend,
                                 -- same trust boundary as app_role (see db.py's
                                 -- create_conversation docstring)
    title               TEXT,   -- derived from the first question; stable across the
                                 -- conversation's life (INSERT ... ON CONFLICT DO NOTHING)
    persona             TEXT NOT NULL,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_activity_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS cmt_veda_rag.conversation_turns (
    turn_id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id        TEXT NOT NULL REFERENCES cmt_veda_rag.conversations(conversation_id) ON DELETE CASCADE,
    turn_order              INTEGER NOT NULL,
    question                 TEXT NOT NULL,
    rewritten_question        TEXT NOT NULL,
    scope                      TEXT,
    answer                      TEXT NOT NULL,
    cited_source_ids             TEXT[],
    insufficient_evidence         BOOLEAN NOT NULL DEFAULT FALSE,
    created_at                     TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_turns_conversation_id ON cmt_veda_rag.conversation_turns (conversation_id, turn_order);
CREATE INDEX IF NOT EXISTS idx_conversations_last_activity ON cmt_veda_rag.conversations (last_activity_at);
CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON cmt_veda_rag.conversations (user_id, last_activity_at);
