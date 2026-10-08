-- CMT Veda AI RAG v2 -- migration 0003: scientific content class + source provenance
--
-- Purely additive and idempotent (ADD COLUMN IF NOT EXISTS; no column dropped,
-- renamed or rewritten). Run after schema.sql and 0002 (deploy/migrate.sh
-- applies every migrations/*.sql in sorted order).
--
-- content_type  WHAT KIND of scientific evidence a document is, orthogonal to
--               source_type ('discovery' | 'direct_upload', derived from
--               ingestion_method): research_paper | clinical_trial |
--               genetic_variant | guideline | consensus_statement |
--               outcome_measure. Existing rows are the literature corpus, so
--               they default to research_paper. The allowlist is enforced in
--               the application (config.CONTENT_TYPES) rather than a CHECK, so
--               adding a class does not need another migration.
-- source_*      provenance of the ORIGINAL source the text was extracted from.
--               documents.source_format (migration 0002) now carries the
--               original representation (pdf | xml | html | text | markdown);
--               XML/HTML-derived text is a valid scientific source.

ALTER TABLE cmt_veda_rag.documents
    ADD COLUMN IF NOT EXISTS content_type         TEXT NOT NULL DEFAULT 'research_paper',
    ADD COLUMN IF NOT EXISTS source_mime_type     TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS source_content_hash  TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS source_document_ref  TEXT DEFAULT '',
    ADD COLUMN IF NOT EXISTS extraction_status    TEXT DEFAULT '';

CREATE INDEX IF NOT EXISTS idx_documents_content_type
    ON cmt_veda_rag.documents (content_type);
