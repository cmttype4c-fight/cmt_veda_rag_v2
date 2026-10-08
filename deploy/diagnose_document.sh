#!/bin/sh
# Read-only state dump for one RAG document: state, retry_count, extracted-text length,
# processing attempts, and audit trail. Needs only psql + RAG_POSTGRES_DSN (never printed).
#   RAG_POSTGRES_DSN=... ./deploy/diagnose_document.sh <document_id|CMT-RAG-000034>
set -eu
: "${RAG_POSTGRES_DSN:?set RAG_POSTGRES_DSN}"
K="${1:?usage: diagnose_document.sh <document_id|source_id>}"
q() { psql "$RAG_POSTGRES_DSN" -X -q -v ON_ERROR_STOP=1 -v k="$K" "$@"; }
q <<'SQL'
\echo == document
SELECT document_id, source_id, approval_status, retry_count, length(coalesce(extracted_text,'')) AS text_len, source_format FROM cmt_veda_rag.documents WHERE document_id::text=:'k' OR source_id=:'k';
\echo == processing_jobs
SELECT p.attempt, p.status, p.rq_job_id, left(coalesce(p.error,''),90) AS error FROM cmt_veda_rag.processing_jobs p JOIN cmt_veda_rag.documents d USING (document_id) WHERE d.document_id::text=:'k' OR d.source_id=:'k' ORDER BY p.attempt, p.rq_job_id;
\echo == audit
SELECT a.from_state, a.to_state, a.actor FROM cmt_veda_rag.ingestion_audit a JOIN cmt_veda_rag.documents d USING (document_id) WHERE d.document_id::text=:'k' OR d.source_id=:'k' ORDER BY a.id;
SQL
