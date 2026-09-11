"""
discovery.py
------------
The actual Discovery Engine <-> RAG contract (completion follow-up,
section F). This is real, callable, tested logic — not an in-memory state
transition standing in for the integration.

What this sandbox CAN prove: the validation/rejection rules, the
approval-gate enforcement, duplicate detection, and the linkage between
`discovery_candidate_id` and the resulting document — all exercised in
tests/test_pipeline_e2e.py against SqliteBackend.

What this sandbox CANNOT prove: an actual live HTTP call from your real
Discovery Engine, or fetching source material from a URL it points at (no
network here). The functions below accept already-extracted `source_text`
as an argument rather than reaching out to fetch it themselves — that
matches how the integration should work in production too (Discovery
should push validated, extracted content + metadata to RAG's admin API,
not hand RAG a URL and expect RAG to go fetch untrusted external content
itself). If your Discovery Engine's real contract is "here's a URL, you
fetch it", that fetch step itself is NOT IMPLEMENTED here and needs to be
added server-side with its own validation (content-type check, size
limit, timeout) before being wired to `submit_and_ingest`.
"""

import uuid
from dataclasses import dataclass, field
from typing import Optional

from db import DBBackend, DuplicateDocumentError
from embeddings import EmbeddingBackend
from vector_store import VectorStore
from ingestion_pipeline import IngestionInput, ingest_document, IngestionValidationError


class DiscoveryRejectionError(Exception):
    """Raised for any reason a Discovery candidate cannot enter the RAG —
    not approved, malformed, duplicate, or unauthorized material."""


@dataclass
class DiscoveryCandidatePayload:
    discovery_candidate_id: str
    proposed_title: str
    proposed_source_url: str = ""


def register_candidate(db: DBBackend, payload: DiscoveryCandidatePayload) -> None:
    """Discovery calls this when it finds something — before any admin
    review. Purely bookkeeping; does not touch the document/chunk tables
    at all."""
    if not payload.discovery_candidate_id or not payload.proposed_title:
        raise DiscoveryRejectionError("Malformed discovery candidate: missing id or title.")
    existing = db.get_discovery_candidate(payload.discovery_candidate_id)
    if existing is not None:
        raise DiscoveryRejectionError(
            f"Discovery candidate {payload.discovery_candidate_id} already registered."
        )
    db.create_discovery_candidate(
        payload.discovery_candidate_id, payload.proposed_title, payload.proposed_source_url,
    )


def reject_candidate(db: DBBackend, discovery_candidate_id: str, reviewed_by: str) -> None:
    candidate = db.get_discovery_candidate(discovery_candidate_id)
    if candidate is None:
        raise DiscoveryRejectionError(f"No such discovery candidate: {discovery_candidate_id}")
    db.record_discovery_review(discovery_candidate_id, reviewed_by, "rejected", linked_document_id=None)


def approve_and_ingest(
    db: DBBackend,
    embedder: EmbeddingBackend,
    vector_store: VectorStore,
    discovery_candidate_id: str,
    source_text: str,
    metadata: IngestionInput,
    reviewed_by: str,
) -> "db.DocumentRecord":
    """The actual approve -> ingest path (section F). This is the single
    enforcement point: a document cannot become INDEXED via the Discovery
    path unless it goes through here, and this function refuses:
      * a candidate that was never registered (`register_candidate` first)
      * a candidate that was already reviewed (approved OR rejected) —
        no re-approving/duplicate-ingesting the same candidate
      * missing/empty source text ("inaccessible source material")
      * a document that duplicates one already in the corpus (delegated
        to ingest_document's duplicate check on doi/pmid/title)
    """
    candidate = db.get_discovery_candidate(discovery_candidate_id)
    if candidate is None:
        raise DiscoveryRejectionError(
            f"Cannot approve unregistered discovery candidate: {discovery_candidate_id}"
        )
    if candidate.get("review_decision"):
        raise DiscoveryRejectionError(
            f"Discovery candidate {discovery_candidate_id} was already reviewed "
            f"({candidate['review_decision']}); refusing to re-ingest."
        )
    if not source_text or not source_text.strip():
        raise DiscoveryRejectionError(
            f"Discovery candidate {discovery_candidate_id} has no accessible source "
            f"text — refusing to ingest inaccessible/unretrieved material."
        )

    metadata.discovery_candidate_id = discovery_candidate_id
    metadata.ingestion_method = "discovery"
    metadata.raw_text = source_text

    try:
        doc = ingest_document(db, embedder, vector_store, metadata, actor=reviewed_by)
    except DuplicateDocumentError as e:
        db.record_discovery_review(discovery_candidate_id, reviewed_by, "rejected_duplicate", linked_document_id=None)
        raise DiscoveryRejectionError(str(e)) from e
    except IngestionValidationError as e:
        db.record_discovery_review(discovery_candidate_id, reviewed_by, "rejected_invalid", linked_document_id=None)
        raise DiscoveryRejectionError(str(e)) from e

    db.record_discovery_review(discovery_candidate_id, reviewed_by, "approved", linked_document_id=doc.document_id)
    return doc
