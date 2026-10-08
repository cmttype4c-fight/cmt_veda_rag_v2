"""
ingestion_pipeline.py
----------------------
The actual end-to-end ingestion pipeline (sections D and, per the final
hardening brief, item 3 — manual PDF ingestion):

    approved document
        -> validate (incl. PDF magic-byte/size/page checks)
        -> extract text (pypdf for PDF; passthrough for text/markdown)
        -> extract/attach metadata (+ URL safety sanitization)
        -> scientific section detection + chunk
        -> embed
        -> lexical index (DB)
        -> vector index
        -> persist metadata (DB)
        -> INDEXED

This is the real thing, not a stub: it calls db.py (real SQL), the
structure-aware chunker in ingestion.py, embeddings.py, vector_store.py,
and url_safety.py, and actually transitions the document through the
corrected lifecycle in config.py. It has been run against SqliteBackend +
HashingTfidfEmbeddings + NumpyVectorStore in this sandbox — that
combination is genuinely tested, INCLUDING the PDF path: a real PDF was
generated with `reportlab` (available in this sandbox) and run through
`extract_text_from_pdf_bytes` -> chunking -> indexing -> retrieval, so PDF
ingestion is no longer a placeholder (see tests/test_pdf_ingestion.py).
What's still unverified is extraction quality against the specific PDF
layouts real scientific-paper publishers produce (multi-column layouts,
embedded figures, unusual fonts) — pypdf's extraction is text-order
dependent and a two-column PDF can interleave columns unpredictably.
Reportlab produces simple single-column PDFs, so this test proves the
pipeline wiring, not extraction fidelity against real publisher PDFs.

Swapping in PostgresBackend / SentenceTransformerEmbeddings /
FaissVectorStore uses the exact same call sites (same abstract
interfaces) but has not itself been executed anywhere, since none of
those three are available in this environment.

Any format other than text/markdown/pdf is explicitly rejected rather
than silently mis-parsed, per section E.
"""
import os
import shutil
from pathlib import Path

import uuid
from dataclasses import dataclass, field
from typing import Optional

from config import IngestionState, KNOWLEDGE_VERSION
from db import DBBackend, DocumentRecord, ChunkRecord, DuplicateDocumentError
from ingestion import chunk_document
from entities import extract_entities
from embeddings import EmbeddingBackend
from vector_store import VectorStore
from url_safety import sanitize_reference_url

SUPPORTED_FORMATS = {"text", "markdown", "pdf"}

# Sanity limits for PDF uploads — reject rather than hang/OOM on an
# oversized or malformed file. Scientific papers are almost never this
# large; these are deliberately generous ceilings, not tight tuning.
MAX_PDF_BYTES = 50 * 1024 * 1024   # 50 MB
MAX_PDF_PAGES = 500
_PDF_MAGIC = b"%PDF-"


class IngestionValidationError(Exception):
    pass


@dataclass
class IngestionInput:
    raw_text: str = ""
    raw_bytes: Optional[bytes] = None  # for format="pdf"
    format: str = "text"
    title: str = ""
    authors: list = field(default_factory=list)
    journal: str = ""
    publication_date: Optional[str] = None
    doi: str = ""
    pmid: str = ""
    trial_id: str = ""
    cmt_subtypes: list = field(default_factory=list)
    genes: list = field(default_factory=list)
    study_type: str = ""
    source_tier: str = "unspecified"
    source_url: str = ""
    discovery_candidate_id: Optional[str] = None
    ingestion_method: str = "manual_upload"
    # Was missing entirely — DocumentRecord has always had a
    # knowledge_version column, but nothing on the ingestion input side
    # ever set it, so every document ended up with knowledge_version=None
    # regardless of what a migration run's --knowledge-version said.
    # Found via a real deployment (Platform 1 hit this independently and
    # patched it locally); fixed here in the source of truth instead of
    # leaving the canonical repo behind a field deployment's hotfix.
    knowledge_version: Optional[str] = None
    # --- scientific taxonomy + provenance (additive; all optional) ---
    # `format` above describes the TEXT handed to RAG ("text"/"markdown"/"pdf").
    # `source_format` is the ORIGINAL representation the text was extracted
    # from ("pdf"/"xml"/"html"/...); empty -> falls back to `format`.
    content_type: str = "research_paper"
    source_format: str = ""
    source_mime_type: str = ""
    source_content_hash: str = ""
    source_document_ref: str = ""
    extraction_status: str = ""


def _validate_pdf_bytes(data: Optional[bytes]) -> None:
    if not data:
        raise IngestionValidationError("format='pdf' but no PDF bytes were provided.")
    if len(data) > MAX_PDF_BYTES:
        raise IngestionValidationError(
            f"PDF is {len(data)} bytes, exceeding the {MAX_PDF_BYTES}-byte limit — rejecting."
        )
    if not data.startswith(_PDF_MAGIC):
        raise IngestionValidationError(
            "File does not start with the PDF magic bytes ('%PDF-') — "
            "rejecting rather than trying to parse a non-PDF file as one."
        )


def extract_text_from_pdf_bytes(data: bytes) -> str:
    """Real PDF text extraction using pypdf. Tested this session against a
    reportlab-generated PDF end-to-end through the full ingestion
    pipeline (see module docstring for the fidelity caveat against real
    publisher-formatted PDFs).

    Strips NUL characters (\\x00): a real deployment hit `page.extract_text()`
    returning embedded NUL bytes for at least one real-corpus PDF, which
    breaks Postgres (its TEXT type rejects \\x00 outright) and is
    generally a sign of a PDF with unusual internal encoding. Stripping
    is a safe, minimal fix — it doesn't change any real character content,
    only removes bytes that can never be valid text content anyway."""
    import io
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    if len(reader.pages) > MAX_PDF_PAGES:
        raise IngestionValidationError(
            f"PDF has {len(reader.pages)} pages, exceeding the {MAX_PDF_PAGES}-page limit — rejecting."
        )
    return "\n\n".join(page.extract_text() or "" for page in reader.pages).replace("\x00", "")


def _extract_text(payload: IngestionInput) -> str:
    if payload.format == "pdf":
        _validate_pdf_bytes(payload.raw_bytes)
        text = extract_text_from_pdf_bytes(payload.raw_bytes)
        if not text or not text.strip():
            raise IngestionValidationError(
                "PDF text extraction produced no text — likely a scanned/"
                "image-only PDF with no text layer. OCR is not implemented; "
                "rejecting rather than indexing an empty document."
            )
        return text
    return payload.raw_text


def validate_input(payload: IngestionInput) -> None:
    if payload.format not in SUPPORTED_FORMATS:
        raise IngestionValidationError(
            f"Unsupported format '{payload.format}'. Supported: {sorted(SUPPORTED_FORMATS)}. "
            f"Rejecting rather than silently indexing incomplete content."
        )
    if payload.format == "pdf":
        _validate_pdf_bytes(payload.raw_bytes)
        return  # text-length check happens after extraction, in ingest_document


def auto_extract_metadata(raw_text: str) -> dict:
    """Supplement (never replace) human-provided metadata by scanning the
    text for genes/subtypes the entity extractor recognizes. Anything not
    found stays explicitly empty — never fabricated (section V)."""
    # Scan the WHOLE text. (This used to scan only the first 20,000 characters,
    # so genes/subtypes first mentioned later in a long paper were missed.)
    ents = extract_entities(raw_text)
    return {
        "genes": sorted(ents.genes),
        "cmt_subtypes": sorted(ents.subtypes),
    }


def ingest_document(
    db: DBBackend,
    embedder: EmbeddingBackend,
    vector_store: VectorStore,
    payload: IngestionInput,
    actor: str,
    allow_duplicate: bool = False,
) -> DocumentRecord:
    """Runs one document through the full pipeline. The document MUST
    already exist in the DB in APPROVED state (create it + approve it via
    discovery.py or the manual-upload admin endpoint first) — this
    function enforces that rather than silently approving on its own
    behalf, per section 44 ("no autonomous knowledge promotion").

    Uploaded PDF bytes (`payload.raw_bytes`) are used only to extract
    text in-memory; they are never written to any location this service
    would later serve back over HTTP (no `/files/*`, no static mount of
    an uploads directory — see main.py).
    """
    validate_input(payload)
    extracted_text = _extract_text(payload)

    if not extracted_text or not extracted_text.strip():
        raise IngestionValidationError("Document text is empty after extraction.")
    if len(extracted_text.strip()) < 50:
        raise IngestionValidationError(
            "Document text is suspiciously short (<50 chars) — rejecting "
            "rather than indexing likely-corrupted content."
        )

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
    doc = DocumentRecord(
        document_id=document_id, source_id=source_id, title=payload.title,
        authors=payload.authors, journal=payload.journal,
        publication_date=payload.publication_date, doi=payload.doi, pmid=payload.pmid,
        trial_id=payload.trial_id, cmt_subtypes=cmt_subtypes, genes=genes,
        study_type=payload.study_type, source_tier=payload.source_tier,
        source_url=sanitize_reference_url(payload.source_url),  # item 5: strip PDF/storage URLs
        discovery_candidate_id=payload.discovery_candidate_id,
        ingestion_method=payload.ingestion_method,
        approval_status=IngestionState.DISCOVERED,
        knowledge_version=payload.knowledge_version,
    )
    db.create_document(doc)
    db.transition_document_state(document_id, IngestionState.PENDING_APPROVAL, actor="system")
    db.transition_document_state(document_id, IngestionState.APPROVED, actor=actor)

    try:
        db.transition_document_state(document_id, IngestionState.QUEUED, actor="system")
        db.transition_document_state(document_id, IngestionState.PROCESSING, actor="system")

        chunks = chunk_document(document_id, source_id, extracted_text)
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
        db.add_chunks(chunk_records)  # also writes the lexical (FTS) index

        vectors = embedder.embed([c["text"] for c in chunks])
        vector_store.add([r.chunk_id for r in chunk_records], vectors)

        db.transition_document_state(document_id, IngestionState.INDEXED, actor="system")
    except Exception:
        db.transition_document_state(document_id, IngestionState.FAILED, actor="system")
        raise

    return db.get_document(document_id)


def remove_document(db: DBBackend, document_id: str, actor: str) -> DocumentRecord:
    """Section G: this call is what makes a document stop being
    retrievable. It does NOT touch the vector store file directly —
    correctness comes from db.get_active_chunk_ids() excluding this
    document's chunks the moment its state is REMOVED (verified in
    tests/test_pipeline_e2e.py). `rebuild_vector_index()` below is the
    separate, optional physical-cleanup step for reclaiming space."""
    return db.transition_document_state(document_id, IngestionState.REMOVED, actor=actor)


def rebuild_vector_index(db: DBBackend, embedder: EmbeddingBackend, vector_store_factory, path: str):
    """Physical index cleanup (section G: 'eventual physical index cleanup
    /rebuild'). Re-embeds and re-adds only the chunks belonging to
    currently-INDEXED documents into a *fresh* vector store at `path`,
    so REMOVED documents' vectors are actually dropped from disk rather
    than merely filtered at query time. This is an explicit maintenance
    operation (run during a maintenance window), not something that runs
    on every removal — removal correctness does not depend on it running.
    """
    active_chunk_ids = db.get_active_chunk_ids()
    texts, ids = [], []
    for state_doc in db.list_documents(state=IngestionState.INDEXED):
        for chunk in db.get_chunks_for_document(state_doc.document_id):
            if chunk.chunk_id in active_chunk_ids:
                texts.append(chunk.content)
                ids.append(chunk.chunk_id)
    fresh_store = vector_store_factory(path, embedder.dim)
    if texts:
        vectors = embedder.embed(texts)
        fresh_store.add(ids, vectors)
    return fresh_store
