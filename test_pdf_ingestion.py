"""
tests/test_pdf_ingestion.py
-----------------------------
Real PDF ingestion test (final hardening brief, item 3). Generates an
actual PDF with `reportlab` (available in this sandbox, no network
needed), then runs it through the exact same `ingest_document()` pipeline
manual text uploads and Discovery use — proving there is no separate
PDF-only code path, and that `extract_text_from_pdf_bytes()` (pypdf) is
no longer just a placeholder for the manually-constructed-text-input
case.

Caveat repeated from ingestion_pipeline.py's docstring: reportlab
produces a simple single-column PDF. This proves the pipeline WIRING
(validate -> extract -> chunk -> embed -> index -> retrieve) works for a
real PDF file, not that pypdf's extraction is robust against the layout
quirks of real publisher-formatted, multi-column scientific PDFs. That
would need testing against your actual corpus.

Run with: python3 tests/test_pdf_ingestion.py
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db import SqliteBackend
from embeddings import HashingTfidfEmbeddings
from vector_store import NumpyVectorStore
from ingestion_pipeline import (
    IngestionInput, ingest_document, IngestionValidationError,
    MAX_PDF_BYTES, _validate_pdf_bytes,
)
from retrieval import HybridRetriever, assess_evidence_sufficiency
from config import IngestionState


def _make_test_pdf(text_paragraphs: list[str]) -> bytes:
    """Builds a real, valid PDF in memory using reportlab."""
    import io
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    for para in text_paragraphs:
        story.append(Paragraph(para, styles["Normal"]))
        story.append(Spacer(1, 12))
    doc.build(story)
    return buf.getvalue()


def test_pdf_magic_byte_and_size_validation():
    try:
        _validate_pdf_bytes(b"this is not a pdf")
        raise AssertionError("expected rejection of non-PDF bytes")
    except IngestionValidationError as e:
        print("Non-PDF bytes correctly rejected:", e)

    try:
        _validate_pdf_bytes(b"")
        raise AssertionError("expected rejection of empty bytes")
    except IngestionValidationError as e:
        print("Empty bytes correctly rejected:", e)

    try:
        _validate_pdf_bytes(b"%PDF-1.4\n" + b"x" * (MAX_PDF_BYTES + 1))
        raise AssertionError("expected rejection of oversized PDF")
    except IngestionValidationError as e:
        print("Oversized PDF correctly rejected:", e)

    # a real, tiny, valid PDF should pass magic-byte/size validation
    pdf_bytes = _make_test_pdf(["A short valid test PDF about CMT."])
    _validate_pdf_bytes(pdf_bytes)  # should not raise
    print("Valid PDF magic bytes/size: PASS")


def test_real_pdf_end_to_end_ingestion_and_retrieval():
    tmpdir = tempfile.mkdtemp(prefix="cmt_veda_pdf_test_")
    try:
        db = SqliteBackend(os.path.join(tmpdir, "test.db"))
        embedder = HashingTfidfEmbeddings(n_features=512)
        vs = NumpyVectorStore(os.path.join(tmpdir, "vs", "store"), dim=512)

        pdf_bytes = _make_test_pdf([
            "Abstract",
            "CMT4C is an autosomal recessive demyelinating neuropathy caused "
            "by biallelic mutations in SH3TC2, most commonly p.Arg1109X.",
            "Introduction",
            "SH3TC2 encodes a protein required for normal Schwann cell "
            "myelination of peripheral nerves.",
            "Methods",
            "We sequenced SH3TC2 in a cohort of patients with suspected CMT4C.",
            "Results",
            "All CMT4C patients carried biallelic SH3TC2 variants, most "
            "commonly p.Arg1109X, confirming SH3TC2 as the causative gene.",
            "Discussion",
            "These SH3TC2/CMT4C findings implicate HDAC6-mediated "
            "aggresome-autophagy pathways specifically in SH3TC2-related "
            "CMT4C, not in CMT broadly.",
        ])
        print(f"Generated a real {len(pdf_bytes)}-byte PDF via reportlab.")
        assert pdf_bytes.startswith(b"%PDF-"), "reportlab output doesn't look like a real PDF"

        payload = IngestionInput(
            raw_bytes=pdf_bytes, format="pdf",
            title="SH3TC2-Related CMT4C (PDF upload test)",
            doi="10.1000/cmt4c-pdf-test", genes=["SH3TC2"], cmt_subtypes=["CMT4C"],
            source_tier="peer_reviewed", ingestion_method="manual_upload",
        )
        doc = ingest_document(db, embedder, vs, payload, actor="admin-pdf-test")
        assert doc.approval_status == IngestionState.INDEXED, \
            f"expected INDEXED, got {doc.approval_status}"
        print(f"PDF ingested and INDEXED: document_id={doc.document_id} source_id={doc.source_id}")

        chunks = db.get_chunks_for_document(doc.document_id)
        assert len(chunks) > 0, "no chunks were produced from the PDF"
        combined_text = " ".join(c.content for c in chunks).lower()
        assert "sh3tc2" in combined_text, "extracted PDF text doesn't contain expected content"
        assert "p.arg1109x" in combined_text.replace(" ", "").lower() or "arg1109x" in combined_text.lower()
        print(f"{len(chunks)} chunks extracted from the real PDF, containing expected text.")

        # retrievable through the full pipeline, same as any other ingested document
        retriever = HybridRetriever(db, embedder, vs)
        final, scope, ents = retriever.retrieve_and_rank("What is the SH3TC2 gene?")
        assessment = assess_evidence_sufficiency(final, question="What is the SH3TC2 gene?")
        assert assessment.sufficient, "PDF-sourced content should be retrievable and sufficient"
        assert any(c.document_id == doc.document_id for c in assessment.supporting_chunks), \
            "the PDF-ingested document did not surface for a relevant query"
        print("PDF-sourced content is retrievable through the standard pipeline: PASS")

        db.close()
        print("\nREAL PDF INGESTION END-TO-END: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_scanned_image_only_pdf_rejected():
    """A PDF with no extractable text layer (e.g. a scanned image) must
    be rejected, not silently indexed as empty."""
    import io
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    # deliberately draw nothing (or only non-text graphics) -> no text layer
    c.showPage()
    c.save()
    empty_pdf_bytes = buf.getvalue()

    tmpdir = tempfile.mkdtemp(prefix="cmt_veda_pdf_empty_test_")
    try:
        db = SqliteBackend(os.path.join(tmpdir, "test.db"))
        embedder = HashingTfidfEmbeddings(n_features=256)
        vs = NumpyVectorStore(os.path.join(tmpdir, "vs", "store"), dim=256)
        payload = IngestionInput(raw_bytes=empty_pdf_bytes, format="pdf", title="Blank PDF")
        try:
            ingest_document(db, embedder, vs, payload, actor="admin")
            raise AssertionError("expected rejection of a text-less PDF")
        except IngestionValidationError as e:
            print("Text-less/scanned PDF correctly rejected rather than indexed empty:", e)
        db.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        print(f"\n--- {t.__name__} ---")
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        sys.exit(1)
