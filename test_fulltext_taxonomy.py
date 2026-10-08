"""
test_fulltext_taxonomy.py
-------------------------
Regression tests for: no scientific-content truncation, XML/HTML/PDF-derived
text accepted with provenance preserved, the scientific content classes
(research_paper / clinical_trial / genetic_variant / ...) kept distinct, and
the PostgreSQL document insert persisting every column.

Runs on the sandbox-executable stack (SqliteBackend + HashingTfidfEmbeddings +
NumpyVectorStore + InlineQueueAdapter), like test_intake_lifecycle.py.
Run: python3 test_fulltext_taxonomy.py   (or pytest)
"""

import contextlib
import dataclasses
import os
import re
import shutil
import sys
import tempfile
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import intake
from config import IngestionState
from db import SqliteBackend, DocumentRecord, PostgresBackend
from embeddings import HashingTfidfEmbeddings
from ingestion import chunk_document, MAX_CHUNK_CHARS
from ingestion_pipeline import IngestionInput, auto_extract_metadata
from vector_store import NumpyVectorStore
from worker import InlineQueueAdapter

TAIL_MARKER = "ZQTAILMARKERXQ"
SENTENCE = ("Cardiovascular autonomic testing in Charcot-Marie-Tooth disease "
            "showed reduced heart-rate variability in PMP22 duplication carriers. ")


def _env():
    tmp = tempfile.mkdtemp(prefix="cmt_ft_")
    db = SqliteBackend(os.path.join(tmp, "t.db"))
    emb = HashingTfidfEmbeddings(n_features=512)
    vs = NumpyVectorStore(os.path.join(tmp, "vs", "s"), dim=512)
    return tmp, db, InlineQueueAdapter(db, emb, vs)


def _long_text(chars: int, paragraphs: bool = True) -> str:
    sep = "\n\n" if paragraphs else " "
    parts, total, i = [], 0, 0
    while total < chars:
        s = f"Section sentence {i}: {SENTENCE * 6}"
        parts.append(s)
        total += len(s) + len(sep)
        i += 1
    return sep.join(parts) + sep + f"Conclusion. The study ends here {TAIL_MARKER}."


def _register(db, text, **kw):
    kw.setdefault("title", "t-" + uuid.uuid4().hex[:6])
    kw.setdefault("doi", "10.9/" + uuid.uuid4().hex[:8])
    st = kw.pop("source_type", "discovery")
    return intake.register_intake(db, IngestionInput(raw_text=text, format="text", **kw), st, actor="t")


# --------------------------------------------------------- no truncation
def test_no_20000_char_truncation_end_to_end():
    tmp, db, adapter = _env()
    try:
        text = _long_text(150_000)
        assert len(text) > 140_000
        doc = _register(db, text, source_format="xml", source_mime_type="application/xml")
        assert len(doc.extracted_text) == len(text) and doc.extracted_text == text   # complete, byte-identical
        doc, _ = intake.approve_intake(db, doc.document_id, "admin", adapter.enqueue_fn)
        adapter.run_pending()
        final = db.get_document(doc.document_id)
        assert final.approval_status == IngestionState.INDEXED, final.approval_status
        assert len(final.extracted_text) == len(text)
        hits = db.lexical_search(TAIL_MARKER, limit=5)
        assert hits, "text beyond the first 20,000 characters must be searchable"
        print(f"PASS: {len(text):,}-char document stored complete, indexed, tail searchable")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def _assert_full_coverage(text, chunks):
    assert chunks, "no chunks"
    # no oversize chunks (the embedder would silently truncate them)
    assert max(len(c["text"]) for c in chunks if c["metadata"].section not in ("table_caption", "figure_caption")) <= MAX_CHUNK_CHARS
    original_words = text.split()
    body = [w for c in chunks if c["metadata"].section not in ("table_caption", "figure_caption")
            for w in c["text"].split()]
    assert body == original_words, "chunking dropped or reordered text"


def test_chunking_paragraphed_long_document_covers_everything():
    text = _long_text(120_000)
    _assert_full_coverage(text, chunk_document("d", "CMT-RAG-000001", text))
    print("PASS: multi-paragraph 120k-char document chunked with full coverage")


def test_chunking_single_giant_paragraph_is_split_not_truncated():
    """XML-derived text is often one block with no blank lines. Previously
    that became ONE multi-thousand-char chunk (silently truncated at embed)."""
    text = _long_text(100_000, paragraphs=False)
    assert "\n" not in text
    chunks = chunk_document("d", "CMT-RAG-000001", text)
    assert len(chunks) > 50
    _assert_full_coverage(text, chunks)
    print(f"PASS: single 100k-char paragraph split into {len(chunks)} bounded chunks, nothing lost")


def test_chunking_text_with_no_whitespace_or_punctuation():
    text = "A" * 10_000
    chunks = chunk_document("d", "CMT-RAG-000001", text)
    assert "".join(c["text"] for c in chunks) == text
    assert max(len(c["text"]) for c in chunks) <= MAX_CHUNK_CHARS
    print("PASS: pathological no-whitespace text hard-wrapped without loss")


def test_entity_scan_covers_whole_document():
    text = ("Background filler. " * 4000) + " Late in the paper, PMP22 duplication (CMT1A) is discussed."
    assert text.index("PMP22") > 60_000
    meta = auto_extract_metadata(text)
    assert "PMP22" in meta["genes"], meta
    print("PASS: genes first mentioned after 60,000 chars are still extracted")


# ------------------------------------------- XML / HTML / PDF provenance
def test_source_formats_accepted_and_provenance_preserved():
    tmp, db, adapter = _env()
    try:
        for fmt, mime in (("xml", "application/xml"), ("html", "text/html"),
                          ("pdf", "application/pdf"), ("text", "text/plain"), ("markdown", "text/markdown")):
            d = _register(db, _long_text(5000), source_format=fmt, source_mime_type=mime,
                          source_content_hash="sha256:" + uuid.uuid4().hex,
                          source_document_ref="docs/" + fmt + "/ref", extraction_status="success",
                          source_url="https://example.org/paper")
            got = db.get_document(d.document_id)
            assert (got.source_format, got.source_mime_type, got.extraction_status) == (fmt, mime, "success")
            assert got.source_content_hash.startswith("sha256:") and got.source_document_ref.endswith("/ref")
            d, _ = intake.approve_intake(db, d.document_id, "admin", adapter.enqueue_fn)
            adapter.run_pending()
            assert db.get_document(d.document_id).approval_status == IngestionState.INDEXED, fmt
        print("PASS: xml/html/pdf/text/markdown-derived text all indexed; format+mime+hash+ref+status preserved")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_source_format_defaults_to_text_format_and_rejects_unknown():
    tmp, db, adapter = _env()
    try:
        assert _register(db, _long_text(3000)).source_format == "text"      # unchanged legacy behaviour
        try:
            _register(db, _long_text(3000), source_format="docx")
            assert False, "expected InvalidContentType"
        except intake.InvalidContentType:
            pass
        print("PASS: legacy default preserved; unknown source_format rejected")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------------ source taxonomy
def test_content_types_remain_distinct_and_filterable():
    tmp, db, adapter = _env()
    try:
        ids = {}
        ids["research_paper"] = _register(db, _long_text(4000)).document_id
        ids["clinical_trial"] = _register(
            db, "NCT0000000. Recruiting. Phase 2 trial of an investigational therapy in CMT1A. "
                "Sites: Example Hospital. Eligibility: adults 18-65 with genetically confirmed CMT1A.",
            content_type="clinical_trial", trial_id="NCT0000000", source_type="discovery").document_id
        ids["genetic_variant"] = _register(
            db, "NM_001303256.3(MORC2):c.539C>T (p.Thr180Ile). ClinVar classification: pathogenic. "
                "Condition: Charcot-Marie-Tooth disease axonal type 2Z.",
            content_type="genetic_variant", source_type="discovery").document_id
        for kind, did in ids.items():
            assert db.get_document(did).content_type == kind
        for kind, did in ids.items():
            got = db.list_documents_page(content_type=kind)
            assert [d.document_id for d in got] == [did], (kind, got)
        assert len(db.list_documents_page()) == 3
        print("PASS: research_paper / clinical_trial / genetic_variant stay distinct and filterable")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_default_content_type_is_research_paper_for_existing_clients():
    tmp, db, _ = _env()
    try:
        assert _register(db, _long_text(3000)).content_type == "research_paper"
        print("PASS: omitted content_type -> research_paper (existing clients unaffected)")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_invalid_content_type_rejected_before_any_row():
    tmp, db, _ = _env()
    try:
        before = len(db.list_documents_page())
        try:
            _register(db, _long_text(3000), content_type="newsletter_article")
            assert False, "expected InvalidContentType"
        except intake.InvalidContentType:
            pass
        assert len(db.list_documents_page()) == before
        print("PASS: unknown content_type (e.g. newsletter_article) rejected, no row created")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_full_text_floor_applies_to_literature_not_structured_records():
    tmp, db, _ = _env()
    try:
        short = "NM_000001.1(GENE):c.1A>G. ClinVar classification: likely pathogenic. Condition: Charcot-Marie-Tooth disease."
        assert 50 < len(short) < config.MIN_FULL_TEXT_CHARS
        _register(db, short, content_type="genetic_variant")                 # structured -> allowed
        _register(db, short + " Recruiting trial.", content_type="clinical_trial")
        try:
            _register(db, short, content_type="research_paper")              # abstract-like paper -> rejected
            assert False, "expected FullTextRuleViolation"
        except intake.FullTextRuleViolation:
            pass
        try:
            _register(db, "too short", content_type="genetic_variant")       # corrupted-content floor still applies
            assert False
        except Exception as e:
            assert "short" in str(e).lower()
        print("PASS: abstract-length paper rejected; short trial/variant records allowed; <50 chars always rejected")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_source_type_vocabulary_unchanged_and_independent_of_content_type():
    tmp, db, _ = _env()
    try:
        a = _register(db, _long_text(3000), source_type="discovery")
        b = _register(db, _long_text(3000), source_type="direct_upload", content_type="clinical_trial")
        assert (a.source_type, b.source_type) == ("discovery", "direct_upload")
        assert (a.content_type, b.content_type) == ("research_paper", "clinical_trial")
        try:
            _register(db, _long_text(3000), source_type="newsletter")
            assert False
        except intake.InvalidSourceType:
            pass
        print("PASS: discovery/direct_upload unchanged; content_type independent; newsletter still rejected")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_full_lifecycle_for_every_content_type():
    tmp, db, adapter = _env()
    try:
        texts = {
            "research_paper": _long_text(6000),
            "clinical_trial": "Trial NCT0000001 in Charcot-Marie-Tooth disease type 1A. Status: recruiting. " * 3,
            "genetic_variant": "ClinVar: PMP22 deletion. Classification pathogenic. Charcot-Marie-Tooth disease. " * 2,
        }
        for kind, text in texts.items():
            d = _register(db, text, content_type=kind)
            assert d.approval_status == IngestionState.PENDING_APPROVAL        # human approval still required
            assert db.list_documents_page(state=IngestionState.INDEXED, content_type=kind) == []
            d, job = intake.approve_intake(db, d.document_id, "admin", adapter.enqueue_fn)
            assert d.approval_status == IngestionState.QUEUED
            adapter.run_pending()
            assert db.get_document(d.document_id).approval_status == IngestionState.INDEXED, kind
        print("PASS: pending_approval -> approved -> queued -> processing -> indexed for all content types")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


def test_ask_veda_context_labels_source_kind():
    from generation import build_messages
    from retrieval import Candidate
    cands = [Candidate(chunk_id="c1", document_id="d1", source_id="CMT-RAG-000001", text="x",
                       metadata={"source_tier": "unspecified", "content_type": "genetic_variant"})]
    msgs, _ = build_messages("q", cands, next(iter(__import__("generation").PERSONA_INSTRUCTIONS)), "detailed", "auto")
    user = msgs[1]["content"]
    assert "source_kind: genetic_variant" in user
    assert "ClinVar" in msgs[0]["content"] and "never present a database" in msgs[0]["content"].lower()
    print("PASS: evidence blocks carry source_kind; grounding rules forbid blurring ClinVar with papers")


def test_pending_row_with_missing_text_is_repaired_in_place_on_reregister():
    """CMT-RAG-000034 was registered by the buggy Postgres insert: pending, but
    extracted_text empty. Re-registering the same source must repair it (same
    ids, still pending) instead of 409, and a genuine duplicate must still 409."""
    tmp, db, adapter = _env()
    try:
        text = _long_text(30_000)
        d = _register(db, text, doi="10.1/orphan", source_format="xml")
        db.set_intake_extras(d.document_id, extracted_text="", source_format="")   # simulate the old bug
        assert db.get_document(d.document_id).extracted_text == ""
        fixed = _register(db, text, doi="10.1/orphan", source_format="xml", source_mime_type="application/xml")
        assert fixed.document_id == d.document_id and fixed.source_id == d.source_id
        assert fixed.approval_status == IngestionState.PENDING_APPROVAL
        assert fixed.extracted_text == text and fixed.source_format == "xml"
        assert len(db.list_documents_page()) == 1
        try:                                                    # now it has text: a real duplicate
            _register(db, text, doi="10.1/orphan")
            assert False, "expected DuplicateDocumentError"
        except Exception as e:
            assert type(e).__name__ == "DuplicateDocumentError", e
        fixed, _ = intake.approve_intake(db, fixed.document_id, "admin", adapter.enqueue_fn)
        adapter.run_pending()
        assert db.get_document(d.document_id).approval_status == IngestionState.INDEXED
        print("PASS: empty-text pending row repaired in place (same ids) -> approve -> indexed; real duplicates still 409")
    finally:
        db.close(); shutil.rmtree(tmp, ignore_errors=True)


# ------------------------------------------------ PostgreSQL persistence
class _RecordingCursor:
    def __init__(self): self.calls = []
    def execute(self, sql, params=None): self.calls.append((sql, params))


def test_postgres_create_document_persists_every_column():
    """LIVE BUG: PostgresBackend.create_document inserted only the 21 round-3
    columns, so extracted_text (and source_format, retry_count, ...) were
    silently dropped and processing failed with 'No extracted text stored'."""
    rec = _RecordingCursor()
    pg = object.__new__(PostgresBackend)                  # no psycopg2 / server needed

    @contextlib.contextmanager
    def fake_cursor():
        yield rec
    pg._cursor = fake_cursor

    doc = DocumentRecord(document_id=str(uuid.uuid4()), source_id="CMT-RAG-000034", title="t",
                         extracted_text="FULL TEXT BODY " * 5000, source_format="xml",
                         content_type="research_paper", source_mime_type="application/xml",
                         source_content_hash="sha256:abc", source_document_ref="ref", extraction_status="success",
                         retry_count=0, ingestion_method="discovery")
    pg.create_document(doc)
    (sql, params), = rec.calls
    cols = re.search(r"\(([^)]*)\) VALUES", sql).group(1).replace(" ", "").split(",")
    assert len(cols) == len(params) == sql.count("%s"), (len(cols), len(params), sql.count("%s"))
    # every stored DocumentRecord field is inserted (created_at/updated_at default in the DB)
    fields = {f.name for f in dataclasses.fields(DocumentRecord)} - {"created_at", "updated_at"}
    assert fields == set(cols), (fields ^ set(cols))
    assert params[cols.index("extracted_text")] == doc.extracted_text
    assert params[cols.index("source_format")] == "xml"
    assert params[cols.index("content_type")] == "research_paper"
    print("PASS: Postgres INSERT covers every DocumentRecord column incl. extracted_text, provenance, content_type")


def test_postgres_row_mapping_ignores_unknown_columns():
    row = {f.name: getattr(DocumentRecord(document_id="x", source_id="y"), f.name)
           for f in dataclasses.fields(DocumentRecord)}
    row["approval_status"] = "pending_approval"
    row["column_added_by_a_future_migration"] = 1
    doc = PostgresBackend._row_to_document(row)
    assert doc.approval_status == IngestionState.PENDING_APPROVAL
    print("PASS: Postgres row mapping tolerates columns newer than the code")


_TESTS = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]

if __name__ == "__main__":
    failures = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:
            import traceback
            failures += 1
            print(f"FAIL: {t.__name__}: {e!r}")
            traceback.print_exc(limit=3)
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    sys.exit(1 if failures else 0)
