"""
tests/test_context_and_lexical_fixes.py
------------------------------------------
Formal regression tests for the fixes made in response to a real VPS
deployment report (context-overflow crash + persistent "insufficient
evidence" on basic questions). Each test corresponds to one specific,
previously-ad-hoc verification — checked in here so these don't silently
regress.

Run with: python3 tests/test_context_and_lexical_fixes.py
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from db import SqliteBackend
from embeddings import HashingTfidfEmbeddings
from vector_store import NumpyVectorStore
from ingestion_pipeline import IngestionInput, ingest_document, extract_text_from_pdf_bytes
from retrieval import HybridRetriever, Candidate
from generation import (
    fit_candidates_to_token_budget, ContextBudgetError, get_generator,
)
from config import Persona


def _fresh_env():
    tmpdir = tempfile.mkdtemp(prefix="cmt_veda_ctxfix_test_")
    db = SqliteBackend(os.path.join(tmpdir, "test.db"))
    embedder = HashingTfidfEmbeddings(n_features=512)
    vs = NumpyVectorStore(os.path.join(tmpdir, "vs", "store"), dim=512)
    return tmpdir, db, embedder, vs


# ---------------------------------------------------------------------
# Token-based context budget (the actual crash: "Requested tokens
# (16023) exceed context window of 4096")
# ---------------------------------------------------------------------
def test_context_budget_drops_least_relevant_candidates_first():
    generator = get_generator("mock")
    big_text = "SH3TC2 CMT4C aggresome autophagy HDAC6 pathway discussion. " * 40
    candidates = [
        Candidate(chunk_id=f"c{i}", document_id=f"d{i}", source_id=f"CMT-RAG-{i:06d}",
                  text=big_text, section="body", metadata={})
        for i in range(12)
    ]
    messages, max_tokens, final, dropped = fit_candidates_to_token_budget(
        "What is Charcot-Marie-Tooth disease?", candidates, Persona.STUDENT,
        "detailed", "auto", generator, n_ctx=4096, safety_margin_tokens=128,
    )
    prompt_tokens = generator.count_tokens(messages[0]["content"] + "\n" + messages[1]["content"])
    assert prompt_tokens + max_tokens + 128 <= 4096
    assert dropped > 0
    assert [c.source_id for c in final] == [c.source_id for c in candidates[:len(final)]], \
        "must drop from the END (least relevant), keeping the most relevant candidates"
    print(f"Dropped {dropped}/12 oversized candidates, result fits n_ctx: PASS")


def test_context_budget_raises_when_even_zero_candidates_dont_fit():
    generator = get_generator("mock")
    candidates = [Candidate(chunk_id="c1", document_id="d1", source_id="CMT-RAG-000001",
                             text="some evidence text", section="body", metadata={})]
    try:
        fit_candidates_to_token_budget(
            "What is CMT?", candidates, Persona.STUDENT, "deep", "auto",
            generator, n_ctx=10, safety_margin_tokens=5,
        )
        raise AssertionError("expected ContextBudgetError for an impossibly small n_ctx")
    except ContextBudgetError:
        print("Correctly raised ContextBudgetError rather than silently truncating: PASS")


def test_context_budget_leaves_well_fitting_prompts_untouched():
    generator = get_generator("mock")
    candidates = [Candidate(chunk_id="c1", document_id="d1", source_id="CMT-RAG-000001",
                             text="CMT is inherited.", section="body", metadata={})]
    messages, max_tokens, final, dropped = fit_candidates_to_token_budget(
        "What is CMT?", candidates, Persona.STUDENT, "detailed", "auto", generator, n_ctx=4096,
    )
    assert dropped == 0 and len(final) == 1
    print("Well-fitting prompt: nothing dropped: PASS")


# ---------------------------------------------------------------------
# Backend-conditional lexical saturation
# ---------------------------------------------------------------------
def test_lexical_saturation_defaults_by_backend():
    import subprocess
    sqlite_val = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0,'.'); import config; print(config.LEXICAL_SATURATION)"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, env={**os.environ, "RAG_DB_BACKEND": "sqlite"},
    ).stdout.strip()
    postgres_val = subprocess.run(
        [sys.executable, "-c", "import sys; sys.path.insert(0,'.'); import config; print(config.LEXICAL_SATURATION)"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        capture_output=True, text=True, env={**os.environ, "RAG_DB_BACKEND": "postgres"},
    ).stdout.strip()
    assert sqlite_val == "5.0", f"sqlite default changed unexpectedly: {sqlite_val}"
    assert postgres_val == "0.5", f"postgres default changed unexpectedly: {postgres_val}"
    print(f"LEXICAL_SATURATION defaults: sqlite={sqlite_val}, postgres={postgres_val} -- PASS")


# ---------------------------------------------------------------------
# SQLite FTS5: stopword removal + porter stemming
# ---------------------------------------------------------------------
def test_fts5_stopwords_removed_from_query():
    tmpdir, db, embedder, vs = _fresh_env()
    try:
        sanitized = db._sanitize_fts_query("What is Charcot-Marie-Tooth disease?")
        assert "what" not in sanitized.lower().split(" or ")
        assert "is" not in [t.strip() for t in sanitized.lower().split(" or ")]
        assert "charcot" in sanitized.lower()
        print(f"Stopwords excluded from FTS5 query: {sanitized!r} -- PASS")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_fts5_porter_stemming_matches_morphological_variants():
    tmpdir, db, embedder, vs = _fresh_env()
    try:
        ingest_document(db, embedder, vs, IngestionInput(
            raw_text="Abstract\nManagement is supportive and evidence-based for CMT patients. " * 4,
            title="Supportive care doc",
        ), actor="test")
        # query uses "supported" (different inflection than "supportive"
        # in the indexed text) -- porter stemming should still match
        hits = db.lexical_search("What treatments are supported for CMT", limit=10)
        assert hits, "porter stemmer should match 'supported' against indexed 'supportive'"
        print(f"Porter stemming matches 'supported' <-> 'supportive': {len(hits)} hit(s) -- PASS")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------
# knowledge_version propagation (found missing during a real deployment)
# ---------------------------------------------------------------------
def test_knowledge_version_propagates_to_stored_document():
    tmpdir, db, embedder, vs = _fresh_env()
    try:
        payload = IngestionInput(
            raw_text="Abstract\nCMT is an inherited peripheral neuropathy. " * 5,
            title="KV propagation test", knowledge_version="v2-2026-09-12",
        )
        doc = ingest_document(db, embedder, vs, payload, actor="test")
        assert doc.knowledge_version == "v2-2026-09-12", \
            f"knowledge_version not propagated: got {doc.knowledge_version!r}"
        print("knowledge_version propagates from IngestionInput to DocumentRecord: PASS")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------
# PDF NUL-character stripping
# ---------------------------------------------------------------------
def test_pdf_text_extraction_strips_nul_characters():
    import io
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf)
    c.drawString(100, 750, "SH3TC2 causes CMT4C")
    c.save()
    text = extract_text_from_pdf_bytes(buf.getvalue())
    assert "\x00" not in text
    print("PDF extraction strips NUL characters: PASS")


# ---------------------------------------------------------------------
# Diagnostic script's preflight checks (empty vs. healthy vector store)
# ---------------------------------------------------------------------
def test_diagnose_retrieval_preflight_detects_empty_vector_store():
    import io
    from contextlib import redirect_stdout
    from diagnose_retrieval import preflight_checks

    tmpdir, db, embedder, vs = _fresh_env()
    try:
        ingest_document(db, embedder, vs, IngestionInput(
            raw_text="Abstract\nCMT is an inherited neuropathy. " * 5, title="doc",
        ), actor="test")

        # Healthy case: vs has vectors
        buf = io.StringIO()
        with redirect_stdout(buf):
            preflight_checks(db, embedder, vs)
        assert "ZERO vectors" not in buf.getvalue()
        print("Preflight: healthy vector store correctly NOT flagged: PASS")

        # Broken case: a totally separate, never-populated vector store,
        # simulating the exact reported bug (migration and validation
        # pointed at different --vector-path/--vector-backend)
        empty_vs = NumpyVectorStore(os.path.join(tmpdir, "empty_vs", "store"), dim=512)
        buf2 = io.StringIO()
        with redirect_stdout(buf2):
            preflight_checks(db, embedder, empty_vs)
        assert "ZERO vectors" in buf2.getvalue()
        print("Preflight: empty vector store (the reported bug scenario) correctly flagged: PASS")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


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
