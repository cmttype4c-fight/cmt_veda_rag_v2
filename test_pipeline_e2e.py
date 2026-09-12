"""
tests/test_pipeline_e2e.py
---------------------------
Real, executed end-to-end tests against SqliteBackend + HashingTfidfEmbeddings
+ NumpyVectorStore + MockGroundedGenerator (the sandbox-executable stack).
No fastapi/pydantic/psycopg2/faiss/llama_cpp involved — these tests exercise
db.py, embeddings.py, vector_store.py, ingestion_pipeline.py, discovery.py,
and retrieval.py directly, which is everything except the HTTP layer itself.

Covers the completion follow-up's mandatory tests:
  AK (37) — the 9 required questions + one deliberately unsupported question
  AN (40) — removal actually prevents retrieval
  AO (41) — Discovery approved vs not-approved
  AP (42) — manual ingestion + Source ID persistence across a restart

What this file does NOT cover (and why):
  - AL (38) role-based *language* differences: that's a property of the real
    LLM's output, not of this pipeline. MockGroundedGenerator does extractive
    stitching, not persona-aware prose, so testing "researcher answer is
    deeper" against it would be meaningless. What IS tested here is that the
    correct PERSONA is resolved and that persona-filtered source display
    would receive the right data (full metadata is always available to
    main.py's _persona_filtered_source — see DELIVERABLES.md for why the
    filtering step itself isn't executable in this sandbox).
  - AM (39) HTTP-level source-protection tests (hitting `/files/*` etc.):
    needs a running FastAPI server; see tests/test_regression.py's
    structural checks (route topology) and benchmark.py's adversarial
    section for what to run against a real deployment.

Run with: pytest tests/test_pipeline_e2e.py -v
(Falls back to a plain __main__ runner at the bottom if pytest isn't
installed — that's what was actually used to execute this file in the
sandbox that built it.)
"""

import os
import sys
import shutil
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db import SqliteBackend
from embeddings import HashingTfidfEmbeddings
from vector_store import NumpyVectorStore
from ingestion_pipeline import IngestionInput, ingest_document, remove_document
from discovery import (register_candidate, approve_and_ingest, reject_candidate,
                        DiscoveryCandidatePayload, DiscoveryRejectionError)
from retrieval import HybridRetriever, assess_evidence_sufficiency
from generation import build_messages, validate_citations, get_generator
from config import Persona, resolve_persona, IngestionState


def _fresh_env():
    tmpdir = tempfile.mkdtemp(prefix="cmt_veda_e2e_")
    db_path = os.path.join(tmpdir, "test.db")
    vs_path = os.path.join(tmpdir, "vs", "store")
    db = SqliteBackend(db_path)
    embedder = HashingTfidfEmbeddings(n_features=512)
    vs = NumpyVectorStore(vs_path, dim=512)
    return tmpdir, db_path, vs_path, db, embedder, vs


def _seed_corpus(db, embedder, vs):
    """A small synthetic corpus covering the mandatory-question topics."""
    docs = [
        dict(raw_text=(
            "Abstract\nCharcot-Marie-Tooth (CMT) disease is a group of "
            "inherited peripheral neuropathies affecting the motor and "
            "sensory nerves, causing progressive muscle weakness and "
            "sensory loss, typically starting in the feet and legs. " * 3
        ), title="CMT: A Clinical Overview", source_tier="review"),

        dict(raw_text=(
            "Abstract\nThe causes of CMT are mutations in genes affecting "
            "peripheral nerve myelin or axons; CMT is caused by mutations "
            "most commonly inherited in an autosomal dominant pattern, "
            "though autosomal recessive and X-linked forms also occur.\n\n"
            "Introduction\nUnderstanding what causes CMT starts with the "
            "peripheral nerve. CMT is caused by disruption of the myelin "
            "sheath or the axon itself, depending on which genes are "
            "affected.\n\n"
            "Discussion\nCommon genes known to cause CMT include PMP22, "
            "GJB1, and MFN2. Duplication of PMP22 is the single most "
            "common cause of CMT overall."
        ), title="Genetic Causes of CMT", source_tier="review"),

        dict(raw_text=(
            "Abstract\nCommon symptoms of CMT include distal muscle "
            "weakness, foot deformities such as high arches (pes cavus), "
            "hammertoes, reduced tendon reflexes, and sensory loss in the "
            "feet and hands. " * 3
        ), title="Clinical Symptoms of CMT", source_tier="review"),

        dict(raw_text=(
            "Abstract\nCMT4C is an autosomal recessive demyelinating "
            "neuropathy caused by biallelic mutations in SH3TC2. The "
            "p.Arg1109X mutation disrupts Schwann cell myelination via "
            "impaired protein degradation, involving aggresome formation, "
            "autophagy, and HDAC6-dependent pathways specific to this "
            "subtype.\n\n"
            "Introduction\nSH3TC2 encodes a protein required for normal "
            "Schwann cell myelination of peripheral nerves. Biallelic "
            "SH3TC2 mutations are the most common cause of CMT4C.\n\n"
            "Methods\nWe sequenced SH3TC2 in a cohort of patients with "
            "suspected CMT4C and characterized SH3TC2 protein "
            "localization in patient-derived Schwann cells.\n\n"
            "Results\nAll CMT4C patients carried biallelic SH3TC2 "
            "variants, most commonly p.Arg1109X, confirming SH3TC2 as the "
            "causative gene.\n\n"
            "Discussion\nThese SH3TC2/CMT4C findings implicate "
            "HDAC6-mediated aggresome-autophagy pathways specifically in "
            "SH3TC2-related CMT4C, not in CMT broadly."
        ), title="SH3TC2-Related CMT4C: Molecular Mechanisms",
            doi="10.1000/cmt4c", genes=["SH3TC2", "HDAC6"], cmt_subtypes=["CMT4C"],
            source_tier="peer_reviewed"),

        dict(raw_text=(
            "Abstract\nCMT is diagnosed through a combination of clinical "
            "examination, nerve conduction studies showing reduced "
            "conduction velocity, and confirmatory genetic testing for "
            "known causative variants. " * 3
        ), title="Diagnostic Approach to CMT", source_tier="review"),

        dict(raw_text=(
            "Abstract\nThere is currently no cure for CMT. Management is "
            "supportive and evidence-based, including physical therapy, "
            "orthotic devices such as ankle-foot orthoses, and orthopedic "
            "surgery for severe foot deformities; a randomized trial of "
            "ascorbic acid showed no significant benefit for CMT1A. " * 3
        ), title="Supportive Management and Evidence Review in CMT",
            source_tier="review"),

        dict(raw_text=(
            "Commentary\nSeveral repurposed compounds, including Drug Y, "
            "have been informally discussed as possible future candidates "
            "worth exploring in CMT models, though no studies exist yet. " * 3
        ), title="Future Directions Commentary"),
    ]
    for d in docs:
        ingest_document(db, embedder, vs, IngestionInput(**d), actor="test-admin")


def test_ak_general_cmt_question():
    tmpdir, *_ , db, embedder, vs = _fresh_env()
    try:
        _seed_corpus(db, embedder, vs)
        retriever = HybridRetriever(db, embedder, vs)
        final, scope, ents = retriever.retrieve_and_rank("What is Charcot-Marie-Tooth disease?")
        assessment = assess_evidence_sufficiency(final, question="What is Charcot-Marie-Tooth disease?")
        assert assessment.sufficient
        from collections import Counter
        counts = Counter(c.document_id for c in final)
        assert max(counts.values()) <= 2, f"general question dominated by one doc: {counts}"
        print("AK/general-CMT: PASS (answerable, not dominated by one narrow paper)")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_ak_mandatory_questions_answerable():
    tmpdir, *_ , db, embedder, vs = _fresh_env()
    try:
        _seed_corpus(db, embedder, vs)
        retriever = HybridRetriever(db, embedder, vs)
        questions = [
            "What is Charcot-Marie-Tooth disease?",
            "What causes CMT?",
            "What are common symptoms of CMT?",
            "What is CMT4C?",
            "What is SH3TC2?",
            "What is p.Arg1109X?",
            "What role does HDAC6 play in CMT?",
            "How is CMT diagnosed?",
        ]
        for q in questions:
            final, scope, ents = retriever.retrieve_and_rank(q)
            assessment = assess_evidence_sufficiency(final, question=q)
            assert assessment.sufficient, f"expected answerable: {q!r} (scope={scope})"
        print(f"AK/{len(questions)} mandatory questions: all answerable")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_ak_treatment_question_grounded_not_overclaimed():
    """'What treatments are supported by evidence for CMT?' — the corpus
    only has a supportive-care review (including a *negative* trial
    result). This should be answerable, and the retrieved text should not
    be a fabricated positive-outcome claim (nothing to fabricate here since
    we never call a real LLM — this checks the RETRIEVED evidence itself
    doesn't overclaim, which is what the LLM would be grounded on)."""
    tmpdir, *_ , db, embedder, vs = _fresh_env()
    try:
        _seed_corpus(db, embedder, vs)
        retriever = HybridRetriever(db, embedder, vs)
        q = "What treatments are supported by evidence for CMT?"
        final, scope, ents = retriever.retrieve_and_rank(q)
        assessment = assess_evidence_sufficiency(final, question=q)
        assert assessment.sufficient
        combined = " ".join(c.text for c in assessment.supporting_chunks).lower()
        assert "no cure" in combined or "no significant" in combined or "supportive" in combined
        print("AK/treatment question: retrieved evidence is honest about limited efficacy, not fabricated")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_ak_unsupported_question_refuses():
    """The deliberately unsupported question required by section 37/AK:
    the corpus only ever *mentions* Drug Y informally, never with outcome
    data — must refuse rather than guess."""
    tmpdir, *_ , db, embedder, vs = _fresh_env()
    try:
        _seed_corpus(db, embedder, vs)
        retriever = HybridRetriever(db, embedder, vs)
        q = "What is the long-term efficacy of Drug Y in CMT?"
        final, scope, ents = retriever.retrieve_and_rank(q)
        assessment = assess_evidence_sufficiency(final, question=q)
        assert not assessment.sufficient, "expected refusal for unsupported Drug Y efficacy question"
        print("AK/unsupported question: correctly refuses rather than guessing")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_ak_specific_sh3tc2_question_gets_specific_evidence():
    tmpdir, *_ , db, embedder, vs = _fresh_env()
    try:
        _seed_corpus(db, embedder, vs)
        retriever = HybridRetriever(db, embedder, vs)
        final, scope, ents = retriever.retrieve_and_rank("What is the SH3TC2 gene?")
        assert scope == "specific"
        assert "SH3TC2" in ents.genes
        combined = " ".join(c.text for c in final).lower()
        assert "sh3tc2" in combined
        print("AK/SH3TC2-specific question: correctly retrieves SH3TC2-specific evidence")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_an_removal_actually_prevents_retrieval():
    """Section AN (40), mandatory: index A, query successfully, remove A,
    query again — A must not contribute."""
    tmpdir, *_ , db, embedder, vs = _fresh_env()
    try:
        payload = IngestionInput(
            raw_text="Abstract\nCMT4C is caused by SH3TC2 mutations affecting Schwann cell myelination. " * 4,
            title="A removable CMT4C paper", doi="10.9999/removable",
            genes=["SH3TC2"], cmt_subtypes=["CMT4C"],
        )
        doc = ingest_document(db, embedder, vs, payload, actor="admin")
        assert doc.approval_status == IngestionState.INDEXED

        retriever = HybridRetriever(db, embedder, vs)
        before, *_ = retriever.retrieve_and_rank("What is CMT4C?")
        assert any(c.document_id == doc.document_id for c in before), "should find it before removal"

        remove_document(db, doc.document_id, actor="admin")

        after, *_ = retriever.retrieve_and_rank("What is CMT4C?")
        assert not any(c.document_id == doc.document_id for c in after), "removed document still retrievable!"
        print("AN/removal: document retrievable before removal, NOT retrievable after — PASS")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_ao_discovery_approved_vs_not_approved():
    """Section AO (41), mandatory."""
    tmpdir, *_ , db, embedder, vs = _fresh_env()
    try:
        register_candidate(db, DiscoveryCandidatePayload(
            discovery_candidate_id="disc-approved", proposed_title="Approved paper",
        ))
        doc = approve_and_ingest(
            db, embedder, vs, "disc-approved",
            source_text="Abstract\nCMT is an inherited peripheral neuropathy. " * 6,
            metadata=IngestionInput(raw_text="", title="Approved paper"),
            reviewed_by="admin",
        )
        assert doc.approval_status == IngestionState.INDEXED

        register_candidate(db, DiscoveryCandidatePayload(
            discovery_candidate_id="disc-not-approved", proposed_title="Never reviewed paper",
        ))
        # never approved or rejected — just sits as a registered candidate.
        all_docs = db.list_documents()
        linked_doc_ids = {d.discovery_candidate_id for d in all_docs}
        assert "disc-not-approved" not in linked_doc_ids
        assert "disc-approved" in linked_doc_ids
        print("AO/Discovery: approved candidate indexed; unapproved candidate never entered production RAG — PASS")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


def test_ap_manual_ingestion_and_source_id_persist_across_restart():
    """Section AP (42), mandatory: admin upload -> approval -> ingestion ->
    indexing -> retrieval, then verify metadata + Source ID survive a
    (simulated) restart."""
    tmpdir, db_path, vs_path, db, embedder, vs = _fresh_env()
    try:
        payload = IngestionInput(
            raw_text="Abstract\nManual upload test document about CMT diagnosis via nerve conduction studies. " * 4,
            title="Manually uploaded diagnostic paper", source_tier="review",
        )
        doc = ingest_document(db, embedder, vs, payload, actor="admin-manual")
        assert doc.approval_status == IngestionState.INDEXED
        original_source_id = doc.source_id
        original_doc_id = doc.document_id
        db.close()

        # --- simulate restart: brand new process-like objects, same files ---
        db2 = SqliteBackend(db_path)
        vs2 = NumpyVectorStore(vs_path, dim=512)

        refetched = db2.get_document(original_doc_id)
        assert refetched.source_id == original_source_id, "Source ID changed across restart!"
        assert refetched.approval_status == IngestionState.INDEXED

        retriever2 = HybridRetriever(db2, embedder, vs2)
        results, *_ = retriever2.retrieve_and_rank("How is CMT diagnosed?")
        assert any(c.document_id == original_doc_id for c in results), "not retrievable after restart!"
        print(f"AP/manual ingestion + restart: Source ID {original_source_id} stable, still retrievable — PASS")
        db2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_role_mapping_still_server_authoritative():
    assert resolve_persona("patient") == Persona.STUDENT
    assert resolve_persona("clinician") == Persona.CLINICIAN
    assert resolve_persona("researcher") == Persona.RESEARCHER
    assert resolve_persona("anything-else") == Persona.STUDENT
    print("Role mapping (section 16/P): PASS")


def test_generation_plumbing_and_citation_validation_together():
    """Not a claim about real-model grounding quality (see module
    docstring) — proves the retrieval -> prompt -> generate -> citation
    validation chain is wired correctly end-to-end."""
    tmpdir, *_ , db, embedder, vs = _fresh_env()
    try:
        _seed_corpus(db, embedder, vs)
        retriever = HybridRetriever(db, embedder, vs)
        final, scope, ents = retriever.retrieve_and_rank("What are common symptoms of CMT?")
        assessment = assess_evidence_sufficiency(final, question="What are common symptoms of CMT?")
        assert assessment.sufficient
        messages, max_tokens = build_messages(
            "What are common symptoms of CMT?", assessment.supporting_chunks,
            Persona.STUDENT, "detailed", "auto",
        )
        generator = get_generator("mock")
        raw_answer = generator.generate(messages, max_tokens)
        retrieved_ids = {c.source_id for c in assessment.supporting_chunks}
        validated = validate_citations(raw_answer, retrieved_ids)
        assert validated.cited_source_ids, "expected at least one valid citation"
        assert not validated.dropped_unsupported_citations
        print("End-to-end plumbing (retrieval->prompt->generate->citation-validate): PASS")
    finally:
        db.close(); shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failures = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    if failures:
        sys.exit(1)
