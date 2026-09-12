"""
tests/test_conversation.py
----------------------------
Real tests for multi-turn support: conversation persistence (DB layer),
the heuristic query rewriter in isolation, and an end-to-end before/after
that demonstrates the exact problem from the chat gets fixed — "What
causes it?" as a bare follow-up fails to retrieve CMT4C-specific evidence,
but with conversation-aware rewriting it succeeds.

Run with: python3 tests/test_conversation.py
"""

import os
import sys
import shutil
import tempfile
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from db import SqliteBackend, TurnRecord
from embeddings import HashingTfidfEmbeddings
from vector_store import NumpyVectorStore
from ingestion_pipeline import IngestionInput, ingest_document
from retrieval import HybridRetriever, assess_evidence_sufficiency
from conversation import HeuristicQueryRewriter, RewriteResult


def _fresh_env():
    tmpdir = tempfile.mkdtemp(prefix="cmt_veda_conv_test_")
    db = SqliteBackend(os.path.join(tmpdir, "test.db"))
    embedder = HashingTfidfEmbeddings(n_features=512)
    vs = NumpyVectorStore(os.path.join(tmpdir, "vs", "store"), dim=512)
    return tmpdir, db, embedder, vs


def _seed_cmt4c_and_overview(db, embedder, vs):
    ingest_document(db, embedder, vs, IngestionInput(
        raw_text=(
            "Abstract\nCMT4C is caused by biallelic mutations in SH3TC2, "
            "most commonly p.Arg1109X, disrupting Schwann cell "
            "myelination.\n\nIntroduction\nSH3TC2 encodes a protein "
            "required for normal myelination.\n\nDiscussion\nThese "
            "SH3TC2/CMT4C findings implicate HDAC6-mediated pathways "
            "specific to this subtype, not CMT broadly."
        ),
        title="SH3TC2-Related CMT4C", doi="10.1000/cmt4c-conv-test",
        genes=["SH3TC2"], cmt_subtypes=["CMT4C"], source_tier="peer_reviewed",
    ), actor="test")
    ingest_document(db, embedder, vs, IngestionInput(
        raw_text=(
            "Abstract\nCMT is a group of inherited peripheral "
            "neuropathies affecting motor and sensory nerves, causing "
            "progressive weakness and sensory loss. " * 3
        ),
        title="CMT overview", source_tier="review",
    ), actor="test")


# ---------------------------------------------------------------------
# DB persistence
# ---------------------------------------------------------------------
def test_conversation_persistence():
    tmpdir, db, embedder, vs = _fresh_env()
    try:
        conv_id = str(uuid.uuid4())
        db.create_conversation(conv_id, persona="student")
        db.create_conversation(conv_id, persona="student")  # idempotent, must not raise
        assert db.get_conversation(conv_id) is not None

        db.add_turn(TurnRecord(
            turn_id=str(uuid.uuid4()), conversation_id=conv_id, turn_order=0,
            question="What is CMT4C?", rewritten_question="What is CMT4C?",
            scope="specific", answer="CMT4C is...", cited_source_ids=["CMT-RAG-000001"],
        ))
        db.add_turn(TurnRecord(
            turn_id=str(uuid.uuid4()), conversation_id=conv_id, turn_order=1,
            question="What causes it?", rewritten_question="What causes CMT4C?",
            scope="specific", answer="It is caused by...", cited_source_ids=["CMT-RAG-000001"],
        ))

        turns = db.get_turns(conv_id)
        assert len(turns) == 2
        assert turns[0].question == "What is CMT4C?"
        assert turns[1].rewritten_question == "What causes CMT4C?"
        print("Conversation + turns persisted and retrieved in order: PASS")

        # simulate restart
        db.close()
        db2 = SqliteBackend(os.path.join(tmpdir, "test.db"))
        turns_after_restart = db2.get_turns(conv_id)
        assert len(turns_after_restart) == 2
        print("Conversation history survives a simulated restart: PASS")

        limited = db2.get_turns(conv_id, limit=1)
        assert len(limited) == 1 and limited[0].turn_order == 1
        print("get_turns(limit=...) returns only the most recent turns: PASS")

        db2.delete_conversation(conv_id)
        assert db2.get_conversation(conv_id) is None
        assert db2.get_turns(conv_id) == []
        print("delete_conversation removes the conversation and its turns: PASS")
        db2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------
# Heuristic rewriter in isolation
# ---------------------------------------------------------------------
def test_heuristic_rewriter_resolves_pronoun():
    rewriter = HeuristicQueryRewriter()
    history = [TurnRecord(
        turn_id="t1", conversation_id="c1", turn_order=0,
        question="What is CMT4C?", rewritten_question="What is CMT4C?",
        scope="specific", answer="...",
    )]
    result = rewriter.rewrite("What causes it?", history)
    print(f"'What causes it?' + history -> {result.question!r} (method={result.method})")
    assert result.was_rewritten
    assert "CMT4C" in result.question
    assert "it" not in result.question.lower().split()


def test_heuristic_rewriter_leaves_self_contained_questions_alone():
    rewriter = HeuristicQueryRewriter()
    result = rewriter.rewrite("What is the SH3TC2 gene?", history=[])
    assert not result.was_rewritten
    assert result.question == "What is the SH3TC2 gene?"
    print("Self-contained question (names its own gene) left unchanged: PASS")


def test_heuristic_rewriter_no_history_no_rewrite():
    rewriter = HeuristicQueryRewriter()
    result = rewriter.rewrite("What causes it?", history=[])
    assert not result.was_rewritten, "should not invent an answer with nothing to resolve against"
    print("Follow-up with NO prior history left unresolved rather than guessed: PASS")


def test_heuristic_rewriter_prefers_recent_turn():
    """If the conversation has moved from CMT4C to a different specific
    topic, a bare follow-up should resolve against the MOST RECENT
    specific entity, not an earlier one."""
    rewriter = HeuristicQueryRewriter()
    history = [
        TurnRecord(turn_id="t1", conversation_id="c1", turn_order=0,
                   question="What is CMT4C?", rewritten_question="What is CMT4C?",
                   scope="specific", answer="..."),
        TurnRecord(turn_id="t2", conversation_id="c1", turn_order=1,
                   question="What is CMT1A?", rewritten_question="What is CMT1A?",
                   scope="specific", answer="..."),
    ]
    result = rewriter.rewrite("How is it diagnosed?", history)
    assert "CMT1A" in result.question, f"expected the most recent topic (CMT1A), got: {result.question}"
    print(f"Resolves against the MOST RECENT topic, not an earlier one: {result.question!r} — PASS")


# ---------------------------------------------------------------------
# End-to-end: the exact problem from the chat, before and after
# ---------------------------------------------------------------------
def test_followup_question_before_and_after_rewriting():
    tmpdir, db, embedder, vs = _fresh_env()
    try:
        _seed_cmt4c_and_overview(db, embedder, vs)
        retriever = HybridRetriever(db, embedder, vs)
        rewriter = HeuristicQueryRewriter()

        # Turn 1: a specific question, answered normally.
        turn1_q = "What is CMT4C?"
        final1, scope1, ents1 = retriever.retrieve_and_rank(turn1_q)
        assessment1 = assess_evidence_sufficiency(final1, question=turn1_q)
        assert assessment1.sufficient
        print(f"Turn 1 ({turn1_q!r}): answered, scope={scope1}")

        history = [TurnRecord(
            turn_id="t1", conversation_id="c1", turn_order=0,
            question=turn1_q, rewritten_question=turn1_q, scope=scope1,
            answer="CMT4C is caused by SH3TC2 mutations...",
            cited_source_ids=[c.source_id for c in assessment1.supporting_chunks],
        )]

        # --- BEFORE: bare follow-up with no conversation awareness ---
        followup_q = "What causes it?"
        final_before, scope_before, _ = retriever.retrieve_and_rank(followup_q)
        assessment_before = assess_evidence_sufficiency(final_before, question=followup_q)
        combined_before = " ".join(c.text for c in assessment_before.supporting_chunks).lower()
        sh3tc2_found_before = "sh3tc2" in combined_before
        print(f"\nBEFORE (no rewrite): {followup_q!r} -> scope={scope_before}, "
              f"sufficient={assessment_before.sufficient}, mentions SH3TC2={sh3tc2_found_before}")
        # This is the bug: a bare "What causes it?" carries no entity
        # signal, so it's classified general and doesn't reliably surface
        # the CMT4C-specific answer turn 1 was actually about.
        assert scope_before == "general", "confirms the bug precondition: no entity signal without rewriting"

        # --- AFTER: same follow-up, resolved using conversation history ---
        rewrite_result = rewriter.rewrite(followup_q, history)
        assert rewrite_result.was_rewritten
        print(f"\nAFTER (rewritten): {followup_q!r} -> {rewrite_result.question!r}")

        final_after, scope_after, ents_after = retriever.retrieve_and_rank(rewrite_result.question)
        assessment_after = assess_evidence_sufficiency(final_after, question=rewrite_result.question)
        combined_after = " ".join(c.text for c in assessment_after.supporting_chunks).lower()
        assert scope_after == "specific"
        assert assessment_after.sufficient
        assert "sh3tc2" in combined_after
        print(f"AFTER: scope={scope_after}, sufficient={assessment_after.sufficient}, "
              f"mentions SH3TC2={('sh3tc2' in combined_after)}")

        print("\nFollow-up question handling: confirmed BROKEN without rewriting, "
              "confirmed FIXED with conversation-aware rewriting.")

        db.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------
# User ownership, listing, and resuming a conversation "later"
# ---------------------------------------------------------------------
def test_conversation_title_stable_and_user_scoped():
    tmpdir, db, embedder, vs = _fresh_env()
    try:
        conv_id = str(uuid.uuid4())
        db.create_conversation(conv_id, persona="student", user_id="user-42", title="What is CMT4C?")
        db.add_turn(TurnRecord(
            turn_id=str(uuid.uuid4()), conversation_id=conv_id, turn_order=0,
            question="What is CMT4C?", rewritten_question="What is CMT4C?",
            scope="specific", answer="CMT4C is...", cited_source_ids=["CMT-RAG-000001"],
        ))
        # A later "request" (2nd question) passes a different candidate
        # title — must NOT overwrite the original.
        db.create_conversation(conv_id, persona="student", user_id="user-42", title="What causes it?")
        conv = db.get_conversation(conv_id)
        assert conv["title"] == "What is CMT4C?", f"title drifted to {conv['title']!r}"
        assert conv["user_id"] == "user-42"
        print("Title stable across repeated create_conversation calls: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_list_conversations_for_user_isolated_and_ordered():
    tmpdir, db, embedder, vs = _fresh_env()
    try:
        conv1, conv2, other_user_conv = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
        db.create_conversation(conv1, persona="student", user_id="user-42", title="First chat")
        db.add_turn(TurnRecord(turn_id="t1", conversation_id=conv1, turn_order=0,
                                question="q", rewritten_question="q", scope="general", answer="a"))
        db.create_conversation(conv2, persona="clinician", user_id="user-42", title="Second chat")
        db.add_turn(TurnRecord(turn_id="t2", conversation_id=conv2, turn_order=0,
                                question="q", rewritten_question="q", scope="general", answer="a"))
        db.create_conversation(other_user_conv, persona="student", user_id="user-99", title="Not user-42's")

        listing = db.list_conversations_for_user("user-42")
        ids = [c["conversation_id"] for c in listing]
        assert conv1 in ids and conv2 in ids
        assert other_user_conv not in ids, "cross-user leakage in list_conversations_for_user!"
        assert listing[0]["conversation_id"] == conv2, "expected most-recently-active conversation first"
        print(f"list_conversations_for_user: {len(listing)} conversations, correctly scoped and ordered: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_ownership_authorization_logic():
    from conversation import is_authorized_for_conversation
    assert is_authorized_for_conversation({"user_id": None}, None) is True
    assert is_authorized_for_conversation({"user_id": None}, "anyone") is True
    assert is_authorized_for_conversation({"user_id": "u1"}, "u1") is True
    assert is_authorized_for_conversation({"user_id": "u1"}, "u2") is False
    assert is_authorized_for_conversation({"user_id": "u1"}, None) is False
    print("Ownership authorization (unowned=open, owned=must-match): PASS")


def test_delete_stale_conversations():
    from datetime import datetime, timezone, timedelta
    tmpdir, db, embedder, vs = _fresh_env()
    try:
        fresh_id, stale_id = str(uuid.uuid4()), str(uuid.uuid4())
        db.create_conversation(fresh_id, persona="student", user_id="u1", title="Recent")
        db.create_conversation(stale_id, persona="student", user_id="u1", title="Old")

        old_ts = (datetime.now(timezone.utc) - timedelta(days=60)).isoformat()
        with db._conn:
            db._conn.execute(
                "UPDATE conversations SET last_activity_at = ? WHERE conversation_id = ?",
                (old_ts, stale_id),
            )

        deleted = db.delete_stale_conversations(older_than_days=30)
        assert deleted == 1
        assert db.get_conversation(stale_id) is None
        assert db.get_conversation(fresh_id) is not None
        print("delete_stale_conversations: removes only conversations past the threshold: PASS")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_resume_conversation_after_simulated_restart():
    """The actual feature being asked for: user chats, comes back LATER
    (simulated here as closing the DB connection entirely, then opening a
    brand new one against the same file — as different as a real restart
    gets in a single-process test), finds their conversation via
    list_conversations_for_user, reads the full transcript, and asks a
    NEW follow-up that still resolves correctly against the OLD
    persisted history."""
    from generation import build_messages, validate_citations, get_generator
    from config import Persona

    tmpdir = tempfile.mkdtemp(prefix="cmt_veda_resume_test_")
    db_path = os.path.join(tmpdir, "test.db")
    vs_path = os.path.join(tmpdir, "vs", "store")
    db = SqliteBackend(db_path)
    embedder = HashingTfidfEmbeddings(n_features=512)
    vs = NumpyVectorStore(vs_path, dim=512)
    try:
        _seed_cmt4c_and_overview(db, embedder, vs)
        conv_id = str(uuid.uuid4())
        user_id = "patient-abc123"

        # --- Session 1: user asks one question, then leaves ---
        db.create_conversation(conv_id, persona="student", user_id=user_id, title="What is CMT4C?")
        db.add_turn(TurnRecord(
            turn_id=str(uuid.uuid4()), conversation_id=conv_id, turn_order=0,
            question="What is CMT4C?", rewritten_question="What is CMT4C?",
            scope="specific", answer="CMT4C is caused by SH3TC2 mutations...",
            cited_source_ids=["CMT-RAG-000001"],
        ))
        db.close()
        del db  # nothing left in memory referencing session 1

        # --- (time passes) ---

        # --- Session 2: user returns, e.g. reopens the app days later ---
        db2 = SqliteBackend(db_path)
        vs2 = NumpyVectorStore(vs_path, dim=512)  # also reloaded fresh from disk

        # (a) the client finds the conversation in the user's history
        conversations = db2.list_conversations_for_user(user_id)
        assert len(conversations) == 1
        assert conversations[0]["conversation_id"] == conv_id
        assert conversations[0]["title"] == "What is CMT4C?"
        print(f"Found resumable conversation: {conversations[0]['title']!r}")

        # (b) the client renders the full prior transcript
        transcript = db2.get_turns(conv_id)
        assert len(transcript) == 1
        assert transcript[0].question == "What is CMT4C?"
        print(f"Resumed transcript: {len(transcript)} prior turn(s) rendered")

        # (c) the user asks a NEW follow-up in this resumed session
        retriever = HybridRetriever(db2, embedder, vs2)
        rewriter = HeuristicQueryRewriter()
        generator = get_generator("mock")

        history = db2.get_turns(conv_id, limit=4)
        followup = "What causes it?"
        rewrite_result = rewriter.rewrite(followup, history)
        assert rewrite_result.was_rewritten
        assert "CMT4C" in rewrite_result.question
        print(f"New follow-up in resumed session: {followup!r} -> {rewrite_result.question!r}")

        ranked, scope, ents = retriever.retrieve_and_rank(rewrite_result.question)
        assessment = assess_evidence_sufficiency(ranked, question=rewrite_result.question)
        assert assessment.sufficient

        messages, max_tokens = build_messages(
            rewrite_result.question, assessment.supporting_chunks, Persona.STUDENT, "detailed", "auto",
        )
        raw_answer = generator.generate(messages, max_tokens)
        retrieved_ids = {c.source_id for c in assessment.supporting_chunks}
        validated = validate_citations(raw_answer, retrieved_ids)

        db2.add_turn(TurnRecord(
            turn_id=str(uuid.uuid4()), conversation_id=conv_id, turn_order=len(history),
            question=followup, rewritten_question=rewrite_result.question, scope=scope,
            answer=validated.text, cited_source_ids=validated.cited_source_ids,
        ))

        final_transcript = db2.get_turns(conv_id)
        assert len(final_transcript) == 2
        print("\nFull resume-later flow (list -> render -> continue -> persist): PASS")
        db2.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_two_turn_conversation_via_db_mediated_history():
    """Simulates what main.py's ask() actually does across two separate
    calls — turn 2 reads turn 1's history back from the DATABASE, not
    from an in-memory Python object still lying around. This is the
    thing that matters for the real deployment: each HTTP request is an
    independent process-level call, so persistence-mediated history (not
    just passing a list around) is what has to actually work.
    main.py itself can't be imported/run here (no fastapi) — this
    exercises the exact same sequence of calls its ask() function makes:
    create_conversation -> get_turns -> rewrite -> retrieve_and_rank ->
    assess_evidence_sufficiency -> build_messages -> generate ->
    validate_citations -> add_turn.
    """
    from generation import build_messages, validate_citations, get_generator
    from config import Persona

    tmpdir, db, embedder, vs = _fresh_env()
    try:
        _seed_cmt4c_and_overview(db, embedder, vs)
        retriever = HybridRetriever(db, embedder, vs)
        rewriter = HeuristicQueryRewriter()
        generator = get_generator("mock")
        conv_id = str(uuid.uuid4())

        def simulated_ask(question: str, turn_order: int):
            db.create_conversation(conv_id, persona="student")
            history = db.get_turns(conv_id, limit=4)  # <-- read from DB, not memory
            rewrite_result = rewriter.rewrite(question, history)
            effective_q = rewrite_result.question

            ranked, scope, ents = retriever.retrieve_and_rank(effective_q)
            assessment = assess_evidence_sufficiency(ranked, question=effective_q)
            assert assessment.sufficient, f"expected sufficient evidence for {effective_q!r}"

            messages, max_tokens = build_messages(
                effective_q, assessment.supporting_chunks, Persona.STUDENT, "detailed", "auto",
            )
            raw_answer = generator.generate(messages, max_tokens)
            retrieved_ids = {c.source_id for c in assessment.supporting_chunks}
            validated = validate_citations(raw_answer, retrieved_ids)

            db.add_turn(TurnRecord(
                turn_id=str(uuid.uuid4()), conversation_id=conv_id, turn_order=turn_order,
                question=question, rewritten_question=effective_q, scope=scope,
                answer=validated.text, cited_source_ids=validated.cited_source_ids,
            ))
            return effective_q, validated.text

        # Turn 1: independent "request"
        eff_q1, answer1 = simulated_ask("What is CMT4C?", turn_order=0)
        assert eff_q1 == "What is CMT4C?"  # self-contained, no rewrite needed
        print(f"Turn 1: {eff_q1!r} -> {answer1[:80]}...")

        # Turn 2: a SEPARATE "request" — history comes only from the DB
        eff_q2, answer2 = simulated_ask("What causes it?", turn_order=1)
        assert "CMT4C" in eff_q2, f"expected CMT4C resolved from DB-persisted history, got {eff_q2!r}"
        print(f"Turn 2: 'What causes it?' -> rewritten to {eff_q2!r} -> {answer2[:80]}...")

        # Confirm both turns are durably persisted with the right linkage
        final_history = db.get_turns(conv_id)
        assert len(final_history) == 2
        assert final_history[1].rewritten_question == eff_q2
        print("\nTwo-turn conversation, DB-mediated history: PASS")

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
