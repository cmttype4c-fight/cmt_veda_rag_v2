#!/usr/bin/env python3
"""
scripts/run_qwen_validation.py
--------------------------------
Real-model validation (final hardening brief, item 8). Runs the actual
Qwen2.5-1.5B-Instruct Q4_K_M model — not the mock generator — against the
9 mandatory questions plus one deliberately unsupported question, under
all three personas, and specifically re-tests the exact prior failure:
a general "What is CMT?" question must NOT produce an HDAC6/SH3TC2-heavy
answer.

**This script has not been run.** There is no `llama_cpp` package and no
`.gguf` model file anywhere in the sandbox that built this repo (no
network to fetch either). Nothing in this codebase claims otherwise —
`tests/test_pipeline_e2e.py` and friends all use `MockGroundedGenerator`,
which does simple extractive stitching and proves NOTHING about whether a
real language model follows the grounding rules. This script is what
closes that gap — but only once someone actually runs it against the real
model on the real VPS/staging box, with a real indexed corpus behind it.

Usage (on the machine with the model + a running/importable RAG stack):
    python3 scripts/run_qwen_validation.py \\
        --db-backend sqlite --sqlite-path ./cmt_veda_rag.db \\
        --vector-path ./vector_store/store \\
        --model-path ./models/qwen2.5-1.5b-instruct-q4_k_m.gguf \\
        --output qwen_validation_report.json

What it checks AUTOMATICALLY (no human judgment needed):
  - Every cited Source ID in every answer was actually retrieved
    (delegates to generation.validate_citations — a hallucinated
    citation is a hard failure).
  - The deliberately-unsupported question triggers the exact refusal
    message, not a fabricated answer.
  - For the general "What is CMT?" question specifically: counts how many
    of the words {SH3TC2, HDAC6, aggresome, autophagy, CMT4C} appear in
    the answer text, and flags (does not silently pass) if 2+ of those
    narrow/mechanism-specific terms appear — that combination is exactly
    the fingerprint of the original bug. A flag here needs a HUMAN to
    read the actual answer and judge whether it's a genuine
    over-generalization or an appropriately-scoped mention (e.g. "some
    subtypes, such as CMT4C, additionally involve..." is fine; leading
    the entire answer with SH3TC2/HDAC6 mechanism is the bug).

What still needs a HUMAN to judge (printed for manual review, not
auto-scored):
  - Factual consistency of the SAME underlying facts across student/
    clinician/researcher phrasings of the same question.
  - Role-appropriate terminology and depth actually differing
    (student=accessible, clinician=clinical terms, researcher=deepest).
  - No forced "What is it?/Why?/Treatment?/Prognosis?" sections invented
    when the evidence doesn't support them.
  - General prose quality / whether refusals read naturally rather than
    like an error message glued onto otherwise-fluent text.
"""

import argparse
import json
import sys

sys.path.insert(0, ".")

from db import get_backend
from embeddings import get_embedding_backend
from vector_store import get_vector_store
from retrieval import HybridRetriever, assess_evidence_sufficiency
from generation import build_messages, validate_citations, get_generator, fit_candidates_to_token_budget, ContextBudgetError
from config import Persona, INSUFFICIENT_EVIDENCE_MESSAGE, CONTEXT_SAFETY_MARGIN_TOKENS

MANDATORY_QUESTIONS = [
    "What is Charcot-Marie-Tooth disease?",
    "What causes CMT?",
    "What are common symptoms of CMT?",
    "What is CMT4C?",
    "What is SH3TC2?",
    "What is p.Arg1109X?",
    "What role does HDAC6 play in CMT?",
    "How is CMT diagnosed?",
    "What treatments are supported by evidence for CMT?",
]
# Fill in a question about something genuinely absent from your corpus —
# the placeholder below is deliberately generic; replace it with something
# you know your real corpus has no evidence for.
UNSUPPORTED_QUESTION = "What is the long-term efficacy of [REPLACE ME: a compound not in your corpus] in CMT?"

NARROW_MECHANISM_TERMS = ["sh3tc2", "hdac6", "aggresome", "autophagy", "cmt4c"]
GENERAL_CMT_QUESTION = "What is Charcot-Marie-Tooth disease?"


def run_one(retriever, generator, question: str, persona: Persona, n_ctx: int,
            answer_length="detailed", table_format="auto", verbose=True):
    candidate_pool = retriever.retrieve(question)  # RAW pool, pre-rerank/diversify
    ranked, scope, ents = retriever.retrieve_and_rank(question)
    assessment = assess_evidence_sufficiency(ranked, question=question)

    diag = {
        "retrieved_candidate_count": len(candidate_pool),
        "final_candidate_count": len(ranked),
        "scope": scope,
        "sufficient": assessment.sufficient,
        "supporting_chunk_source_ids": [c.source_id for c in assessment.supporting_chunks],
    }
    if verbose:
        print(f"  [diag] retrieved={diag['retrieved_candidate_count']} "
              f"final={diag['final_candidate_count']} scope={scope} "
              f"sufficient={assessment.sufficient} "
              f"supporting_sources={diag['supporting_chunk_source_ids']}")

    if not assessment.sufficient:
        return {
            "question": question, "persona": persona.value, "scope": scope,
            "insufficient_evidence": True, "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "cited_source_ids": [], "dropped_unsupported_citations": [], "diagnostics": diag,
        }

    final = assessment.supporting_chunks

    # Token-based context fit — NOT a character-based guess. This is the
    # authoritative answer to "does the resulting llama.cpp request fit
    # within n_ctx": measured with the model's real tokenizer, not
    # estimated. See generation.py's fit_candidates_to_token_budget
    # docstring for the full diagnosis of why the original
    # "Requested tokens (16023) exceed context window of 4096" crash
    # happened (CONTEXT_CHARS_PER_CHUNK was declared but never applied).
    try:
        messages, max_tokens, final, dropped = fit_candidates_to_token_budget(
            question, final, persona, answer_length, table_format,
            generator, n_ctx=n_ctx, safety_margin_tokens=CONTEXT_SAFETY_MARGIN_TOKENS,
        )
    except ContextBudgetError as e:
        diag["context_budget_error"] = str(e)
        if verbose:
            print(f"  [diag] CONTEXT BUDGET ERROR: {e}")
        return {
            "question": question, "persona": persona.value, "scope": scope,
            "insufficient_evidence": True, "answer": INSUFFICIENT_EVIDENCE_MESSAGE,
            "cited_source_ids": [], "dropped_unsupported_citations": [], "diagnostics": diag,
        }

    prompt_tokens = generator.count_tokens(messages[0]["content"] + "\n" + messages[1]["content"])
    diag.update({
        "prompt_tokens": prompt_tokens, "requested_output_tokens": max_tokens,
        "n_ctx": n_ctx, "fits_within_n_ctx": prompt_tokens + max_tokens + CONTEXT_SAFETY_MARGIN_TOKENS <= n_ctx,
        "candidates_dropped_for_context_fit": dropped,
    })
    if verbose:
        print(f"  [diag] prompt_tokens={prompt_tokens} requested_output_tokens={max_tokens} "
              f"n_ctx={n_ctx} fits={diag['fits_within_n_ctx']} dropped_for_fit={dropped}")

    raw_answer = generator.generate(messages, max_tokens)
    retrieved_ids = {c.source_id for c in final}
    validated = validate_citations(raw_answer, retrieved_ids)

    return {
        "question": question, "persona": persona.value, "scope": scope,
        "insufficient_evidence": False, "answer": validated.text,
        "cited_source_ids": validated.cited_source_ids,
        "dropped_unsupported_citations": validated.dropped_unsupported_citations,
        "diagnostics": diag,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-backend", choices=["sqlite", "postgres"], default="sqlite")
    ap.add_argument("--sqlite-path", default="./cmt_veda_rag.db")
    ap.add_argument("--postgres-dsn", default="")
    ap.add_argument("--vector-path", default="./vector_store/store")
    ap.add_argument("--vector-backend", choices=["numpy", "faiss"], default="faiss")
    ap.add_argument("--embedding-backend", choices=["hashing_tfidf", "sentence_transformers"], default="sentence_transformers")
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=4)
    ap.add_argument("--unsupported-question", default=UNSUPPORTED_QUESTION)
    ap.add_argument("--output", default="qwen_validation_report.json")
    args = ap.parse_args()

    db = get_backend(args.db_backend, sqlite_path=args.sqlite_path, postgres_dsn=args.postgres_dsn)
    embedder = get_embedding_backend(args.embedding_backend)
    vector_store = get_vector_store(args.vector_backend, args.vector_path, embedder.dim)
    retriever = HybridRetriever(db, embedder, vector_store)
    generator = get_generator("llama_cpp", model_path=args.model_path, n_ctx=args.n_ctx, n_threads=args.n_threads)

    report = {"results": [], "auto_check_failures": [], "needs_human_review": []}

    print("=== Mandatory questions x 3 personas ===")
    for q in MANDATORY_QUESTIONS:
        for persona in (Persona.STUDENT, Persona.CLINICIAN, Persona.RESEARCHER):
            r = run_one(retriever, generator, q, persona, n_ctx=args.n_ctx)
            report["results"].append(r)
            print(f"\n[{persona.value}] {q}")
            print(f"  {r['answer'][:300]}{'...' if len(r['answer']) > 300 else ''}")
            if r["dropped_unsupported_citations"]:
                msg = f"HALLUCINATED CITATION(S) dropped: {r['dropped_unsupported_citations']} for {q!r} ({persona.value})"
                print(f"  !! {msg}")
                report["auto_check_failures"].append(msg)

    print("\n=== Section-48 regression: general 'What is CMT?' must not be SH3TC2/HDAC6-heavy ===")
    general_result = next(r for r in report["results"]
                           if r["question"] == GENERAL_CMT_QUESTION and r["persona"] == "student")
    answer_lower = general_result["answer"].lower()
    hits = [t for t in NARROW_MECHANISM_TERMS if t in answer_lower]
    if len(hits) >= 2:
        msg = (f"General CMT answer contains {len(hits)} narrow mechanism terms {hits} — "
               f"READ THE FULL ANSWER MANUALLY to judge whether this is the section-48 bug "
               f"recurring or an appropriately-scoped mention.")
        print(f"  FLAGGED: {msg}")
        report["needs_human_review"].append(msg)
    else:
        print(f"  OK (found {len(hits)}/5 narrow mechanism terms: {hits}) — "
              f"still worth a human skim, this only checks keyword presence, not framing.")

    print(f"\n=== Deliberately unsupported question ===")
    for persona in (Persona.STUDENT, Persona.CLINICIAN, Persona.RESEARCHER):
        r = run_one(retriever, generator, args.unsupported_question, persona, n_ctx=args.n_ctx)
        report["results"].append(r)
        refused = r["insufficient_evidence"] or INSUFFICIENT_EVIDENCE_MESSAGE.lower() in r["answer"].lower()
        print(f"  [{persona.value}] refused correctly: {refused}")
        if not refused:
            msg = f"Unsupported question was NOT refused under persona={persona.value}: {r['answer'][:200]}"
            print(f"  !! {msg}")
            report["auto_check_failures"].append(msg)

    report["needs_human_review"].append(
        "Read every [student vs clinician vs researcher] triple for the same "
        "question above and confirm: (a) the underlying facts are consistent "
        "across personas, (b) terminology/depth genuinely differs by persona, "
        "(c) no forced What-is-it/Why/Treatment/Prognosis section appears "
        "where the evidence doesn't support it."
    )

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"Automated check failures: {len(report['auto_check_failures'])}")
    for f_ in report["auto_check_failures"]:
        print(f"  - {f_}")
    print(f"Items needing human review: {len(report['needs_human_review'])}")
    print(f"Full report written to {args.output}")

    if hasattr(db, "close"):
        db.close()

    sys.exit(1 if report["auto_check_failures"] else 0)


if __name__ == "__main__":
    main()
