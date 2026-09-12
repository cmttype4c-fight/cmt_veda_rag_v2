#!/usr/bin/env python3
"""
scripts/diagnose_retrieval.py
--------------------------------
Authoritative diagnostic for the retrieval -> evidence-assessment ->
prompt-construction path, against the REAL repository APIs (no
get_db()/get_embedder() — those never existed; see db.get_backend(),
embeddings.get_embedding_backend(), vector_store.get_vector_store()).

This is a DRY RUN: it loads the real model ONLY to get authoritative
token counts from its actual tokenizer (Llama.tokenize()), and NEVER
calls generate(). Nothing here burns inference time or produces an
answer — it exists purely to answer "what would the RAG actually send to
the model, and does it fit," without guessing.

Usage (matches your reported validation invocation):
    python3 scripts/diagnose_retrieval.py \\
        --db-backend postgres \\
        --postgres-dsn "$RAG_POSTGRES_DSN" \\
        --vector-path /opt/cmtveda/rag-v2/runtime/faiss_index \\
        --vector-backend faiss \\
        --embedding-backend sentence_transformers \\
        --model-path /opt/cmtveda/rag/runtime/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \\
        --n-ctx 4096 --n-threads 2

For each question, prints (per your numbered request):
  - retrieved candidate count (raw pool, pre-rerank/diversify)
  - final candidate count (post-rerank/diversify, pre-evidence-gate)
  - evidence sufficiency + reason
  - supporting chunk Source IDs (and per-chunk semantic/lexical/metadata/
    confidence scores — this is what actually tells you WHY sufficiency
    came out the way it did)
  - generated prompt: character count AND real token count (via the
    model's own tokenizer — not a char/4 guess)
  - requested output tokens (max_tokens for the persona/answer_length)
  - whether prompt_tokens + max_tokens + safety_margin fits n_ctx

Also runs three structural checks BEFORE touching any question, so a
misconfigured vector store or embedding mismatch is caught immediately
rather than producing 40 minutes of "insufficient evidence" confusion:
  1. vector_store.size() — if this is 0, your migration almost certainly
     wrote to a different --vector-path/--vector-backend than this run
     is pointed at (the single most likely explanation for retrieval
     returning nothing despite Postgres having 197 chunks).
  2. A canary embedding's L2 norm — catches a broken/degenerate
     embedding model silently producing near-zero vectors.
  3. db.get_active_chunk_ids() count vs. Postgres's own indexed-chunk
     count — catches an approval_status/knowledge_version mismatch
     between what you think is indexed and what the retrieval layer
     actually sees as active.
"""

import argparse
import sys

sys.path.insert(0, ".")

from db import get_backend
from embeddings import get_embedding_backend
from vector_store import get_vector_store
from retrieval import HybridRetriever, assess_evidence_sufficiency
from generation import build_messages, fit_candidates_to_token_budget, ContextBudgetError, get_generator
from config import Persona, CONTEXT_SAFETY_MARGIN_TOKENS

DEFAULT_QUESTIONS = [
    "What is Charcot-Marie-Tooth disease?",
    "What causes CMT?",
    "What are common symptoms of CMT?",
]


def preflight_checks(db, embedder, vector_store):
    print("=== Preflight checks ===")

    size = vector_store.size()
    print(f"1. vector_store.size() = {size}")
    if size == 0:
        print("   !! ZERO vectors in the loaded vector store. This is almost certainly")
        print("      why every question returns 'insufficient evidence' — the semantic")
        print("      side of retrieval has nothing to search. Most likely cause: the")
        print("      migration run used a DIFFERENT --vector-path and/or --vector-backend")
        print("      than this diagnostic (and your validation run) are pointed at.")
        print("      scripts/migrate_corpus.py defaults to --vector-backend numpy and")
        print("      --embedding-backend hashing_tfidf if you didn't override them —")
        print("      re-check the EXACT flags used for the real migration run.")
    else:
        print("   OK — vector store is non-empty.")

    canary_vec = embedder.embed(["Charcot-Marie-Tooth disease peripheral neuropathy"])[0]
    import numpy as np
    norm = float(np.linalg.norm(canary_vec))
    print(f"2. Canary embedding L2 norm = {norm:.4f}")
    if norm < 1e-6:
        print("   !! Near-zero norm — the embedding model is producing degenerate output.")
        print("      Check that sentence-transformers actually loaded its weights (a failed")
        print("      first-time download that silently fell back to random init would look")
        print("      like this).")
    else:
        print("   OK — embedding model produces non-degenerate vectors.")

    active_ids = db.get_active_chunk_ids()
    print(f"3. db.get_active_chunk_ids() count = {len(active_ids)}")
    if len(active_ids) == 0:
        print("   !! ZERO chunks are marked retrievable (approval_status='indexed').")
        print("      Check the documents' actual approval_status in Postgres directly:")
        print("      SELECT approval_status, count(*) FROM cmt_veda_rag.documents GROUP BY 1;")
    else:
        print("   OK — chunks are marked active/indexed.")
    print()


def diagnose_question(retriever, generator, question: str, n_ctx: int, persona=Persona.STUDENT,
                       answer_length="detailed", table_format="auto"):
    print(f"--- {question!r} ---")

    raw_pool = retriever.retrieve(question)
    print(f"  retrieved candidate count (raw pool): {len(raw_pool)}")

    final, scope, ents = retriever.retrieve_and_rank(question)
    print(f"  final candidate count (post-rerank/diversify): {len(final)}")
    print(f"  query scope: {scope}  entities: genes={ents.genes} subtypes={ents.subtypes}")

    for c in final[:5]:
        print(f"    chunk source_id={c.source_id} semantic={c.semantic_score:.3f} "
              f"lexical={c.lexical_score:.3f} metadata={c.metadata_score:.3f} "
              f"confidence={c.confidence:.3f} rank_score={c.rank_score:.3f}")
    if len(final) > 5:
        print(f"    ... and {len(final) - 5} more")

    assessment = assess_evidence_sufficiency(final, question=question)
    print(f"  evidence sufficient: {assessment.sufficient}  reason: {assessment.reason}")
    print(f"  supporting chunk source_ids: {[c.source_id for c in assessment.supporting_chunks]}")

    if not assessment.sufficient:
        print("  (skipping prompt construction — no evidence to build a prompt from)\n")
        return

    try:
        messages, max_tokens, kept, dropped = fit_candidates_to_token_budget(
            question, assessment.supporting_chunks, persona, answer_length, table_format,
            generator, n_ctx=n_ctx, safety_margin_tokens=CONTEXT_SAFETY_MARGIN_TOKENS,
        )
    except ContextBudgetError as e:
        print(f"  !! CONTEXT BUDGET ERROR (even zero evidence doesn't fit): {e}\n")
        return

    char_count = len(messages[0]["content"]) + len(messages[1]["content"])
    prompt_tokens = generator.count_tokens(messages[0]["content"] + "\n" + messages[1]["content"])
    fits = prompt_tokens + max_tokens + CONTEXT_SAFETY_MARGIN_TOKENS <= n_ctx

    print(f"  prompt character count: {char_count}")
    print(f"  prompt TOKEN count (real tokenizer): {prompt_tokens}")
    print(f"  requested output tokens (max_tokens): {max_tokens}")
    print(f"  candidates dropped to fit context budget: {dropped}")
    print(f"  fits within n_ctx={n_ctx} (with {CONTEXT_SAFETY_MARGIN_TOKENS}-token margin): {fits}")
    if char_count > 0:
        print(f"  (character-to-token ratio for this prompt: {char_count / max(prompt_tokens, 1):.2f} "
              f"chars/token — compare to the ~4 chars/token assumed by any character-based budget)")
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db-backend", choices=["sqlite", "postgres"], required=True)
    ap.add_argument("--sqlite-path", default="./cmt_veda_rag.db")
    ap.add_argument("--postgres-dsn", default="")
    ap.add_argument("--vector-path", required=True)
    ap.add_argument("--vector-backend", choices=["numpy", "faiss"], required=True)
    ap.add_argument("--embedding-backend", choices=["hashing_tfidf", "sentence_transformers"], required=True)
    ap.add_argument("--model-path", required=True,
                     help="Needed even for this dry-run diagnostic, to get REAL token counts "
                          "from the model's own tokenizer via Llama.tokenize(). generate() is "
                          "never called.")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--n-threads", type=int, default=2)
    ap.add_argument("--questions", nargs="*", default=None)
    args = ap.parse_args()

    questions = args.questions or DEFAULT_QUESTIONS

    print(f"Connecting: db_backend={args.db_backend}  vector_backend={args.vector_backend} "
          f"(path={args.vector_path})  embedding_backend={args.embedding_backend}\n")

    db = get_backend(args.db_backend, sqlite_path=args.sqlite_path, postgres_dsn=args.postgres_dsn)
    embedder = get_embedding_backend(args.embedding_backend)
    vector_store = get_vector_store(args.vector_backend, args.vector_path, embedder.dim)
    retriever = HybridRetriever(db, embedder, vector_store)

    preflight_checks(db, embedder, vector_store)

    print("Loading model for tokenization only (generate() will not be called)...")
    generator = get_generator("llama_cpp", model_path=args.model_path, n_ctx=args.n_ctx, n_threads=args.n_threads)
    print("Model loaded.\n")

    for q in questions:
        diagnose_question(retriever, generator, q, n_ctx=args.n_ctx)

    if hasattr(db, "close"):
        db.close()


if __name__ == "__main__":
    main()
