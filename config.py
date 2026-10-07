"""
config.py
---------
Central configuration for CMT Veda AI RAG v2.

Nothing in here talks to the network or a database — it is pure constants
and small enums so every other module (retrieval, generation, ingestion,
main) imports the same source of truth.
"""

import os
from enum import Enum


# ---------------------------------------------------------------------
# EXISTING CONFIG (carried over from v1, unchanged where practical)
# ---------------------------------------------------------------------
INDEX_FOLDER = os.environ.get("RAG_INDEX_FOLDER", "./faiss_index")
DOCS_FOLDER = os.environ.get("RAG_DOCS_FOLDER", "./docs")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

GGUF_MODEL_PATH = os.environ.get(
    "RAG_GGUF_MODEL_PATH", "./models/qwen2.5-1.5b-instruct-q4_k_m.gguf"
)
N_CTX = int(os.environ.get("RAG_N_CTX", "4096"))
# Reserved headroom below N_CTX for: the model's own chat-template
# overhead (role markers/special tokens llama.cpp adds beyond the raw
# message text), and to avoid landing EXACTLY at the boundary. This is
# on top of max_tokens (the requested completion length) — the real
# check is prompt_tokens + max_tokens + this margin <= N_CTX.
CONTEXT_SAFETY_MARGIN_TOKENS = int(os.environ.get("RAG_CONTEXT_SAFETY_MARGIN_TOKENS", "128"))
N_THREADS = int(os.environ.get("RAG_N_THREADS", str(os.cpu_count() or 4)))

RAG_API_KEY = os.environ.get("RAG_API_KEY", "").strip()
# Separate, higher-privilege key for admin/ingestion operations (section 43).
# Never accept this from a browser; only from the Veda admin backend / CLI.
RAG_ADMIN_API_KEY = os.environ.get("RAG_ADMIN_API_KEY", "").strip()

RAG_MAX_QUESTION_CHARS = int(os.environ.get("RAG_MAX_QUESTION_CHARS", "4000"))

# Current knowledge version. Bump this whenever the index is rebuilt so every
# answer can be tied back to the corpus state that produced it (section 32).
KNOWLEDGE_VERSION = os.environ.get("RAG_KNOWLEDGE_VERSION", "v2-unset")

# ---------------------------------------------------------------------
# RETRIEVAL TUNING (sections 6-9)
# ---------------------------------------------------------------------
CANDIDATE_POOL_MIN = int(os.environ.get("RAG_CANDIDATE_POOL_MIN", "20"))
CANDIDATE_POOL_MAX = int(os.environ.get("RAG_CANDIDATE_POOL_MAX", "40"))
FINAL_CONTEXT_CHUNKS_MIN = int(os.environ.get("RAG_FINAL_CHUNKS_MIN", "6"))
FINAL_CONTEXT_CHUNKS_MAX = int(os.environ.get("RAG_FINAL_CHUNKS_MAX", "12"))

# Max chunks allowed from a single source document when the question is
# classified as "general" (section 9 - source diversification). Specific
# questions are not capped.
MAX_CHUNKS_PER_DOC_GENERAL = int(os.environ.get("RAG_MAX_CHUNKS_PER_DOC_GENERAL", "2"))

CONTEXT_CHARS_PER_CHUNK = int(os.environ.get("RAG_CONTEXT_CHARS_PER_CHUNK", "1200"))

# Hybrid retrieval weights (semantic vs lexical vs metadata boosts). These are
# starting points; section 8 explicitly says "tune after benchmarking".
WEIGHT_SEMANTIC = float(os.environ.get("RAG_WEIGHT_SEMANTIC", "0.55"))
WEIGHT_LEXICAL = float(os.environ.get("RAG_WEIGHT_LEXICAL", "0.30"))
WEIGHT_METADATA = float(os.environ.get("RAG_WEIGHT_METADATA", "0.15"))

# Lexical score saturation divisor (used to squash a backend's raw
# lexical score into 0..1 before blending into `confidence` — see
# retrieval.py's rerank()/assess_evidence_sufficiency()). THIS IS
# BACKEND-DEPENDENT and was a real bug: it was hardcoded to 5.0,
# calibrated against SQLite FTS5's bm25()-derived score (typically
# single digits for a strong match). Postgres's ts_rank_cd() lives on a
# completely different scale (typically well under 1.0 even for a
# strong match), so the same /5.0 divisor made the lexical signal
# contribute almost nothing to `confidence` on Postgres specifically —
# which, combined with a near-zero semantic contribution whenever the
# vector store is empty/mismatched, was enough to push otherwise-good
# evidence below EVIDENCE_SCORE_FLOOR. The Postgres default below is a
# reasoned STARTING POINT (ts_rank_cd for a solid plainto_tsquery match
# is commonly in the ~0.1-0.6 range), not empirically calibrated against
# a live Postgres instance — benchmark against your real corpus and tune
# via RAG_LEXICAL_SATURATION if scores still look off.
LEXICAL_SATURATION = float(os.environ.get(
    "RAG_LEXICAL_SATURATION",
    "5.0" if os.environ.get("RAG_DB_BACKEND", "sqlite") == "sqlite" else "0.5",
))

# Evidence sufficiency gate (section 13). If the best reranked score for the
# top chunk is below this, or too few candidates clear a floor score, the
# system refuses rather than guesses.
EVIDENCE_SCORE_FLOOR = float(os.environ.get("RAG_EVIDENCE_SCORE_FLOOR", "0.28"))
EVIDENCE_MIN_SUPPORTING_CHUNKS = int(os.environ.get("RAG_EVIDENCE_MIN_CHUNKS", "1"))

INSUFFICIENT_EVIDENCE_MESSAGE = (
    "The current CMT Veda knowledge base does not contain sufficient evidence "
    "to reliably answer this question."
)

# ---------------------------------------------------------------------
# ROLE -> PERSONA MAPPING (section 2) — server-authoritative, never
# trusted from the browser/client.
# ---------------------------------------------------------------------
class Persona(str, Enum):
    STUDENT = "student"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"


APP_ROLE_TO_PERSONA = {
    "patient": Persona.STUDENT,
    "caregiver": Persona.STUDENT,
    "student": Persona.STUDENT,
    "other": Persona.STUDENT,
    "clinician": Persona.CLINICIAN,
    "researcher": Persona.RESEARCHER,
}


def resolve_persona(app_role: str) -> Persona:
    """Map a CMT Veda application role to a RAG persona.

    This is the ONLY place persona is decided. It must be called with a role
    the Veda API server has already authenticated (never a raw client field
    trusted at face value) — see main.py's require_api_key / role handling.
    Unknown roles fail closed to the most conservative persona (student)
    rather than raising, so a misconfigured caller never accidentally gets
    clinician/researcher-level source detail.
    """
    return APP_ROLE_TO_PERSONA.get((app_role or "").strip().lower(), Persona.STUDENT)


# ---------------------------------------------------------------------
# ANSWER DEPTH (section 4)
# ---------------------------------------------------------------------
LENGTH_PRESETS = {
    "concise": {
        "max_tokens": int(os.environ.get("RAG_MAX_TOKENS_CONCISE", "180")),
        "instruction": "Answer in 2-4 concise sentences. No filler.",
    },
    "detailed": {
    "max_tokens": int(os.environ.get("RAG_MAX_TOKENS_DETAILED", "450")),
        "instruction": (
            "Give a genuinely explanatory answer (roughly 150-300 words). Do not "
            "pad length with restated points, and do not invent structure "
            "(e.g. 'What is it? / Why? / Treatment') unless the retrieved "
            "evidence actually supports each of those topics."
        ),
    },
    "deep": {
    "max_tokens": int(os.environ.get("RAG_MAX_TOKENS_DEEP", "750")),
        "instruction": (
            "Give a thorough, well-structured answer (multiple short paragraphs "
            "or a few labeled sections only where the evidence genuinely "
            "supports them). Cover mechanism, clinical features, study "
            "characteristics, and any relevant nuance/limitations found in the "
            "context."
        ),
    },
}
# Backward-compat aliases for the old short/medium/long values (section 33 -
# "do not break this contract unnecessarily").
LENGTH_PRESETS["short"] = LENGTH_PRESETS["concise"]
LENGTH_PRESETS["medium"] = LENGTH_PRESETS["detailed"]
LENGTH_PRESETS["long"] = LENGTH_PRESETS["deep"]
DEFAULT_ANSWER_LENGTH = "detailed"

TABLE_INSTRUCTIONS = {
    "auto": "If the information is naturally comparative or list-like (e.g. subtypes, "
            "genes, symptoms by severity), present it as a compact Markdown table. "
            "Otherwise write normal prose.",
    "on": "Wherever possible, structure the answer as a Markdown table (with a short "
          "intro/outro sentence), even for information that could also be written as prose.",
    "off": "Do not use any tables. Answer only in prose/paragraph form.",
}

# ---------------------------------------------------------------------
# DISCOVERY / INGESTION LIFECYCLE (section 21 of the original spec;
# corrected per the "PLATFORM 1 — RAG v2 COMPLETION" follow-up, section C)
# ---------------------------------------------------------------------
class IngestionState(str, Enum):
    DISCOVERED = "discovered"
    PENDING_APPROVAL = "pending_approval"
    APPROVED = "approved"
    QUEUED = "queued"
    PROCESSING = "processing"
    INDEXED = "indexed"
    FAILED = "failed"
    REMOVED = "removed"


# CRITICAL DISTINCTION (this was wrong in the previous version and is the
# specific bug this follow-up asked to fix): APPROVED means the document is
# *authorized to enter the ingestion pipeline*. It does NOT mean the
# document's chunks are retrievable. Only INDEXED is retrievable. Every
# retrieval query MUST filter on this, at the database level, not just in
# application code that might be bypassed.
RETRIEVABLE_STATE = IngestionState.INDEXED


def is_retrievable(state: "IngestionState") -> bool:
    return state == RETRIEVABLE_STATE


# Single source of truth for legal lifecycle transitions (used by
# ingestion.py's in-memory IngestionRecord and by db.py's persisted
# version, so there is exactly one place this rule lives).
_ALLOWED_TRANSITIONS = {
    IngestionState.DISCOVERED: {IngestionState.PENDING_APPROVAL},
    IngestionState.PENDING_APPROVAL: {IngestionState.APPROVED, IngestionState.REMOVED},
    IngestionState.APPROVED: {IngestionState.QUEUED, IngestionState.REMOVED},
    IngestionState.QUEUED: {IngestionState.PROCESSING, IngestionState.FAILED},
    IngestionState.PROCESSING: {IngestionState.INDEXED, IngestionState.FAILED},
    IngestionState.INDEXED: {IngestionState.REMOVED},
    IngestionState.FAILED: {IngestionState.QUEUED, IngestionState.REMOVED},
    IngestionState.REMOVED: set(),  # terminal; record is archived, not deleted
}


class IllegalStateTransitionError(ValueError):
    """Raised by validate_transition() for a lifecycle transition that
    isn't in _ALLOWED_TRANSITIONS (e.g. approving an already-INDEXED
    document, or retrying one that isn't FAILED). A ValueError subclass
    on purpose -- every pre-existing `except ValueError` call site (the
    legacy /admin/ingest/remove handler, discovery.py, etc.) keeps
    catching it unchanged. Added in round 4 so main.py's new
    /admin/intake/* routes can map THIS specific case to 409 Conflict
    instead of the generic ValueError->404 mapping, which was
    mislabeling "document exists but is in the wrong state" as "no such
    document" -- a real contract bug caught while writing the final API
    contract doc, fixed here rather than documented as correct."""


def validate_transition(current: "IngestionState", new_state: "IngestionState") -> None:
    """Raises IllegalStateTransitionError (a ValueError) on an illegal
    lifecycle transition."""
    allowed = _ALLOWED_TRANSITIONS.get(current, set())
    if new_state not in allowed:
        raise IllegalStateTransitionError(f"Illegal ingestion state transition {current} -> {new_state}")

SOURCE_ID_PREFIX = "CMT-RAG-"
SOURCE_ID_DIGITS = 6

# ---------------------------------------------------------------------
# PERSISTENCE BACKEND (section A of the completion follow-up)
# ---------------------------------------------------------------------
# "postgres" is the real production backend (db.py: PostgresBackend, needs
# psycopg2 + a running Postgres server — NOT available in this sandbox).
# "sqlite" is a local/offline substitute (db.py: SqliteBackend, stdlib
# only) used for development, CI, and to actually execute the tests in
# this repo end-to-end without a database server. It is NOT proposed as a
# second production database — see DELIVERABLES.md.
RAG_DB_BACKEND = os.environ.get("RAG_DB_BACKEND", "sqlite")
RAG_SQLITE_PATH = os.environ.get("RAG_SQLITE_PATH", "./cmt_veda_rag.db")
RAG_POSTGRES_DSN = os.environ.get("RAG_POSTGRES_DSN", "")

# Embedding/vector backend, same pattern: "sentence_transformers"+"faiss"
# are the real production pair (needs network to fetch model weights the
# first time, plus the faiss package); "hashing_tfidf"+"numpy" is the
# dependency-free local substitute used for testing in this sandbox.
RAG_EMBEDDING_BACKEND = os.environ.get("RAG_EMBEDDING_BACKEND", "hashing_tfidf")
RAG_VECTOR_BACKEND = os.environ.get("RAG_VECTOR_BACKEND", "numpy")
RAG_VECTOR_STORE_PATH = os.environ.get("RAG_VECTOR_STORE_PATH", "./vector_store")

# Generation backend: "llama_cpp" is real production (needs the GGUF model
# file, not available here); "mock" is a deterministic, non-LLM stand-in
# used ONLY to test pipeline wiring/citation validation in this sandbox —
# it does not demonstrate that a real model follows the grounding rules.
RAG_GENERATION_BACKEND = os.environ.get("RAG_GENERATION_BACKEND", "mock")

# ---------------------------------------------------------------------
# INTAKE LIFECYCLE (round 4: register/approve/queue/process split,
# bulk operations, async worker — see DELIVERABLES_ROUND4.md)
# ---------------------------------------------------------------------

# The only two source types the intake API will accept. Enforced here AND
# at the database level (the source_type enum in the round-4 migration) —
# defense in depth, same pattern as RETRIEVABLE_STATE above. This is also
# the enforcement point for "Newsletter independence" (brief §16/§8): the
# Newsletter pipeline has no caller anywhere in this codebase that reaches
# register_intake(), and even if something did call it with a value like
# 'newsletter' or 'editorial', this allowlist rejects it before a document
# row is ever created.
ALLOWED_SOURCE_TYPES = {"discovery", "direct_upload"}

# Full-text-only rule (brief §4). This is a minimum-length HEURISTIC, not
# a real abstract-vs-full-text classifier — honestly flagged as a gap, not
# hidden. A short but genuine full-text letter/note could theoretically be
# rejected; an unusually long abstract could theoretically pass. Real
# enforcement of "genuine full text" ultimately depends on what Discovery
# and the admin UI assert about the material before it ever reaches this
# API — this floor just catches the obvious case (an abstract, or nothing
# at all) at the backend level per the brief's explicit instruction not to
# rely on the UI alone.
MIN_FULL_TEXT_CHARS = int(os.environ.get("RAG_MIN_FULL_TEXT_CHARS", "1000"))

# Retry policy (brief §12, your confirmed decision: 5 attempts, exponential
# backoff). PostgreSQL (documents.retry_count, processing_jobs) is the
# durable source of truth for all of this — these constants only decide
# the *policy*, never the bookkeeping, which always goes through db.py.
MAX_RETRIES = int(os.environ.get("RAG_MAX_RETRIES", "5"))
RETRY_BACKOFF_BASE_SECONDS = int(os.environ.get("RAG_RETRY_BACKOFF_BASE_SECONDS", "30"))


def retry_backoff_seconds(attempt: int) -> int:
    """Exponential backoff: 30s, 60s, 120s, 240s, 480s for attempts 1-5.
    `attempt` is 1-indexed (the attempt that just failed)."""
    return RETRY_BACKOFF_BASE_SECONDS * (2 ** max(attempt - 1, 0))


# Redis/RQ — the asynchronous job-delivery layer ONLY (brief §12/§5). Never
# consulted on the Ask Veda retrieval path (retrieval.py queries the
# already-indexed corpus directly; nothing there touches Redis or RQ).
# NOT available in this sandbox (no `redis` or `rq` package, no network to
# install them — same constraint DELIVERABLES.md documents for psycopg2).
# worker.py's InlineQueueAdapter is the tested, synchronous substitute;
# RQ/Redis wiring in worker.py is written but unexecuted here.
REDIS_URL = os.environ.get("RAG_REDIS_URL", "redis://localhost:6379/0")
RQ_QUEUE_NAME = os.environ.get("RAG_RQ_QUEUE_NAME", "cmt_veda_intake")
RQ_JOB_TIMEOUT_SECONDS = int(os.environ.get("RAG_RQ_JOB_TIMEOUT_SECONDS", "600"))

# Which enqueue_fn main.py's /admin/intake/* routes wire up at startup.
# "redis_rq" (default — the real production path, worker.enqueue_processing)
# requires the redis/rq packages and a reachable Redis; set to "inline" to
# run the API with worker.InlineQueueAdapter instead (same adapter
# test_intake_lifecycle.py uses) on a from-scratch dev/CI box with no
# Redis — never silently substituted; this flag is the only switch.
RAG_QUEUE_BACKEND = os.environ.get("RAG_QUEUE_BACKEND", "redis_rq")

