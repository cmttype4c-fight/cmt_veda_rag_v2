"""
main.py  (RAG v2 — completion follow-up)
------------------------------------------
Production RAG service for CMT Veda AI, wired to the REAL persistence,
embedding, vector-store, and ingestion pieces built in this follow-up
(db.py, embeddings.py, vector_store.py, ingestion_pipeline.py,
discovery.py) instead of the in-memory placeholders from the previous
version.

Backend selection is via config.py's RAG_DB_BACKEND / RAG_EMBEDDING_BACKEND
/ RAG_VECTOR_BACKEND / RAG_GENERATION_BACKEND env vars. Defaults are the
sandbox-executable substitutes (sqlite / hashing_tfidf / numpy / mock) —
**you must set these to postgres / sentence_transformers / faiss /
llama_cpp for production**, none of which were available to test in this
environment. See DELIVERABLES.md for exactly what was and wasn't run.

Unchanged from the previous version (kept per "do not rebuild the parts
that already work"):
  * NO /files/{filename} route exists anywhere in this file.
  * Persona resolution is server-authoritative (config.resolve_persona).
  * Sources are persona-filtered (StudentSource vs RichSource).
  * Citation validation strips unretrieved Source IDs from the answer.
"""

import logging
import os
import time
import threading
import uuid
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Literal, Optional
import secrets

from fastapi import FastAPI, HTTPException, Header, APIRouter, UploadFile, File, Form
from pydantic import BaseModel

import config
from config import Persona, resolve_persona, IngestionState, INSUFFICIENT_EVIDENCE_MESSAGE
from db import get_backend, DuplicateDocumentError, TurnRecord
from embeddings import get_embedding_backend
from vector_store import get_vector_store
from retrieval import HybridRetriever, Candidate, assess_evidence_sufficiency
from generation import build_messages, validate_citations, get_generator, fit_candidates_to_token_budget, ContextBudgetError
from ingestion_pipeline import IngestionInput, ingest_document, remove_document, IngestionValidationError
from url_safety import sanitize_reference_url
from conversation import FallbackQueryRewriter, LLMQueryRewriter, is_authorized_for_conversation
from discovery import (
    register_candidate, approve_and_ingest, reject_candidate,
    DiscoveryCandidatePayload, DiscoveryRejectionError,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cmt-veda-ai")

state = {}
_gen_lock = threading.Lock()  # llama.cpp generation is not safely re-entrant


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not config.RAG_API_KEY:
        raise RuntimeError(
            "RAG_API_KEY is not configured. Refusing to start the RAG API without "
            "server-to-server authentication."
        )

    logger.info("Connecting to DB backend: %s", config.RAG_DB_BACKEND)
    db = get_backend(config.RAG_DB_BACKEND, sqlite_path=config.RAG_SQLITE_PATH,
                      postgres_dsn=config.RAG_POSTGRES_DSN)
    state["db"] = db

    logger.info("Loading embedding backend: %s", config.RAG_EMBEDDING_BACKEND)
    embedder = get_embedding_backend(config.RAG_EMBEDDING_BACKEND)
    state["embedder"] = embedder

    logger.info("Loading vector store backend: %s", config.RAG_VECTOR_BACKEND)
    vector_store = get_vector_store(config.RAG_VECTOR_BACKEND, config.RAG_VECTOR_STORE_PATH, embedder.dim)
    state["vector_store"] = vector_store

    state["retriever"] = HybridRetriever(db, embedder, vector_store)

    logger.info("Loading generation backend: %s", config.RAG_GENERATION_BACKEND)
    if config.RAG_GENERATION_BACKEND == "llama_cpp":
        # Diagnostic-only log for performance measurement.
        logger.info(
            "llama_cpp config: n_ctx=%s n_threads=%s (os.cpu_count()=%s) model_path=%s",
            config.N_CTX, config.N_THREADS, os.cpu_count(), config.GGUF_MODEL_PATH,
        )

        state["generator"] = get_generator(
            "llama_cpp", model_path=config.GGUF_MODEL_PATH,
            n_ctx=config.N_CTX, n_threads=config.N_THREADS,
        )
    else:
        state["generator"] = get_generator(config.RAG_GENERATION_BACKEND)

    # Query rewriter for multi-turn follow-ups (conversation.py). Reuses
    # the SAME generator instance for its LLM-fallback half rather than
    # loading a second model — a rewrite call and an answer-generation
    # call both go through _gen_lock so they never overlap.
    state["rewriter"] = FallbackQueryRewriter(llm_rewriter=LLMQueryRewriter(state["generator"]))

    all_docs = db.list_documents()
    indexed = [d for d in all_docs if d.approval_status == IngestionState.INDEXED]
    logger.info(
        "Startup complete. %s document(s) INDEXED (of %s total). knowledge_version=%s. "
        "answer_length max_tokens: concise=%s detailed=%s deep=%s. Ready to serve.",
        len(indexed), len(all_docs), config.KNOWLEDGE_VERSION,
        config.LENGTH_PRESETS["concise"]["max_tokens"],
        config.LENGTH_PRESETS["detailed"]["max_tokens"],
        config.LENGTH_PRESETS["deep"]["max_tokens"],
    )
    yield
    if hasattr(db, "close"):
        db.close()
    state.clear()


app = FastAPI(
    title="CMT Veda AI RAG v2",
    lifespan=lifespan,
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
)


# ---------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------
def require_api_key(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
) -> None:
    if not config.RAG_API_KEY:
        raise HTTPException(status_code=503, detail="Knowledge service authentication is not configured.")
    supplied = x_api_key
    if not supplied and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer" and token:
            supplied = token.strip()
    if not supplied or not secrets.compare_digest(supplied, config.RAG_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized.")


def require_admin_key(
    x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key"),
) -> None:
    if not config.RAG_ADMIN_API_KEY:
        raise HTTPException(status_code=503, detail="Admin operations are not configured.")
    if not x_admin_api_key or not secrets.compare_digest(x_admin_api_key, config.RAG_ADMIN_API_KEY):
        raise HTTPException(status_code=401, detail="Unauthorized.")


# ---------------------------------------------------------------------
# REQUEST / RESPONSE MODELS
# ---------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str
    user_type: Literal["patient", "caregiver", "student", "other", "clinician", "researcher"] = "patient"
    app_role: Optional[str] = None
    answer_length: Literal["concise", "detailed", "deep", "short", "medium", "long"] = "detailed"
    table_format: Literal["auto", "on", "off"] = "auto"
    # Multi-turn support, fully backward compatible: omitting this field
    # (as every existing Veda integration does today) keeps the endpoint
    # completely stateless, exactly as before — no conversation row is
    # created, no history is loaded, no rewriting happens. A client opts
    # into multi-turn by generating its own UUID for a new conversation
    # and passing the SAME id on every subsequent turn; the server never
    # invents conversation ids on the client's behalf, so there's no
    # ambiguity about "was this a new conversation" to report back.
    conversation_id: Optional[str] = None
    # TRUSTED identifier from Veda's authenticated backend — same trust
    # boundary as app_role (see config.resolve_persona and db.py's
    # create_conversation docstring). This service never authenticates
    # end users itself; Veda must never forward a client-editable value
    # here. Required to make a conversation resumable/listable later —
    # a conversation_id with no user_id attached can still be used
    # within a single session but won't show up in anyone's "previous
    # conversations" list.
    user_id: Optional[str] = None


def _derive_conversation_title(question: str, max_len: int = 60) -> str:
    q = " ".join(question.strip().split())
    return q if len(q) <= max_len else q[: max_len - 1].rstrip() + "…"


class StudentSource(BaseModel):
    source_id: str


class RichSource(BaseModel):
    source_id: str
    title: Optional[str] = None
    authors: Optional[list] = None
    publication_date: Optional[str] = None
    journal: Optional[str] = None
    doi: Optional[str] = None
    pmid: Optional[str] = None
    study_type: Optional[str] = None
    public_reference_url: Optional[str] = None


class AskResponse(BaseModel):
    answer: str
    sources: list
    knowledge_version: str
    insufficient_evidence: bool = False
    conversation_id: Optional[str] = None
    rewritten_question: Optional[str] = None  # only set when a rewrite actually happened


class StatsResponse(BaseModel):
    document_count: int
    indexed_document_count: int
    knowledge_version: str


def _persona_filtered_source(candidate: Candidate, persona: Persona):
    meta = candidate.metadata or {}
    if persona == Persona.STUDENT:
        return StudentSource(source_id=candidate.source_id)
    # Defense in depth (item 5, final hardening brief): sanitize AGAIN at
    # response time, even though ingestion_pipeline.py already sanitizes
    # at write time. A URL that got into the DB some other way (manual
    # edit, a future writer that forgets to sanitize) still can't reach a
    # response — this is the second of the two enforcement points
    # described in url_safety.py's module docstring.
    safe_url = sanitize_reference_url(meta.get("source_url") or "")
    return RichSource(
        source_id=candidate.source_id, title=meta.get("title"), authors=meta.get("authors"),
        publication_date=meta.get("publication_date"), journal=meta.get("journal"),
        doi=meta.get("doi"), pmid=meta.get("pmid"), study_type=meta.get("study_type"),
        public_reference_url=safe_url or None,
    )


@app.post("/api/ask", response_model=AskResponse)
def ask(
    payload: AskRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    # Per-stage timing instrumentation.
    # Server-side logs only; does not change the API response.
    t_start = time.perf_counter()
    timings = {}

    def _mark(label, t_from):
        timings[label] = time.perf_counter() - t_from
        return time.perf_counter()
     
    require_api_key(x_api_key, authorization)

    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    if len(question) > config.RAG_MAX_QUESTION_CHARS:
        raise HTTPException(status_code=413, detail=f"Question exceeds the maximum allowed length of {config.RAG_MAX_QUESTION_CHARS} characters.")

    role_for_mapping = payload.app_role or payload.user_type
    persona = resolve_persona(role_for_mapping)

    db = state["db"]
    conversation_id = payload.conversation_id
    history = []
    t = time.perf_counter()
    if conversation_id:
        db.create_conversation(conversation_id, persona.value, user_id=payload.user_id,
                                title=_derive_conversation_title(question))
        history = db.get_turns(conversation_id, limit=4)
     
        t = _mark("history_load", t)

    # Query rewriting happens BEFORE retrieval and touches ONLY the
    # question that gets retrieved — never the generation prompt. See
    # conversation.py's module docstring for why that boundary matters.
    effective_question = question
    if history:
        with _gen_lock:  # the LLM-fallback half of the rewriter shares the generator
            rewrite_result = state["rewriter"].rewrite(question, history)
        effective_question = rewrite_result.question
        if rewrite_result.was_rewritten:
            logger.info("Rewrote follow-up %r -> %r (method=%s) for conversation_id=%s",
                        question, effective_question, rewrite_result.method, conversation_id)

    t = _mark("rewrite", t)
    retriever: HybridRetriever = state["retriever"]
    ranked, scope, ents = retriever.retrieve_and_rank(effective_question)
    t = _mark("retrieve", t)
    assessment = assess_evidence_sufficiency(ranked, question=effective_question)
    t = _mark("evidence_gate", t)

    def _persist_turn(answer_text: str, cited_ids: list, insufficient: bool):
        if not conversation_id:
            return
        db.add_turn(TurnRecord(
            turn_id=str(uuid.uuid4()), conversation_id=conversation_id,
            turn_order=len(history), question=question,
            rewritten_question=effective_question, scope=scope,
            answer=answer_text, cited_source_ids=cited_ids,
            insufficient_evidence=insufficient,
        ))
        
    def _log_timing_summary(insufficient: bool, generated_tokens=None, requested_max_tokens=None,
                             prompt_tokens=None, candidates_used=None):
        timings["total"] = time.perf_counter() - t_start
        logger.info(
            "TIMING question=%r insufficient=%s scope=%s stages=%s "
            "prompt_tokens=%s requested_max_tokens=%s generated_tokens=%s candidates_used=%s",
            question[:80], insufficient, scope,
            {k: round(v, 3) for k, v in timings.items()},
            prompt_tokens, requested_max_tokens, generated_tokens, candidates_used,
        )

    if not assessment.sufficient:
        logger.info("Insufficient evidence (scope=%s): %s", scope, assessment.reason)

        db.record_answer_audit(
            persona.value,
            scope,
            config.KNOWLEDGE_VERSION,
            [],
            [],
            True,
        )
        t = _mark("audit_write", t)

        _persist_turn(INSUFFICIENT_EVIDENCE_MESSAGE, [], True)
        t = _mark("persist_turn", t)

        _log_timing_summary(insufficient=True)

        return AskResponse(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE,
            sources=[],
            knowledge_version=config.KNOWLEDGE_VERSION,
            insufficient_evidence=True,
            conversation_id=conversation_id,
            rewritten_question=effective_question if effective_question != question else None,
        )

    final_candidates = assessment.supporting_chunks[: config.FINAL_CONTEXT_CHUNKS_MAX]
    generator = state["generator"]

    # Token-based context fit (fixes a real deployment crash:
    # "Requested tokens (16023) exceed context window of 4096" — see
    # generation.py's fit_candidates_to_token_budget docstring for the
    # full diagnosis). Drops least-relevant candidates until the ACTUAL
    # tokenized prompt fits n_ctx, rather than guessing from character
    # counts.
    try:
        messages, max_tokens, final_candidates, dropped = fit_candidates_to_token_budget(
            effective_question, final_candidates, persona, payload.answer_length,
            payload.table_format, generator, n_ctx=config.N_CTX,
            safety_margin_tokens=config.CONTEXT_SAFETY_MARGIN_TOKENS,
        )
        if dropped:
            logger.warning(
                "Dropped %s lowest-ranked candidate(s) to fit n_ctx=%s for question %r",
                dropped, config.N_CTX, effective_question,
            )
    except ContextBudgetError as e:
        # Even zero evidence chunks don't fit — a genuine capacity
        # problem (answer_length/persona prompt too large for this
        # n_ctx), not something to paper over with a fabricated answer.
        logger.error("Context budget error for question %r: %s", effective_question, e)
        db.record_answer_audit(persona.value, scope, config.KNOWLEDGE_VERSION, [], [], True)
        return AskResponse(
            answer=INSUFFICIENT_EVIDENCE_MESSAGE, sources=[],
            knowledge_version=config.KNOWLEDGE_VERSION, insufficient_evidence=True,
            conversation_id=conversation_id,
            rewritten_question=effective_question if effective_question != question else None,
        )

    t = _mark("context_fit", t)

    prompt_tokens = None
    try:
        prompt_tokens = generator.count_tokens(
            messages[0]["content"] + "\n" + messages[1]["content"]
        )
    except Exception:
        pass
     
    with _gen_lock:
        raw_answer = generator.generate(messages, max_tokens)

    t = _mark("llm_generate", t)

    generated_tokens = None
    try:
        generated_tokens = generator.count_tokens(raw_answer)
    except Exception:
        pass

    retrieved_ids = {c.source_id for c in final_candidates}
    validated = validate_citations(raw_answer, retrieved_ids)
    t = _mark("citation_validate", t)
    if validated.dropped_unsupported_citations:
        logger.warning("Model cited unsupported source id(s) %s; stripped.", validated.dropped_unsupported_citations)

    cited_candidates = [c for c in final_candidates if c.source_id in validated.cited_source_ids] or final_candidates
    sources = [_persona_filtered_source(c, persona) for c in cited_candidates]

    db.record_answer_audit(
        persona.value,
        scope,
        config.KNOWLEDGE_VERSION,
        [c.chunk_id for c in final_candidates],
        validated.cited_source_ids,
        False,
    )
    t = _mark("audit_write", t)

    _persist_turn(validated.text, validated.cited_source_ids, False)
    t = _mark("persist_turn", t)

    _log_timing_summary(
        insufficient=False,
        generated_tokens=generated_tokens,
        requested_max_tokens=max_tokens,
        prompt_tokens=prompt_tokens,
        candidates_used=len(final_candidates),
    )

    return AskResponse(
        answer=validated.text, sources=sources,
        knowledge_version=config.KNOWLEDGE_VERSION, insufficient_evidence=False,
        conversation_id=conversation_id,
        rewritten_question=effective_question if effective_question != question else None,
    )


class ConversationSummary(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    persona: str
    created_at: str
    last_activity_at: str


class ConversationTurnView(BaseModel):
    question: str
    answer: str
    cited_source_ids: list = []
    insufficient_evidence: bool = False
    created_at: str


class ConversationDetail(BaseModel):
    conversation_id: str
    title: Optional[str] = None
    persona: str
    turns: list[ConversationTurnView]


def _authorized_for_conversation(conversation: dict, requested_user_id: Optional[str]) -> bool:
    return is_authorized_for_conversation(conversation, requested_user_id)


@app.get("/api/conversations", response_model=list[ConversationSummary])
def list_conversations(
    user_id: str,
    limit: int = 20,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    """Powers a 'your previous conversations' / resume list. `user_id`
    here is the same TRUSTED-from-Veda value as AskRequest.user_id — see
    that field's docstring. This endpoint does not itself verify the
    caller IS that user; Veda's backend is responsible for only ever
    calling this with the user_id of whoever is actually logged in."""
    require_api_key(x_api_key, authorization)
    rows = state["db"].list_conversations_for_user(user_id, limit=limit)
    return [ConversationSummary(**r) for r in rows]


@app.get("/api/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation_detail(
    conversation_id: str,
    user_id: Optional[str] = None,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    """Full transcript for resuming a conversation later — this is what
    a client calls when the user picks a conversation from their history
    (or reopens the app and wants to continue where they left off).
    Note: the stored `answer` text is exactly what was generated for
    that turn's persona at the time — it is NOT re-filtered here. If the
    same conversation_id were somehow resumed under a different persona
    (shouldn't normally happen — persona is set at conversation creation
    and Veda should keep it stable), old answers would still reflect the
    original persona's depth/terminology. `cited_source_ids` are returned
    as bare IDs, not re-expanded into full persona-filtered source
    objects (see conversation.py round's DELIVERABLES note — this was
    kept deliberately simple rather than adding a document lookup by
    source_id that doesn't exist yet elsewhere in this codebase)."""
    require_api_key(x_api_key, authorization)
    db = state["db"]
    conv = db.get_conversation(conversation_id)
    if conv is None or not _authorized_for_conversation(conv, user_id):
        raise HTTPException(status_code=404, detail="No such conversation.")
    turns = db.get_turns(conversation_id)
    return ConversationDetail(
        conversation_id=conv["conversation_id"], title=conv.get("title"),
        persona=conv["persona"],
        turns=[
            ConversationTurnView(
                question=t.question, answer=t.answer, cited_source_ids=t.cited_source_ids,
                insufficient_evidence=t.insufficient_evidence, created_at=t.created_at,
            )
            for t in turns
        ],
    )


@app.delete("/api/conversations/{conversation_id}")
def delete_conversation_endpoint(
    conversation_id: str,
    user_id: Optional[str] = None,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    """Lets Veda implement a real 'clear my chat' action. Ownership-
    checked the same way as the GET-detail endpoint above. No automatic
    retention/expiry job runs on its own — see
    scripts/cleanup_stale_conversations.py, which needs to be scheduled
    (cron/systemd timer) as an operational step, same caveat as every
    other 'script exists, run it yourself' item in DELIVERABLES.md."""
    require_api_key(x_api_key, authorization)
    db = state["db"]
    conv = db.get_conversation(conversation_id)
    if conv is None or not _authorized_for_conversation(conv, user_id):
        raise HTTPException(status_code=404, detail="No such conversation.")
    db.delete_conversation(conversation_id)
    return {"status": "deleted", "conversation_id": conversation_id}


@app.get("/api/stats", response_model=StatsResponse)
def stats(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    authorization: Optional[str] = Header(default=None),
):
    require_api_key(x_api_key, authorization)
    db = state["db"]
    all_docs = db.list_documents()
    indexed = [d for d in all_docs if d.approval_status == IngestionState.INDEXED]
    return StatsResponse(document_count=len(all_docs), indexed_document_count=len(indexed),
                          knowledge_version=config.KNOWLEDGE_VERSION)


@app.get("/api/health")
def health():
    return {"status": "ok", "ready": "db" in state and "generator" in state,
            "knowledge_version": config.KNOWLEDGE_VERSION,
            "db_backend": config.RAG_DB_BACKEND, "generation_backend": config.RAG_GENERATION_BACKEND}


# NOTE (sections 22-23/46 original spec; X/AE completion follow-up):
# there is intentionally NO /files/{filename} route anywhere in this app.


# ---------------------------------------------------------------------
# ADMIN / INGESTION ROUTER — mount internal-only; block at the reverse
# proxy from any public route (sections E, F, section 45 original spec).
# ---------------------------------------------------------------------
admin_router = APIRouter(prefix="/admin")


class ManualUploadRequest(BaseModel):
    raw_text: str
    format: Literal["text", "markdown"] = "text"
    title: str = ""
    authors: list = []
    journal: str = ""
    publication_date: Optional[str] = None
    doi: str = ""
    pmid: str = ""
    trial_id: str = ""
    cmt_subtypes: list = []
    genes: list = []
    study_type: str = ""
    source_tier: str = "unspecified"
    source_url: str = ""
    actor: str


class DocumentStatusResponse(BaseModel):
    document_id: str
    source_id: Optional[str] = None
    state: str


@admin_router.post("/ingest/manual", response_model=DocumentStatusResponse)
def manual_upload(payload: ManualUploadRequest, x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key")):
    """Text/markdown manual upload — JSON body. For PDF uploads use
    POST /admin/ingest/manual/pdf (multipart), since raw PDF bytes don't
    belong base64-bloated into a JSON body."""
    require_admin_key(x_admin_api_key)
    db, embedder, vector_store = state["db"], state["embedder"], state["vector_store"]
    ingestion_input = IngestionInput(
        raw_text=payload.raw_text, format=payload.format, title=payload.title,
        authors=payload.authors, journal=payload.journal, publication_date=payload.publication_date,
        doi=payload.doi, pmid=payload.pmid, trial_id=payload.trial_id,
        cmt_subtypes=payload.cmt_subtypes, genes=payload.genes, study_type=payload.study_type,
        source_tier=payload.source_tier, source_url=payload.source_url, ingestion_method="manual_upload",
    )
    try:
        doc = ingest_document(db, embedder, vector_store, ingestion_input, actor=payload.actor)
    except DuplicateDocumentError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except IngestionValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    logger.info("AUDIT admin manual_upload document_id=%s actor=%s", doc.document_id, payload.actor)
    return DocumentStatusResponse(document_id=doc.document_id, source_id=doc.source_id, state=doc.approval_status.value)


@admin_router.post("/ingest/manual/pdf", response_model=DocumentStatusResponse)
async def manual_upload_pdf(
    file: UploadFile = File(...),
    actor: str = Form(...),
    title: str = Form(""),
    authors_csv: str = Form(""),
    journal: str = Form(""),
    publication_date: Optional[str] = Form(None),
    doi: str = Form(""),
    pmid: str = Form(""),
    trial_id: str = Form(""),
    cmt_subtypes_csv: str = Form(""),
    genes_csv: str = Form(""),
    study_type: str = Form(""),
    source_tier: str = Form("unspecified"),
    source_url: str = Form(""),
    x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key"),
):
    """Real authenticated admin PDF upload path (final hardening brief,
    item 3): PDF -> validation -> metadata -> (already-)approval ->
    text extraction -> scientific chunking -> embeddings -> lexical/
    vector index -> INDEXED, via the SAME ingest_document() pipeline
    Discovery and text/markdown manual uploads use — no separate PDF-only
    code path.

    Uses multipart/form-data (FastAPI's `UploadFile` + `Form(...)`
    fields) rather than JSON+base64, which is the standard approach for
    binary uploads and avoids ~33% payload bloat. This requires the
    `python-multipart` package in the deployment venv in addition to
    `fastapi` — note this explicitly since it's an easy dependency to
    miss. The uploaded bytes are read into memory, passed to
    `ingest_document()` for text extraction, and then go out of scope —
    they are never written to disk or exposed via any route (see
    ingestion_pipeline.py's docstring).
    """
    require_admin_key(x_admin_api_key)

    pdf_bytes = await file.read()
    db, embedder, vector_store = state["db"], state["embedder"], state["vector_store"]
    ingestion_input = IngestionInput(
        raw_bytes=pdf_bytes, format="pdf", title=title,
        authors=[a.strip() for a in authors_csv.split(",") if a.strip()],
        journal=journal, publication_date=publication_date, doi=doi, pmid=pmid, trial_id=trial_id,
        cmt_subtypes=[s.strip() for s in cmt_subtypes_csv.split(",") if s.strip()],
        genes=[g.strip() for g in genes_csv.split(",") if g.strip()],
        study_type=study_type, source_tier=source_tier, source_url=source_url,
        ingestion_method="manual_upload",
    )
    try:
        doc = ingest_document(db, embedder, vector_store, ingestion_input, actor=actor)
    except DuplicateDocumentError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except IngestionValidationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    logger.info("AUDIT admin manual_upload_pdf document_id=%s actor=%s filename=%s",
                doc.document_id, actor, getattr(file, "filename", "?"))
    return DocumentStatusResponse(document_id=doc.document_id, source_id=doc.source_id, state=doc.approval_status.value)


class DiscoveryRegisterRequest(BaseModel):
    discovery_candidate_id: str
    proposed_title: str
    proposed_source_url: str = ""


@admin_router.post("/discovery/register")
def discovery_register(payload: DiscoveryRegisterRequest, x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key")):
    require_admin_key(x_admin_api_key)
    db = state["db"]
    try:
        register_candidate(db, DiscoveryCandidatePayload(**payload.model_dump()))
    except DiscoveryRejectionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"status": "registered", "discovery_candidate_id": payload.discovery_candidate_id}


class DiscoveryApproveRequest(BaseModel):
    discovery_candidate_id: str
    source_text: str
    metadata: ManualUploadRequest


@admin_router.post("/discovery/approve", response_model=DocumentStatusResponse)
def discovery_approve(payload: DiscoveryApproveRequest, x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key")):
    require_admin_key(x_admin_api_key)
    db, embedder, vector_store = state["db"], state["embedder"], state["vector_store"]
    m = payload.metadata
    ingestion_input = IngestionInput(
        raw_text="", format=m.format, title=m.title, authors=m.authors, journal=m.journal,
        publication_date=m.publication_date, doi=m.doi, pmid=m.pmid, trial_id=m.trial_id,
        cmt_subtypes=m.cmt_subtypes, genes=m.genes, study_type=m.study_type,
        source_tier=m.source_tier, source_url=m.source_url,
    )
    try:
        doc = approve_and_ingest(db, embedder, vector_store, payload.discovery_candidate_id,
                                  payload.source_text, ingestion_input, reviewed_by=m.actor)
    except DiscoveryRejectionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    logger.info("AUDIT admin discovery_approve candidate=%s document_id=%s actor=%s",
                payload.discovery_candidate_id, doc.document_id, m.actor)
    return DocumentStatusResponse(document_id=doc.document_id, source_id=doc.source_id, state=doc.approval_status.value)


class DiscoveryRejectRequest(BaseModel):
    discovery_candidate_id: str
    actor: str


@admin_router.post("/discovery/reject")
def discovery_reject(payload: DiscoveryRejectRequest, x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key")):
    require_admin_key(x_admin_api_key)
    db = state["db"]
    try:
        reject_candidate(db, payload.discovery_candidate_id, reviewed_by=payload.actor)
    except DiscoveryRejectionError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {"status": "rejected", "discovery_candidate_id": payload.discovery_candidate_id}


class RemoveRequest(BaseModel):
    document_id: str
    actor: str


@admin_router.post("/ingest/remove", response_model=DocumentStatusResponse)
def remove_document_endpoint(payload: RemoveRequest, x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key")):
    require_admin_key(x_admin_api_key)
    db = state["db"]
    try:
        doc = remove_document(db, payload.document_id, actor=payload.actor)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    logger.info("AUDIT admin remove document_id=%s actor=%s", payload.document_id, payload.actor)
    return DocumentStatusResponse(document_id=doc.document_id, source_id=doc.source_id, state=doc.approval_status.value)


@admin_router.get("/ingest/status/{document_id}", response_model=DocumentStatusResponse)
def ingestion_status(document_id: str, x_admin_api_key: Optional[str] = Header(default=None, alias="X-Admin-API-Key")):
    require_admin_key(x_admin_api_key)
    db = state["db"]
    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="No such document.")
    return DocumentStatusResponse(document_id=doc.document_id, source_id=doc.source_id, state=doc.approval_status.value)


app.include_router(admin_router)


# ---- serve the frontend (unchanged; no frontend files were modified) ----
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/")
    def root():
        return FileResponse(static_dir / "index.html")
