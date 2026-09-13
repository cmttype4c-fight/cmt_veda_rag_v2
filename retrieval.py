"""
retrieval.py
------------
Retrieval pipeline for RAG v2 (sections 6-9, 13; corrected per the
completion follow-up sections H-M):

    query understanding -> broad candidate pool (semantic + DB-backed
        lexical search) -> metadata-aware DETERMINISTIC rerank ->
        source diversification -> evidence sufficiency check
        -> final context

Two things the completion follow-up specifically called out and this
version addresses:

  1. Lexical search is no longer an in-memory rescan-every-chunk BM25
     (that was fine as a dev baseline but doesn't scale to thousands of
     documents). It now goes through db.py's `lexical_search()`, which is
     backed by SQLite FTS5 here and Postgres tsvector/GIN in production —
     both are indexed, persistent lexical search, not a Python loop over
     every chunk on every query.

  2. The reranker is explicitly a DETERMINISTIC weighted combiner of
     semantic + lexical + metadata signals — NOT a neural/cross-encoder
     reranker. It is described that way everywhere in this codebase and
     in DELIVERABLES.md. If a real cross-encoder is later benchmarked and
     justified, only `rerank()`'s scoring function needs to change.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

from config import (
    CANDIDATE_POOL_MAX, MAX_CHUNKS_PER_DOC_GENERAL,
    FINAL_CONTEXT_CHUNKS_MIN, FINAL_CONTEXT_CHUNKS_MAX,
    WEIGHT_SEMANTIC, WEIGHT_LEXICAL, WEIGHT_METADATA,
    EVIDENCE_SCORE_FLOOR, EVIDENCE_MIN_SUPPORTING_CHUNKS,
    LEXICAL_SATURATION,
)
from entities import extract_entities, classify_query_scope, ExtractedEntities
from db import DBBackend
from embeddings import EmbeddingBackend
from vector_store import VectorStore


@dataclass
class Candidate:
    chunk_id: str
    document_id: str
    source_id: str
    text: str
    section: str = "body"
    # RAW per-signal scores as returned by the retrievers. These are never
    # overwritten/normalized in place — pool-relative normalization (needed
    # for fair ranking across a candidate set) lives only in rank_score, so
    # the evidence-sufficiency gate can still judge each candidate on its
    # own absolute merit even when the candidate pool is small or uniform.
    semantic_score: float = 0.0   # cosine similarity, 0-1
    lexical_score: float = 0.0    # raw FTS rank score, unbounded
    metadata_score: float = 0.0   # 0-1 boost from entity/tier matches
    rank_score: float = 0.0       # pool-normalized composite, ordering only
    confidence: float = 0.0       # absolute composite, used for the evidence gate
    metadata: dict = field(default_factory=dict)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def _metadata_boost(candidate_meta: dict, query_ents: ExtractedEntities) -> float:
    """Section 10/J examples: exact gene query -> gene boost, subtype
    query -> subtype boost, PMID/trial-ID exact match -> boost. Purely
    additive signal, capped at 1.0. This IS the "exact scientific
    terminology matters" mechanism (completion follow-up section J) —
    semantic similarity alone never decides an exact-entity query."""
    boost = 0.0
    genes = {g.upper() for g in candidate_meta.get("genes", [])}
    subtypes = {s.upper() for s in candidate_meta.get("cmt_subtypes", [])}
    if query_ents.genes and (genes & query_ents.genes):
        boost += 0.5
    if query_ents.subtypes and (subtypes & query_ents.subtypes):
        boost += 0.5
    if query_ents.trial_ids and candidate_meta.get("trial_id", "").upper() in query_ents.trial_ids:
        boost += 0.5
    if query_ents.pmids and candidate_meta.get("pmid", "") in query_ents.pmids:
        boost += 0.5
    return min(boost, 1.0)


def _metadata_score_for(meta: dict, query_ents: ExtractedEntities, query_scope: str) -> float:
    boost = _metadata_boost(meta, query_ents)
    if query_scope == "general" and meta.get("source_tier") in (
        "review", "guideline", "peer_reviewed",
    ):
        boost = min(boost + 0.15, 1.0)
    return boost


def rerank(
    candidates: list[Candidate],
    question: str,
    query_ents: ExtractedEntities,
    query_scope: str,
) -> list[Candidate]:
    """Deterministic hybrid scoring — explicitly NOT a neural reranker
    (completion follow-up section K). Pool-relative normalization is used
    ONLY for ordering/diversification, never for the absolute
    evidence-sufficiency judgment (see `confidence` on Candidate)."""
    sem_norm = _normalize({c.chunk_id: c.semantic_score for c in candidates})
    lex_norm = _normalize({c.chunk_id: c.lexical_score for c in candidates})

    for c in candidates:
        c.metadata_score = _metadata_score_for(c.metadata, query_ents, query_scope)

        c.rank_score = (
            WEIGHT_SEMANTIC * sem_norm.get(c.chunk_id, 0.0)
            + WEIGHT_LEXICAL * lex_norm.get(c.chunk_id, 0.0)
            + WEIGHT_METADATA * c.metadata_score
        )

        lexical_saturated = min(c.lexical_score / LEXICAL_SATURATION, 1.0)
        c.confidence = (
            WEIGHT_SEMANTIC * c.semantic_score
            + WEIGHT_LEXICAL * lexical_saturated
            + WEIGHT_METADATA * c.metadata_score
        )

    return sorted(candidates, key=lambda c: c.rank_score, reverse=True)


def diversify(ranked: list[Candidate], query_scope: str) -> list[Candidate]:
    """Section 9/L: don't let one narrow-subtype paper dominate a general
    question; specific questions are exempt. The per-document cap is a
    hard ceiling for general questions and is never relaxed to hit the
    target minimum — that relaxation was the exact section-48 bug."""
    if query_scope != "general":
        return ranked[:FINAL_CONTEXT_CHUNKS_MAX]

    per_doc_count: dict[str, int] = defaultdict(int)
    kept: list[Candidate] = []
    for c in ranked:
        if per_doc_count[c.document_id] < MAX_CHUNKS_PER_DOC_GENERAL:
            kept.append(c)
            per_doc_count[c.document_id] += 1
        if len(kept) >= FINAL_CONTEXT_CHUNKS_MAX:
            break
    return kept


# ---------------------------------------------------------------------
# EVIDENCE SUFFICIENCY (section 13, corrected per completion follow-up
# section M: distinguish "mentioned" from "actually answers").
# ---------------------------------------------------------------------
_OUTCOME_LANGUAGE = {
    "efficacy", "effective", "ineffective", "outcome", "outcomes", "results",
    "result", "trial", "significant", "significantly", "improved", "improvement",
    "reduced", "reduction", "response", "responded", "survival", "randomized",
    "placebo", "p<", "p =", "p=", "confidence interval", "hazard ratio",
    "follow-up", "endpoint", "endpoints",
}
_OUTCOME_SEEKING_QUESTION_WORDS = {
    "efficacy", "effective", "effectiveness", "work", "works", "treatment",
    "therapy", "cure", "outcome", "trial", "benefit", "improve",
}


def _requires_outcome_evidence(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in _OUTCOME_SEEKING_QUESTION_WORDS)


def _has_outcome_language(text: str) -> bool:
    t = text.lower()
    return any(term in t for term in _OUTCOME_LANGUAGE)


@dataclass
class EvidenceAssessment:
    sufficient: bool
    reason: str
    supporting_chunks: list[Candidate]


def assess_evidence_sufficiency(ranked: list[Candidate], question: str = "") -> EvidenceAssessment:
    """Refuse rather than guess when evidence is thin OR when the
    question asks for a claim type (efficacy/outcome) that the retrieved
    text doesn't actually contain (completion follow-up section M).

    Two independent signals count as "strong evidence", not one blended
    score>threshold check (section M explicitly warns against relying on
    only `score > threshold`):

      1. Blended `confidence` clears the floor — the general case.
      2. EXACT entity match (section J: "semantic similarity alone should
         not be the only mechanism"): a candidate whose metadata contains
         an exact gene/subtype/trial-ID/PMID match for an entity the
         question explicitly named, AND where that term is actually
         present in the chunk's text (not just tagged in metadata with no
         textual grounding), counts as strong evidence even if a
         semantic-similarity-heavy blended score happens to sit just under
         the floor. A weak cosine-similarity number must not be allowed to
         veto an exact, textually-grounded terminology match — that would
         make semantic similarity the sole arbiter after all.
    """
    confidence_backed = [c for c in ranked if c.confidence >= EVIDENCE_SCORE_FLOOR]
    exact_entity_backed = [
        c for c in ranked
        if c.metadata_score >= 0.5 and c.lexical_score > 0 and c not in confidence_backed
    ]
    strong = confidence_backed + exact_entity_backed

    if len(strong) < EVIDENCE_MIN_SUPPORTING_CHUNKS:
        return EvidenceAssessment(
            sufficient=False,
            reason=(
                f"Only {len(strong)} candidate chunk(s) cleared the evidence "
                f"floor ({EVIDENCE_SCORE_FLOOR}) or had an exact, "
                f"textually-grounded entity match; need at least "
                f"{EVIDENCE_MIN_SUPPORTING_CHUNKS}."
            ),
            supporting_chunks=strong,
        )

    if question and _requires_outcome_evidence(question):
        outcome_backed = [c for c in strong if _has_outcome_language(c.text)]
        if not outcome_backed:
            return EvidenceAssessment(
                sufficient=False,
                reason=(
                    "The question asks about efficacy/outcome, but the "
                    "retrieved evidence only mentions the relevant term(s) "
                    "without outcome/result language — treating 'mentioned' "
                    "as insufficient to 'actually evidences the claim'."
                ),
                supporting_chunks=strong,
            )
        return EvidenceAssessment(sufficient=True, reason="ok (outcome-backed)", supporting_chunks=outcome_backed)

    return EvidenceAssessment(sufficient=True, reason="ok", supporting_chunks=strong)


# ---------------------------------------------------------------------
# HYBRID RETRIEVER — DB-backed (sections 6-8, corrected per section I)
# ---------------------------------------------------------------------
class HybridRetriever:
    """Real semantic (vector_store) + real lexical (db.lexical_search,
    SQLite FTS5 / Postgres tsvector) hybrid retrieval. Both retrieval
    calls are indexed/persistent lookups, not a full corpus scan in
    Python — see module docstring.

    Removal enforcement (section G): `db.lexical_search()` already
    filters to INDEXED documents at the SQL layer. The vector store does
    NOT know about document lifecycle at all (by design — see
    vector_store.py), so its results are filtered here against
    `db.get_active_chunk_ids()` before anything else happens. This is the
    "authoritative active-document filter" the spec asked for: it's
    enforced once, at the retrieval entry point, not scattered across
    call sites.
    """

    def __init__(self, db: DBBackend, embedder: EmbeddingBackend, vector_store: VectorStore):
        self.db = db
        self.embedder = embedder
        self.vector_store = vector_store
        self._doc_metadata_cache: dict[str, dict] = {}

    def _metadata_for_chunk(self, document_id: str) -> dict:
        if document_id not in self._doc_metadata_cache:
            doc = self.db.get_document(document_id)
            self._doc_metadata_cache[document_id] = {
                "genes": doc.genes if doc else [],
                "cmt_subtypes": doc.cmt_subtypes if doc else [],
                "source_tier": doc.source_tier if doc else "unspecified",
                "trial_id": doc.trial_id if doc else "",
                "pmid": doc.pmid if doc else "",
                "title": doc.title if doc else "",
                "authors": doc.authors if doc else [],
                "journal": doc.journal if doc else "",
                "publication_date": doc.publication_date if doc else None,
                "doi": doc.doi if doc else "",
                "study_type": doc.study_type if doc else "",
                "source_url": doc.source_url if doc else "",
            } if doc else {}
        return self._doc_metadata_cache[document_id]

    def retrieve(self, question: str, k: int = CANDIDATE_POOL_MAX) -> list[Candidate]:
        active_ids = self.db.get_active_chunk_ids()  # authoritative, section G

        # --- semantic side ---
        qvec = self.embedder.embed([question])[0]
        semantic_hits = {
            chunk_id: sim
            for chunk_id, sim in self.vector_store.search(qvec, k=k)
            if chunk_id in active_ids  # removed/not-yet-indexed — never surfaces
        }

        # --- lexical side (already INDEXED-filtered at the SQL layer) ---
        lexical_hits = {
            chunk_id: score
            for chunk_id, score in self.db.lexical_search(question, limit=k)
            if chunk_id in active_ids  # defense in depth
        }

        # PERFORMANCE FIX: batch-fetch chunk/document data instead of
        # querying once per candidate.
        #
        # Ordering is deliberately preserved:
        # 1. semantic hits in similarity-rank order
        # 2. lexical-only hits in lexical-rank order
        #
        # This matches the original insertion order and therefore preserves
        # the behavior of the final CANDIDATE_POOL_MAX truncation.

        ordered_chunk_ids = list(semantic_hits.keys()) + [
            cid for cid in lexical_hits.keys()
            if cid not in semantic_hits
        ]

        if not ordered_chunk_ids:
            return []

        # Batch query 1: chunk -> document mapping
        chunk_id_to_document_id = self.db.get_document_ids_for_chunks(
            ordered_chunk_ids
        )

        surviving_chunk_ids = [
            cid
            for cid in ordered_chunk_ids
            if cid in chunk_id_to_document_id
        ]

        if not surviving_chunk_ids:
            return []

        # Batch query 2: fetch all required chunks
        chunks_by_id = self.db.get_chunks_by_ids(surviving_chunk_ids)

        # Keep document order deterministic and remove duplicates
        unique_document_ids = list(
            dict.fromkeys(
                chunk_id_to_document_id[cid]
                for cid in surviving_chunk_ids
            )
        )

        # Batch query 3: fetch all required documents
        documents_by_id = self.db.get_documents_by_ids(
            unique_document_ids
        )

        # Populate metadata cache from the batch result so
        # _metadata_for_chunk() remains correct for other callers,
        # without generating additional DB queries.
        for document_id, doc in documents_by_id.items():
            self._doc_metadata_cache[document_id] = {
    "genes": doc.genes if doc else [],
    "cmt_subtypes": doc.cmt_subtypes if doc else [],
    "source_tier": doc.source_tier if doc else "unspecified",
    "trial_id": doc.trial_id if doc else "",
    "pmid": doc.pmid if doc else "",
    "title": doc.title if doc else "",
    "authors": doc.authors if doc else [],
    "journal": doc.journal if doc else "",
    "publication_date": doc.publication_date if doc else None,
    "doi": doc.doi if doc else "",
    "study_type": doc.study_type if doc else "",
    "source_url": doc.source_url if doc else "",
} if doc else {}

        candidates: dict[str, Candidate] = {}

        for chunk_id in surviving_chunk_ids:
            document_id = chunk_id_to_document_id[chunk_id]
            chunk = chunks_by_id.get(chunk_id)
            doc = documents_by_id.get(document_id)

            candidates[chunk_id] = Candidate(
                chunk_id=chunk_id,
                document_id=document_id,
                source_id=doc.source_id if doc else "",
                text=chunk.content if chunk else "",
                section=chunk.section if chunk else "body",
                semantic_score=float(semantic_hits.get(chunk_id, 0.0)),
                lexical_score=float(lexical_hits.get(chunk_id, 0.0)),
                metadata=self._doc_metadata_cache.get(document_id, {}),
            )

        return list(candidates.values())[:CANDIDATE_POOL_MAX]

    def retrieve_and_rank(self, question: str) -> tuple[list[Candidate], str, ExtractedEntities]:
        self._doc_metadata_cache.clear()
        ents = extract_entities(question)
        scope = classify_query_scope(question, ents)
        candidates = self.retrieve(question)
        ranked = rerank(candidates, question, ents, scope)
        final = diversify(ranked, scope)
        return final, scope, ents
