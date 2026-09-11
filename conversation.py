"""
conversation.py
-----------------
Multi-turn conversation support. The one architectural rule this module
exists to protect: conversation history may only influence what question
gets RETRIEVED — it must never be fed into the GENERATION prompt as fact.

    turn 2: "What causes it?"
        -> (this module) resolve "it" using turn 1's context
        -> "What causes CMT4C?"
        -> the EXACT same retrieve -> evidence-gate -> generate pipeline
           any other question goes through, with fresh evidence pulled
           for "What causes CMT4C?" specifically.

If conversation history were instead spliced into the generation prompt,
the model could restate a fact from turn 1's answer in turn 2 even though
turn 2's own retrieval never surfaced it — silently breaking the grounding
guarantee ("the supplied retrieved evidence is your EXCLUSIVE factual
source") the rest of this system is built around. Keeping history
confined to the rewrite step, upstream of retrieval, is what prevents
that.

Two rewriters, same pattern as embeddings.py/generation.py's Real-vs-
sandbox-substitute split — except here the "sandbox substitute" is
actually a reasonable lightweight production component in its own right,
not just a stand-in:

  * `HeuristicQueryRewriter` — deterministic, no model call, handles the
    common case (a short follow-up with a pronoun/no entities of its
    own) by splicing in the most recent turn's specific entity. Real,
    tested, and genuinely useful in production as a fast first pass —
    skipping an LLM call for the common case matters on a CPU-bound
    2 vCPU box where every generation call already costs ~4s.
  * `LLMQueryRewriter` — real code, uses the same `Generator` interface
    as answer generation, for follow-ups the heuristic can't resolve
    (e.g. genuinely ambiguous reference, or a rephrasing rather than a
    pronoun). NOT executed anywhere in this sandbox (no llama_cpp/model).
    Doubles inference cost for any turn that needs it — benchmark this
    before deciding how often it fires in practice.
"""

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from db import TurnRecord
from entities import extract_entities, ExtractedEntities

_PRONOUN_RE = re.compile(r"\b(it|this|that|these|those|its)\b", re.IGNORECASE)
_MAX_HISTORY_FOR_REWRITE = 4  # only recent turns are relevant to a follow-up


@dataclass
class RewriteResult:
    question: str          # what should actually go to retrieval
    was_rewritten: bool
    method: str             # "none" | "heuristic" | "llm"


def _needs_rewrite(question: str, ents: ExtractedEntities) -> bool:
    """A question that already names its own gene/subtype/mutation/trial
    is self-contained — never rewrite it, regardless of history. This is
    the same 'specific vs general' signal retrieval.py already uses."""
    if ents.specific_terms():
        return False
    if _PRONOUN_RE.search(question):
        return True
    # Short, entity-free questions ("Why?", "How is it treated?") are
    # ambiguous enough to be worth trying to resolve against history even
    # without an explicit pronoun.
    return len(question.split()) <= 6


def _most_recent_specific_term(history: list[TurnRecord]) -> str | None:
    for turn in reversed(history):
        prior_text = turn.rewritten_question or turn.question
        prior_ents = extract_entities(prior_text)
        specific = prior_ents.specific_terms()
        if specific:
            # Deterministic choice among multiple candidates: prefer a
            # named subtype (e.g. "CMT4C") over a bare gene symbol as the
            # more natural noun to splice into a sentence, then fall back
            # to sorted order for stability.
            subtypes = prior_ents.subtypes
            if subtypes:
                return sorted(subtypes)[0]
            return sorted(specific)[0]
    return None


def _splice_entity_into_question(question: str, term: str) -> str:
    q = question.rstrip("?!. ").strip()
    if _PRONOUN_RE.search(q):
        return _PRONOUN_RE.sub(term, q, count=1) + "?"
    # No pronoun to replace (e.g. a bare "Why?" or "What treatments are
    # available?") -> append a clarifying qualifier rather than guessing
    # where to insert the term mid-sentence.
    return f"{q} for {term}?"


class QueryRewriter(ABC):
    @abstractmethod
    def rewrite(self, question: str, history: list[TurnRecord]) -> RewriteResult: ...


class HeuristicQueryRewriter(QueryRewriter):
    def rewrite(self, question: str, history: list[TurnRecord]) -> RewriteResult:
        ents = extract_entities(question)
        if not _needs_rewrite(question, ents):
            return RewriteResult(question=question, was_rewritten=False, method="none")

        recent_history = history[-_MAX_HISTORY_FOR_REWRITE:]
        term = _most_recent_specific_term(recent_history)
        if term is None:
            # Nothing specific to resolve against (e.g. the whole
            # conversation has been general so far) — pass the question
            # through unchanged rather than guessing.
            return RewriteResult(question=question, was_rewritten=False, method="none")

        rewritten = _splice_entity_into_question(question, term)
        return RewriteResult(question=rewritten, was_rewritten=True, method="heuristic")


_REWRITE_SYSTEM_PROMPT = """You rewrite a follow-up question into a standalone question using the
recent conversation, so it can be searched on its own. Rules:
- Resolve pronouns/references (it, that, this) to the specific thing they refer to.
- Do NOT answer the question. Do NOT add facts not already named in the conversation.
- If the question is already standalone, return it unchanged.
- Output ONLY the rewritten question, nothing else."""


class LLMQueryRewriter(QueryRewriter):
    """Real code, uses the same Generator interface as answer generation
    (generation.py). NOT executed anywhere in this sandbox — no
    llama_cpp, no model weights. See module docstring for the
    performance tradeoff (an extra ~4s inference call on the reference
    2 vCPU box) before wiring this in as the default path rather than a
    fallback behind HeuristicQueryRewriter."""

    def __init__(self, generator):
        self._generator = generator

    def rewrite(self, question: str, history: list[TurnRecord]) -> RewriteResult:
        ents = extract_entities(question)
        if not _needs_rewrite(question, ents):
            return RewriteResult(question=question, was_rewritten=False, method="none")

        recent_history = history[-_MAX_HISTORY_FOR_REWRITE:]
        if not recent_history:
            return RewriteResult(question=question, was_rewritten=False, method="none")

        history_text = "\n".join(
            f"Q: {t.rewritten_question or t.question}\nA: {t.answer}" for t in recent_history
        )
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
            {"role": "user", "content": f"Conversation so far:\n{history_text}\n\nFollow-up: {question}\n\nStandalone question:"},
        ]
        rewritten = self._generator.generate(messages, max_tokens=80).strip()
        if not rewritten:
            return RewriteResult(question=question, was_rewritten=False, method="none")
        return RewriteResult(question=rewritten, was_rewritten=True, method="llm")


class FallbackQueryRewriter(QueryRewriter):
    """Production-shaped composition: try the free heuristic first (no
    model call); only fall back to the LLM rewriter if the heuristic
    couldn't resolve anything but the question still looks like it needs
    resolving. Untested end-to-end here (the LLM half can't run), but the
    heuristic-only branch is exercised by every test in
    tests/test_conversation.py."""

    def __init__(self, llm_rewriter: LLMQueryRewriter | None = None):
        self._heuristic = HeuristicQueryRewriter()
        self._llm = llm_rewriter

    def rewrite(self, question: str, history: list[TurnRecord]) -> RewriteResult:
        result = self._heuristic.rewrite(question, history)
        if result.was_rewritten or self._llm is None:
            return result
        ents = extract_entities(question)
        if _needs_rewrite(question, ents) and history:
            return self._llm.rewrite(question, history)
        return result


def is_authorized_for_conversation(conversation: dict, requested_user_id: str | None) -> bool:
    """A conversation created with no user_id (client chose not to pass
    one) is treated as unowned — possessing the conversation_id itself is
    the credential. A conversation that DOES have a user_id requires the
    request's user_id to match. Kept here (not inline in main.py) so it's
    testable without needing fastapi importable — this is pure
    authorization logic with no HTTP dependency, and main.py's endpoints
    are meant to be thin callers of it, not where the rule itself lives.
    A mismatch should be surfaced by the caller as 404, not 403 — so an
    unauthorized caller can't distinguish 'wrong owner' from 'doesn't
    exist', which is the standard pattern for this kind of check."""
    stored_user_id = conversation.get("user_id")
    if stored_user_id is None:
        return True
    return stored_user_id == requested_user_id
