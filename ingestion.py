"""
ingestion.py
------------
Document ingestion for RAG v2: structured chunking (sections 11-12), the
per-chunk metadata schema (section 10), stable public-safe Source IDs
(section 26), and the Discovery/manual-upload lifecycle state machine
(sections 19-21, 44).

This module deliberately does not hard-code a specific database driver.
`SourceIdRegistry` and `IngestionRecord` are storage-agnostic; wire
`SourceIdRegistry.persist()`/`load()` to your actual Postgres table
(see db/schema.sql) in the Platform 1 branch.
"""

import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional

from config import IngestionState, SOURCE_ID_PREFIX, SOURCE_ID_DIGITS
import config

# ---------------------------------------------------------------------
# METADATA SCHEMA (section 10)
# ---------------------------------------------------------------------
@dataclass
class DocumentMetadata:
    document_id: str
    title: str = ""
    authors: list = field(default_factory=list)
    journal: str = ""
    publication_date: Optional[str] = None
    doi: str = ""
    pmid: str = ""
    trial_id: str = ""
    cmt_subtypes: list = field(default_factory=list)
    genes: list = field(default_factory=list)
    study_type: str = ""
    source_tier: str = "unspecified"  # e.g. peer_reviewed, review, guideline,
                                        # trial, preclinical, conference, other
    source_url: str = ""              # canonical PUBLIC bibliographic URL only
                                        # (never an internal file path)
    discovery_candidate_id: Optional[str] = None
    ingestion_method: str = "manual_upload"  # or "discovery"
    approval_status: IngestionState = IngestionState.PENDING_APPROVAL
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    knowledge_version: Optional[str] = None
    # internal-only, never serialized into any persona-facing response:
    internal_storage_path: str = ""


@dataclass
class ChunkMetadata:
    chunk_id: str
    document_id: str
    source_id: str            # public-safe, e.g. "CMT-RAG-000427"
    section: str = "body"     # title/abstract/introduction/methods/results/
                               # discussion/conclusion/table/figure_caption
    order: int = 0


# ---------------------------------------------------------------------
# STABLE, FILENAME-INDEPENDENT SOURCE IDS (section 26)
# ---------------------------------------------------------------------
class SourceIdRegistry:
    """Maps document_id <-> stable public Source ID.

    The ID must survive reindexing and never leak filesystem/storage
    structure. In production this table lives in Postgres (see
    db/schema.sql: source_id_registry); this in-memory version is a
    drop-in reference implementation with load()/persist() hooks.
    """

    def __init__(self):
        self._doc_to_source: dict[str, str] = {}
        self._source_to_doc: dict[str, str] = {}
        self._counter = 0

    def get_or_create(self, document_id: str) -> str:
        if document_id in self._doc_to_source:
            return self._doc_to_source[document_id]
        self._counter += 1
        source_id = f"{SOURCE_ID_PREFIX}{self._counter:0{SOURCE_ID_DIGITS}d}"
        self._doc_to_source[document_id] = source_id
        self._source_to_doc[source_id] = document_id
        return source_id

    def resolve_document_id(self, source_id: str) -> Optional[str]:
        return self._source_to_doc.get(source_id)

    # --- persistence hooks (wire to Postgres in production) ---
    def load(self, rows: list[tuple[str, str]]):
        for document_id, source_id in rows:
            self._doc_to_source[document_id] = source_id
            self._source_to_doc[source_id] = document_id
            n = int(source_id.replace(SOURCE_ID_PREFIX, "") or 0)
            self._counter = max(self._counter, n)

    def persist_rows(self) -> list[tuple[str, str]]:
        return list(self._doc_to_source.items())


# ---------------------------------------------------------------------
# STRUCTURED SCIENTIFIC-PAPER CHUNKING (sections 11-12)
# ---------------------------------------------------------------------
_SECTION_HEADERS = [
    "abstract", "introduction", "background", "methods", "materials and methods",
    "results", "discussion", "conclusion", "conclusions",
    "supplementary information", "supplementary material",
]
_SECTION_HEADER_RE = re.compile(
    r"^\s*(" + "|".join(re.escape(h) for h in _SECTION_HEADERS) + r")\s*:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_TABLE_CAPTION_RE = re.compile(r"^\s*(Table\s+\d+[a-zA-Z]?\.?.*)$", re.MULTILINE)
_FIGURE_CAPTION_RE = re.compile(r"^\s*(Figure\s+\d+[a-zA-Z]?\.?.*)$", re.MULTILINE)

MAX_CHUNK_CHARS = 1800
MIN_CHUNK_CHARS = 200


def split_into_sections(full_text: str) -> list[tuple[str, str]]:
    """Split raw paper text into (section_name, section_text) preserving
    structure instead of cutting at arbitrary character boundaries.
    Falls back to a single 'body' section if no headers are detected.
    """
    matches = list(_SECTION_HEADER_RE.finditer(full_text))
    if not matches:
        return [("body", full_text)]

    sections = []
    # Anything before the first detected header still matters (often the
    # title/author block or an unheaded abstract) — keep it as 'front_matter'.
    if matches[0].start() > 0:
        front = full_text[: matches[0].start()].strip()
        if front:
            sections.append(("front_matter", front))

    for i, m in enumerate(matches):
        name = m.group(1).strip().lower()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        body = full_text[start:end].strip()
        if body:
            sections.append((name, body))
    return sections


_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])")


def _split_oversize_piece(piece: str, limit: int) -> list[str]:
    """Split ONE paragraph that is longer than `limit` without losing any
    text: first on sentence boundaries, then (for a "sentence" that is still
    too long, e.g. a table dump or text with no punctuation) on whitespace,
    and only as a last resort mid-token. Previously such a paragraph became a
    single multi-thousand-character chunk, which the embedder then silently
    truncated -- most of a long XML-derived paper (often one block of text)
    would never have been searchable."""
    if len(piece) <= limit:
        return [piece]
    out, current = [], ""
    for sentence in _SENTENCE_END_RE.split(piece):
        if len(sentence) > limit:
            if current:
                out.append(current)
                current = ""
            words, line = sentence.split(), ""
            for w in words:
                if len(w) > limit:                      # no whitespace at all: hard wrap
                    if line:
                        out.append(line)
                        line = ""
                    out.extend(w[i:i + limit] for i in range(0, len(w), limit))
                elif len(line) + len(w) + 1 <= limit:
                    line = f"{line} {w}".strip()
                else:
                    out.append(line)
                    line = w
            if line:
                out.append(line)
        elif len(current) + len(sentence) + 1 <= limit:
            current = f"{current} {sentence}".strip()
        else:
            if current:
                out.append(current)
            current = sentence
    if current:
        out.append(current)
    return out


def _chunk_long_section(section_name: str, text: str) -> list[str]:
    """Break an over-long section into pieces of at most ~MAX_CHUNK_CHARS on
    paragraph boundaries (then sentence / whitespace boundaries for a single
    over-long paragraph) so we never split mid-sentence/mid-table where
    avoidable AND never drop or oversize any text. The whole section is always
    covered: nothing is truncated."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    paragraphs = re.split(r"\n\s*\n", text)
    chunks, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) + 2 <= MAX_CHUNK_CHARS:
            current = f"{current}\n\n{para}".strip()
        else:
            if current:
                chunks.append(current)
                current = ""
            if len(para) > MAX_CHUNK_CHARS:
                pieces = _split_oversize_piece(para, MAX_CHUNK_CHARS)
                chunks.extend(pieces[:-1])
                current = pieces[-1]
            else:
                current = para
    if current:
        chunks.append(current)
    return chunks or [text]


def extract_tables_and_figures(text: str) -> list[tuple[str, str]]:
    """Pull out table/figure captions as their own chunks (section 12) so
    they stay associated with their caption instead of being silently
    dropped or mashed into surrounding prose. This captures captions
    reliably; full table-cell structure extraction depends on the source
    format (PDF layout, HTML <table>, etc.) and should be handled in the
    format-specific parser upstream of this function where available —
    this is the format-agnostic fallback."""
    out = []
    for m in _TABLE_CAPTION_RE.finditer(text):
        out.append(("table_caption", m.group(1).strip()))
    for m in _FIGURE_CAPTION_RE.finditer(text):
        out.append(("figure_caption", m.group(1).strip()))
    return out


def chunk_document(document_id: str, source_id: str, full_text: str) -> list[dict]:
    """Top-level chunker: structure-aware split, then size-bound each
    section, then attach full ChunkMetadata to every piece. Returns a list
    of {"text": ..., "metadata": ChunkMetadata} dicts ready for embedding.
    """
    results = []
    order = 0
    for section_name, section_text in split_into_sections(full_text):
        if len(section_text) < MIN_CHUNK_CHARS and section_name not in (
            "abstract", "front_matter",
        ):
            # Too short to be useful alone; still keep it rather than drop
            # it, since short "conclusion" sections are common and valuable.
            pass
        for piece in _chunk_long_section(section_name, section_text):
            chunk_id = str(uuid.uuid4())
            results.append({
                "text": piece,
                "metadata": ChunkMetadata(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_id=source_id,
                    section=section_name,
                    order=order,
                ),
            })
            order += 1

    for kind, caption in extract_tables_and_figures(full_text):
        chunk_id = str(uuid.uuid4())
        results.append({
            "text": caption,
            "metadata": ChunkMetadata(
                chunk_id=chunk_id,
                document_id=document_id,
                source_id=source_id,
                section=kind,
                order=order,
            ),
        })
        order += 1

    return results


# ---------------------------------------------------------------------
# DISCOVERY / MANUAL-UPLOAD LIFECYCLE (sections 19-21, 44 of the original
# spec; state machine corrected per the completion follow-up section C —
# APPROVED is "authorized to ingest", not "retrievable". Only INDEXED is
# retrievable; see config.is_retrievable()).
#
# NOTE: as of the completion follow-up, this in-memory IngestionRecord is
# superseded by db.py's persisted version for anything production-facing
# (state must survive a restart — see section B/C of DELIVERABLES.md).
# Both call config.validate_transition() so the legality check itself has
# exactly one implementation whether the record lives in memory or in the
# database.
# ---------------------------------------------------------------------
@dataclass
class IngestionRecord:
    document_id: str
    state: IngestionState = IngestionState.DISCOVERED
    discovery_candidate_id: Optional[str] = None
    ingestion_method: str = "discovery"  # or "manual_upload"
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    history: list = field(default_factory=list)

    def transition(self, new_state: IngestionState, actor: str = "system"):
        config.validate_transition(self.state, new_state)
        self.history.append({
            "from": self.state.value,
            "to": new_state.value,
            "actor": actor,
            "at": datetime.now(timezone.utc).isoformat(),
        })
        self.state = new_state
        if new_state == IngestionState.APPROVED:
            self.approved_by = actor
            self.approved_at = datetime.now(timezone.utc).isoformat()

    def is_retrievable(self) -> bool:
        # This is the corrected semantics from the completion follow-up:
        # APPROVED only authorizes ingestion; retrievability requires the
        # document to have actually finished indexing.
        return config.is_retrievable(self.state)
