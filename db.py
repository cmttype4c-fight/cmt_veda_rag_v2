"""
db.py
-----
Persistence layer for RAG v2 (completion follow-up, section A/B/C).

Two backends behind one abstract interface (`DBBackend`):

  * `SqliteBackend`  — stdlib `sqlite3` only, no network, no server.
    This is what actually RUNS in this sandbox and in the test suite.
    It is a local/offline/CI substitute, not a second production
    database — production is PostgreSQL only (see DELIVERABLES.md).
  * `PostgresBackend` — real `psycopg2` code against the schema in
    db/schema.sql. Written to the same interface as SqliteBackend so
    every call site (ingestion_pipeline.py, discovery.py, retrieval.py,
    main.py) is backend-agnostic. **Not executable in this sandbox**:
    there is no `psycopg2` package and no Postgres server available here
    (no network to install/reach one). Treat this class as
    IMPLEMENTED BUT NOT INTEGRATED until it's been run against a real
    Postgres instance.

Both backends persist:
  documents (incl. source_id, all section-22 metadata, lifecycle state)
  chunks (incl. section, order, and — for SqliteBackend — an FTS5 mirror
    for lexical search, matching Postgres's tsvector/GIN approach)
  discovery_candidates
  ingestion_audit (every lifecycle transition, never deleted)
  answer_audit
  knowledge_versions

All lifecycle transitions go through `config.validate_transition()` before
being written, so illegal transitions (e.g. INDEXED -> APPROVED) are
rejected at the persistence layer itself, not just in application code
that a future call site could bypass.
"""

import dataclasses
import json
import sqlite3
import uuid
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

from config import IngestionState, validate_transition, SOURCE_ID_PREFIX, SOURCE_ID_DIGITS


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class DocumentRecord:
    document_id: str
    source_id: str
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
    source_tier: str = "unspecified"
    source_url: str = ""
    internal_storage_path: str = ""
    discovery_candidate_id: Optional[str] = None
    ingestion_method: str = "manual_upload"
    approval_status: IngestionState = IngestionState.DISCOVERED
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None
    knowledge_version: Optional[str] = None
    created_at: str = field(default_factory=_now)
    updated_at: str = field(default_factory=_now)
    # --- round 4: intake lifecycle split (register/approve/queue/process) ---
    source_format: str = ""              # 'pdf' | 'xml' | 'html' | 'text' | 'markdown'
    uploaded_by: Optional[str] = None    # direct_upload only
    uploaded_at: Optional[str] = None    # direct_upload only
    original_filename: str = ""          # direct_upload only
    extracted_text: str = ""             # populated at register_intake(); consumed by process_intake()
    retry_count: int = 0
    intake_batch_id: Optional[str] = None
    # --- scientific taxonomy + provenance (migration 0003; all additive) ---
    content_type: str = "research_paper"  # research_paper | clinical_trial | genetic_variant | ...
    source_mime_type: str = ""
    source_content_hash: str = ""
    source_document_ref: str = ""
    extraction_status: str = ""

    @property
    def source_type(self) -> str:
        """Brief's vocabulary ('discovery' / 'direct_upload') as a computed
        alias over the existing `ingestion_method` column ('discovery' /
        'manual_upload') — deliberately NOT a new stored column/enum. This
        means the physical schema/call sites for `ingestion_method` are
        untouched (zero risk to the round-3 tests that already pass), while
        every round-4 API response and filter uses the brief's own terms."""
        return "discovery" if self.ingestion_method == "discovery" else "direct_upload"


@dataclass
class ChunkRecord:
    chunk_id: str
    document_id: str
    section: str
    chunk_order: int
    content: str


@dataclass
class TurnRecord:
    """One question/answer pair within a conversation. `question` is
    exactly what the user typed; `rewritten_question` is what actually
    went to retrieval (identical to `question` if no rewrite was needed —
    see conversation.py). Conversation history is used ONLY to produce
    `rewritten_question` for a FUTURE turn's retrieval step — it is never
    fed into the generation prompt as fact. That's a deliberate design
    choice: letting raw prior answers leak into the generation context
    would let the model restate facts from earlier turns that weren't in
    THIS turn's retrieved evidence, which breaks the grounding guarantee
    the whole system depends on."""
    turn_id: str
    conversation_id: str
    turn_order: int
    question: str
    rewritten_question: str
    scope: str
    answer: str
    cited_source_ids: list = field(default_factory=list)
    insufficient_evidence: bool = False
    created_at: str = field(default_factory=_now)


class DuplicateDocumentError(Exception):
    pass


class DBBackend(ABC):
    """Backend-agnostic persistence interface. Every method here must be
    implemented identically in spirit by SqliteBackend and PostgresBackend
    so application code never branches on backend type."""

    # -- documents / lifecycle --
    @abstractmethod
    def create_document(self, doc: DocumentRecord) -> None: ...

    @abstractmethod
    def get_document(self, document_id: str) -> Optional[DocumentRecord]: ...

    @abstractmethod
    def find_duplicate(self, doi: str, pmid: str, title: str) -> Optional[DocumentRecord]: ...

    @abstractmethod
    def transition_document_state(self, document_id: str, new_state: IngestionState, actor: str) -> DocumentRecord: ...

    @abstractmethod
    def list_documents(self, state: Optional[IngestionState] = None) -> list[DocumentRecord]: ...

    # -- source IDs --
    @abstractmethod
    def next_source_id(self) -> str: ...

    # -- chunks --
    @abstractmethod
    def add_chunks(self, chunks: list[ChunkRecord]) -> None: ...

    @abstractmethod
    def get_chunks_for_document(self, document_id: str) -> list[ChunkRecord]: ...

    @abstractmethod
    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]: ...

    @abstractmethod
    def lexical_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Returns [(chunk_id, score), ...] restricted to chunks whose
        parent document is currently INDEXED (retrievable). This
        restriction MUST be enforced in the query itself (a JOIN/WHERE),
        not filtered afterward in Python, so removal is authoritative at
        the data layer."""
        ...

    @abstractmethod
    def get_active_chunk_ids(self) -> set:
        """All chunk_ids belonging to INDEXED documents. Used as the
        authoritative filter for the vector store's search results
        (section G: removal must actually work, even against a vector
        backend that can't delete rows in place)."""
        ...

    @abstractmethod
    def get_document_for_chunk(self, chunk_id: str) -> Optional[str]: ...
        # -- batch fetch (performance) --
    @abstractmethod
    def get_document_ids_for_chunks(self, chunk_ids: list) -> dict:
        """Returns {chunk_id: document_id} for whichever of the given
        ids exist. Empty input returns {} without a query."""
        ...

    @abstractmethod
    def get_chunks_by_ids(self, chunk_ids: list) -> dict:
        """Returns {chunk_id: ChunkRecord} for whichever of the given
        ids exist. Empty input returns {} without a query."""
        ...

    @abstractmethod
    def get_documents_by_ids(self, document_ids: list) -> dict:
        """Returns {document_id: DocumentRecord} for whichever of the
        given ids exist. Empty input returns {} without a query."""
        ...

    # -- discovery --
    @abstractmethod
    def create_discovery_candidate(self, discovery_candidate_id: str, proposed_title: str, proposed_source_url: str) -> None: ...

    @abstractmethod
    def get_discovery_candidate(self, discovery_candidate_id: str) -> Optional[dict]: ...

    @abstractmethod
    def record_discovery_review(self, discovery_candidate_id: str, reviewed_by: str, decision: str, linked_document_id: Optional[str]) -> None: ...

    # -- round 4: intake lifecycle (register/approve/queue/process) --
    @abstractmethod
    def set_intake_extras(self, document_id: str, **fields) -> None:
        """Updates whichever of source_format/uploaded_by/uploaded_at/
        original_filename/extracted_text/intake_batch_id are passed as
        kwargs. Exists so create_document()'s INSERT statement (and its
        fragile positional-tuple shape, unchanged from round 3) never had
        to be touched for round-4 fields — this is a follow-up UPDATE,
        not a change to document creation itself."""
        ...

    @abstractmethod
    def set_retry_count(self, document_id: str, retry_count: int) -> None: ...

    @abstractmethod
    def update_document_metadata(self, document_id: str, fields: dict) -> DocumentRecord:
        """PATCH /admin/intake/{id}. Caller (intake.py) is responsible for
        enforcing the pending_approval-only rule BEFORE calling this —
        this method itself just writes whichever of title/authors/journal/
        publication_date/doi/pmid/trial_id/source_tier/source_url are
        present in `fields`."""
        ...

    @abstractmethod
    def list_documents_page(self, state: Optional[IngestionState] = None,
                             source_type: Optional[str] = None,
                             limit: int = 50, offset: int = 0,
                             content_type: Optional[str] = None) -> list[DocumentRecord]:
        """GET /admin/intake listing — state/source_type filters + paging.
        `source_type` is 'discovery' | 'direct_upload' (the brief's terms);
        translated internally to the existing ingestion_method column."""
        ...

    @abstractmethod
    def create_intake_batch(self, batch_id: str, batch_type: str, created_by: str, item_count: int) -> None: ...

    @abstractmethod
    def record_batch_item(self, batch_id: str, document_id: Optional[str], status: str, error: Optional[str]) -> None:
        """Inserts one intake_batch_items row AND increments the parent
        batch's success_count/failure_count — atomic from the caller's
        point of view (one call, not two), so a crash between the two
        can't desync them."""
        ...

    @abstractmethod
    def get_batch(self, batch_id: str) -> Optional[dict]: ...

    @abstractmethod
    def list_batch_items(self, batch_id: str) -> list[dict]: ...

    @abstractmethod
    def create_processing_job(self, job_id: str, document_id: str, rq_job_id: str, attempt: int) -> None: ...

    @abstractmethod
    def update_processing_job(self, rq_job_id: str, status: str, error: Optional[str] = None) -> None: ...

    @abstractmethod
    def get_latest_processing_job(self, document_id: str) -> Optional[dict]: ...

    # -- audit --
    @abstractmethod
    def record_answer_audit(self, persona: str, query_scope: str, knowledge_version: str,
                             retrieved_chunk_ids: list, cited_source_ids: list,
                             insufficient_evidence: bool) -> None: ...

    @abstractmethod
    def audit_history(self, document_id: str) -> list[dict]: ...

    # -- conversations (multi-turn support + resume-later) --
    @abstractmethod
    def create_conversation(self, conversation_id: str, persona: str,
                             user_id: Optional[str] = None, title: Optional[str] = None) -> None:
        """Idempotent: if the conversation already exists, this is a
        no-op rather than an error, so main.py can call it unconditionally
        on every turn without tracking "is this a new conversation?". This
        means `title` (typically derived from the first question) is only
        ever actually written on the FIRST call for a given
        conversation_id — later calls with a different candidate title
        (the 2nd, 3rd... question) do not overwrite it, so a resumed
        conversation keeps a stable, recognizable title.

        `user_id` is a TRUSTED identifier Veda's backend passes after
        authenticating the real end user — treated exactly like
        `app_role` in config.resolve_persona(): this service does not
        itself authenticate end users (it's server-to-server only), so
        Veda is responsible for never forwarding a client-editable
        user_id. Ownership checks in main.py (a conversation only being
        listable/fetchable/deletable by the user_id that created it) are
        defense in depth on top of that trust boundary, not a substitute
        for it.
        """
        ...

    @abstractmethod
    def get_conversation(self, conversation_id: str) -> Optional[dict]: ...

    @abstractmethod
    def list_conversations_for_user(self, user_id: str, limit: int = 20) -> list[dict]:
        """Most-recently-active first — this is what powers a 'your
        previous conversations' / resume list in the client."""
        ...

    @abstractmethod
    def add_turn(self, turn: TurnRecord) -> None: ...

    @abstractmethod
    def get_turns(self, conversation_id: str, limit: Optional[int] = None) -> list[TurnRecord]:
        """Returns turns in order, oldest first. `limit` (when given)
        returns only the most recent `limit` turns — query rewriting only
        needs recent context, not the entire conversation history. Called
        with no limit, this is also what renders a full resumed
        transcript."""
        ...

    @abstractmethod
    def delete_conversation(self, conversation_id: str) -> None:
        """Patient/caregiver questions may themselves be sensitive even
        though they never touch the scientific corpus — this exists so a
        'clear my chat' action in Veda has something real to call.
        Ownership (does this conversation belong to the requesting
        user_id) is checked in main.py before this is called, not here —
        this method itself will delete whatever conversation_id it's
        given, same as the rest of this class stays a thin persistence
        layer rather than an authz layer."""
        ...

    @abstractmethod
    def delete_stale_conversations(self, older_than_days: int) -> int:
        """Real retention mechanism (not just a documented gap): deletes
        conversations whose last_activity_at is older than the given
        threshold. Returns the count deleted. Meant to be run on a
        schedule (see scripts/cleanup_stale_conversations.py) — this
        method does the deletion; the scheduling is an operational step
        that still needs to be wired up (cron/systemd timer), same
        caveat as every other 'script exists, run it yourself' item in
        DELIVERABLES.md."""
        ...


class SqliteBackend(DBBackend):
    """Real, executable persistence. Uses SQLite FTS5 as the lexical
    search index (functionally analogous to Postgres's tsvector/GIN —
    both are DB-native, persisted, indexed full-text search, as opposed
    to rescanning every chunk in Python on every query)."""

    def __init__(self, path: str):
        self.path = path
        # sqlite3.connect() does not create parent directories — a real
        # bug that would also hit production the first time someone
        # points RAG_SQLITE_PATH at a fresh subdirectory (e.g. exactly
        # what scripts/migrate_corpus.py does for a new versioned index).
        # Fixed here, not worked around in the caller.
        parent = Path(path).parent
        if parent and str(parent) not in (".", ""):
            parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._init_schema()

    def close(self):
        self._conn.close()

    def _init_schema(self):
        self._conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL UNIQUE,
            title TEXT, authors TEXT, journal TEXT, publication_date TEXT,
            doi TEXT, pmid TEXT, trial_id TEXT,
            cmt_subtypes TEXT, genes TEXT, study_type TEXT,
            source_tier TEXT DEFAULT 'unspecified', source_url TEXT,
            internal_storage_path TEXT,
            discovery_candidate_id TEXT,
            ingestion_method TEXT NOT NULL DEFAULT 'manual_upload',
            approval_status TEXT NOT NULL DEFAULT 'discovered',
            approved_by TEXT, approved_at TEXT,
            knowledge_version TEXT,
            created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
            source_format TEXT DEFAULT '',
            uploaded_by TEXT, uploaded_at TEXT,
            original_filename TEXT DEFAULT '',
            extracted_text TEXT DEFAULT '',
            retry_count INTEGER NOT NULL DEFAULT 0,
            intake_batch_id TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            section TEXT NOT NULL DEFAULT 'body',
            chunk_order INTEGER NOT NULL DEFAULT 0,
            content TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
            chunk_id UNINDEXED, content, tokenize = 'porter unicode61'
        );
        CREATE TABLE IF NOT EXISTS discovery_candidates (
            discovery_candidate_id TEXT PRIMARY KEY,
            proposed_title TEXT, proposed_source_url TEXT,
            discovered_at TEXT NOT NULL,
            reviewed_by TEXT, reviewed_at TEXT, review_decision TEXT,
            linked_document_id TEXT
        );
        CREATE TABLE IF NOT EXISTS ingestion_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            from_state TEXT, to_state TEXT NOT NULL,
            actor TEXT NOT NULL, occurred_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS answer_audit (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asked_at TEXT NOT NULL, persona TEXT NOT NULL, query_scope TEXT,
            knowledge_version TEXT, retrieved_chunk_ids TEXT, cited_source_ids TEXT,
            insufficient_evidence INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS source_id_counter (
            id INTEGER PRIMARY KEY CHECK (id = 1), value INTEGER NOT NULL
        );
        INSERT OR IGNORE INTO source_id_counter (id, value) VALUES (1, 0);
        CREATE TABLE IF NOT EXISTS conversations (
            conversation_id TEXT PRIMARY KEY,
            user_id TEXT,
            title TEXT,
            persona TEXT NOT NULL,
            created_at TEXT NOT NULL,
            last_activity_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversation_turns (
            turn_id TEXT PRIMARY KEY,
            conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id) ON DELETE CASCADE,
            turn_order INTEGER NOT NULL,
            question TEXT NOT NULL,
            rewritten_question TEXT NOT NULL,
            scope TEXT,
            answer TEXT NOT NULL,
            cited_source_ids TEXT,
            insufficient_evidence INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_turns_conversation_id ON conversation_turns (conversation_id, turn_order);
        CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations (user_id, last_activity_at);

        -- round 4: intake lifecycle (register/approve/queue/process split,
        -- bulk operations, RQ job tracking). Mirrors the Postgres migration
        -- in migrations/0002_intake_lifecycle.sql.
        CREATE TABLE IF NOT EXISTS intake_batches (
            batch_id TEXT PRIMARY KEY,
            batch_type TEXT NOT NULL,
            created_by TEXT NOT NULL,
            created_at TEXT NOT NULL,
            item_count INTEGER NOT NULL DEFAULT 0,
            success_count INTEGER NOT NULL DEFAULT 0,
            failure_count INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS intake_batch_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT NOT NULL REFERENCES intake_batches(batch_id) ON DELETE CASCADE,
            document_id TEXT,
            status TEXT NOT NULL,
            error TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_batch_items_batch_id ON intake_batch_items (batch_id);
        CREATE TABLE IF NOT EXISTS processing_jobs (
            job_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            rq_job_id TEXT NOT NULL UNIQUE,
            attempt INTEGER NOT NULL DEFAULT 1,
            status TEXT NOT NULL DEFAULT 'queued',
            enqueued_at TEXT NOT NULL,
            started_at TEXT, finished_at TEXT, error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_processing_jobs_document_id ON processing_jobs (document_id);
        """)
        self._conn.commit()
        # Defensive ALTER TABLE for a *pre-existing* database file created
        # before round 4 (CREATE TABLE IF NOT EXISTS above is a no-op
        # against an already-existing `documents` table, so an old DB file
        # would otherwise be missing these columns). Safe to run every
        # startup: SQLite has no "ADD COLUMN IF NOT EXISTS" portable across
        # all versions in use, so duplicate-column errors are caught and
        # ignored rather than relied on not to happen.
        for ddl in (
            "ALTER TABLE documents ADD COLUMN source_format TEXT DEFAULT ''",
            "ALTER TABLE documents ADD COLUMN uploaded_by TEXT",
            "ALTER TABLE documents ADD COLUMN uploaded_at TEXT",
            "ALTER TABLE documents ADD COLUMN original_filename TEXT DEFAULT ''",
            "ALTER TABLE documents ADD COLUMN extracted_text TEXT DEFAULT ''",
            "ALTER TABLE documents ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0",
            "ALTER TABLE documents ADD COLUMN intake_batch_id TEXT",
            "ALTER TABLE documents ADD COLUMN content_type TEXT NOT NULL DEFAULT 'research_paper'",
            "ALTER TABLE documents ADD COLUMN source_mime_type TEXT DEFAULT ''",
            "ALTER TABLE documents ADD COLUMN source_content_hash TEXT DEFAULT ''",
            "ALTER TABLE documents ADD COLUMN source_document_ref TEXT DEFAULT ''",
            "ALTER TABLE documents ADD COLUMN extraction_status TEXT DEFAULT ''",
        ):
            try:
                self._conn.execute(ddl)
                self._conn.commit()
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise

    # ---- documents ----
    def create_document(self, doc: DocumentRecord) -> None:
        with self._conn:
            self._conn.execute(
                """INSERT INTO documents
                (document_id, source_id, title, authors, journal, publication_date,
                 doi, pmid, trial_id, cmt_subtypes, genes, study_type, source_tier,
                 source_url, internal_storage_path, discovery_candidate_id,
                 ingestion_method, approval_status, approved_by, approved_at,
                 knowledge_version, created_at, updated_at,
                 source_format, uploaded_by, uploaded_at, original_filename,
                 extracted_text, retry_count, intake_batch_id,
                 content_type, source_mime_type, source_content_hash,
                 source_document_ref, extraction_status)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (doc.document_id, doc.source_id, doc.title, json.dumps(doc.authors),
                 doc.journal, doc.publication_date, doc.doi, doc.pmid, doc.trial_id,
                 json.dumps(doc.cmt_subtypes), json.dumps(doc.genes), doc.study_type,
                 doc.source_tier, doc.source_url, doc.internal_storage_path,
                 doc.discovery_candidate_id, doc.ingestion_method,
                 doc.approval_status.value if isinstance(doc.approval_status, IngestionState) else doc.approval_status,
                 doc.approved_by, doc.approved_at, doc.knowledge_version,
                 doc.created_at, doc.updated_at,
                 doc.source_format, doc.uploaded_by, doc.uploaded_at, doc.original_filename,
                 doc.extracted_text, doc.retry_count, doc.intake_batch_id,
                 doc.content_type or "research_paper", doc.source_mime_type,
                 doc.source_content_hash, doc.source_document_ref, doc.extraction_status),
            )

    def _row_to_doc(self, row) -> DocumentRecord:
        keys = row.keys()
        return DocumentRecord(
            document_id=row["document_id"], source_id=row["source_id"],
            title=row["title"] or "", authors=json.loads(row["authors"] or "[]"),
            journal=row["journal"] or "", publication_date=row["publication_date"],
            doi=row["doi"] or "", pmid=row["pmid"] or "", trial_id=row["trial_id"] or "",
            cmt_subtypes=json.loads(row["cmt_subtypes"] or "[]"),
            genes=json.loads(row["genes"] or "[]"), study_type=row["study_type"] or "",
            source_tier=row["source_tier"] or "unspecified", source_url=row["source_url"] or "",
            internal_storage_path=row["internal_storage_path"] or "",
            discovery_candidate_id=row["discovery_candidate_id"],
            ingestion_method=row["ingestion_method"],
            approval_status=IngestionState(row["approval_status"]),
            approved_by=row["approved_by"], approved_at=row["approved_at"],
            knowledge_version=row["knowledge_version"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            # round 4 columns: read defensively (`in keys()`) so this keeps
            # working against a pre-round-4 DB file mid-migration too.
            source_format=(row["source_format"] or "") if "source_format" in keys else "",
            uploaded_by=row["uploaded_by"] if "uploaded_by" in keys else None,
            uploaded_at=row["uploaded_at"] if "uploaded_at" in keys else None,
            original_filename=(row["original_filename"] or "") if "original_filename" in keys else "",
            extracted_text=(row["extracted_text"] or "") if "extracted_text" in keys else "",
            retry_count=(row["retry_count"] or 0) if "retry_count" in keys else 0,
            intake_batch_id=row["intake_batch_id"] if "intake_batch_id" in keys else None,
            content_type=(row["content_type"] or "research_paper") if "content_type" in keys else "research_paper",
            source_mime_type=(row["source_mime_type"] or "") if "source_mime_type" in keys else "",
            source_content_hash=(row["source_content_hash"] or "") if "source_content_hash" in keys else "",
            source_document_ref=(row["source_document_ref"] or "") if "source_document_ref" in keys else "",
            extraction_status=(row["extraction_status"] or "") if "extraction_status" in keys else "",
        )

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        row = self._conn.execute("SELECT * FROM documents WHERE document_id = ?", (document_id,)).fetchone()
        return self._row_to_doc(row) if row else None

    def find_duplicate(self, doi: str, pmid: str, title: str) -> Optional[DocumentRecord]:
        row = None
        if doi:
            row = self._conn.execute("SELECT * FROM documents WHERE doi = ? AND doi != ''", (doi,)).fetchone()
        if not row and pmid:
            row = self._conn.execute("SELECT * FROM documents WHERE pmid = ? AND pmid != ''", (pmid,)).fetchone()
        if not row and title:
            row = self._conn.execute(
                "SELECT * FROM documents WHERE lower(title) = lower(?) AND title != ''", (title,)
            ).fetchone()
        return self._row_to_doc(row) if row else None

    def transition_document_state(self, document_id: str, new_state: IngestionState, actor: str) -> DocumentRecord:
        doc = self.get_document(document_id)
        if doc is None:
            raise ValueError(f"No such document: {document_id}")
        validate_transition(doc.approval_status, new_state)
        now = _now()
        with self._conn:
            self._conn.execute(
                "UPDATE documents SET approval_status = ?, updated_at = ?,"
                " approved_by = COALESCE(?, approved_by), approved_at = COALESCE(?, approved_at)"
                " WHERE document_id = ?",
                (new_state.value, now,
                 actor if new_state == IngestionState.APPROVED else None,
                 now if new_state == IngestionState.APPROVED else None,
                 document_id),
            )
            self._conn.execute(
                "INSERT INTO ingestion_audit (document_id, from_state, to_state, actor, occurred_at)"
                " VALUES (?,?,?,?,?)",
                (document_id, doc.approval_status.value, new_state.value, actor, now),
            )
            if new_state == IngestionState.REMOVED:
                # Section G: physically remove from the lexical index
                # immediately. The vector-side authoritative filter is
                # get_active_chunk_ids(), which will now simply exclude
                # this document's chunks (its state is REMOVED, not
                # INDEXED) — no separate flag needed.
                chunk_ids = [r["chunk_id"] for r in self._conn.execute(
                    "SELECT chunk_id FROM chunks WHERE document_id = ?", (document_id,)
                )]
                for cid in chunk_ids:
                    self._conn.execute("DELETE FROM chunks_fts WHERE chunk_id = ?", (cid,))
        return self.get_document(document_id)

    def list_documents(self, state: Optional[IngestionState] = None) -> list[DocumentRecord]:
        if state is None:
            rows = self._conn.execute("SELECT * FROM documents").fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM documents WHERE approval_status = ?", (state.value,)
            ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    # ---- source ids ----
    def next_source_id(self) -> str:
        with self._conn:
            self._conn.execute("UPDATE source_id_counter SET value = value + 1 WHERE id = 1")
            n = self._conn.execute("SELECT value FROM source_id_counter WHERE id = 1").fetchone()["value"]
        return f"{SOURCE_ID_PREFIX}{n:0{SOURCE_ID_DIGITS}d}"

    # ---- chunks ----
    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        with self._conn:
            for c in chunks:
                self._conn.execute(
                    "INSERT INTO chunks (chunk_id, document_id, section, chunk_order, content)"
                    " VALUES (?,?,?,?,?)",
                    (c.chunk_id, c.document_id, c.section, c.chunk_order, c.content),
                )
                self._conn.execute(
                    "INSERT INTO chunks_fts (chunk_id, content) VALUES (?,?)",
                    (c.chunk_id, c.content),
                )

    def get_chunks_for_document(self, document_id: str) -> list[ChunkRecord]:
        rows = self._conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_order", (document_id,)
        ).fetchall()
        return [ChunkRecord(chunk_id=r["chunk_id"], document_id=r["document_id"],
                             section=r["section"], chunk_order=r["chunk_order"],
                             content=r["content"]) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        r = self._conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        if not r:
            return None
        return ChunkRecord(chunk_id=r["chunk_id"], document_id=r["document_id"],
                            section=r["section"], chunk_order=r["chunk_order"], content=r["content"])

    def lexical_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        # FTS5's bm25() gives lower = better; invert so higher = better,
        # matching the rest of the pipeline's "higher is better" convention.
        safe_query = self._sanitize_fts_query(query)
        if not safe_query:
            return []
        try:
            rows = self._conn.execute(
                """SELECT f.chunk_id AS chunk_id, bm25(chunks_fts) AS rank
                   FROM chunks_fts f
                   JOIN chunks c ON c.chunk_id = f.chunk_id
                   JOIN documents d ON d.document_id = c.document_id
                   WHERE chunks_fts MATCH ? AND d.approval_status = 'indexed'
                   ORDER BY rank LIMIT ?""",
                (safe_query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(r["chunk_id"], -float(r["rank"])) for r in rows]

    @staticmethod
    def _sanitize_fts_query(query: str) -> str:
        # Keep only alphanumerics as OR'd terms; avoids FTS5 syntax errors
        # from punctuation in natural-language questions.
        #
        # BUG FIX: this used to include stopwords ("what", "is", "the"...)
        # as OR terms. Postgres's equivalent (ts_rank_cd/plainto_tsquery
        # with the 'english' text search configuration) strips stopwords
        # automatically as part of its built-in linguistic normalization —
        # this SQLite substitute never had that, so a query like "What is
        # Charcot-Marie-Tooth disease?" matched on "what"/"is" too,
        # diluting BM25's IDF-weighted scoring (common words appearing in
        # nearly every chunk contribute near-zero signal but still count
        # as "matched terms" toward the query). Found while testing
        # scripts/diagnose_retrieval.py against this dev substitute — it's
        # specific to the SQLite path, not a Postgres issue, since
        # Postgres already handles this. Reuses embeddings.py's stopword
        # list rather than duplicating one.
        import re
        from embeddings import _STOPWORDS
        terms = [w for w in re.findall(r"[A-Za-z0-9]+", query.lower()) if w not in _STOPWORDS]
        return " OR ".join(terms)

    def get_active_chunk_ids(self) -> set:
        rows = self._conn.execute(
            "SELECT c.chunk_id FROM chunks c JOIN documents d ON d.document_id = c.document_id"
            " WHERE d.approval_status = 'indexed'"
        ).fetchall()
        return {r["chunk_id"] for r in rows}

    def get_document_for_chunk(self, chunk_id: str) -> Optional[str]:
        row = self._conn.execute("SELECT document_id FROM chunks WHERE chunk_id = ?", (chunk_id,)).fetchone()
        return row["document_id"] if row else None
         # ---- batch fetch (performance) ----
    def get_document_ids_for_chunks(self, chunk_ids: list) -> dict:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT chunk_id, document_id FROM chunks WHERE chunk_id IN ({placeholders})",
            list(chunk_ids),
        ).fetchall()
        return {r["chunk_id"]: r["document_id"] for r in rows}

    def get_chunks_by_ids(self, chunk_ids: list) -> dict:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" * len(chunk_ids))
        rows = self._conn.execute(
            f"SELECT * FROM chunks WHERE chunk_id IN ({placeholders})",
            list(chunk_ids),
        ).fetchall()
        return {
            r["chunk_id"]: ChunkRecord(
                chunk_id=r["chunk_id"],
                document_id=r["document_id"],
                section=r["section"],
                chunk_order=r["chunk_order"],
                content=r["content"],
            )
            for r in rows
        }

    def get_documents_by_ids(self, document_ids: list) -> dict:
        if not document_ids:
            return {}
        placeholders = ",".join("?" * len(document_ids))
        rows = self._conn.execute(
            f"SELECT * FROM documents WHERE document_id IN ({placeholders})",
            list(document_ids),
        ).fetchall()
        return {r["document_id"]: self._row_to_doc(r) for r in rows}

    # ---- discovery ----
    def create_discovery_candidate(self, discovery_candidate_id: str, proposed_title: str, proposed_source_url: str) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO discovery_candidates (discovery_candidate_id, proposed_title,"
                " proposed_source_url, discovered_at) VALUES (?,?,?,?)",
                (discovery_candidate_id, proposed_title, proposed_source_url, _now()),
            )

    def get_discovery_candidate(self, discovery_candidate_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM discovery_candidates WHERE discovery_candidate_id = ?",
            (discovery_candidate_id,),
        ).fetchone()
        return dict(row) if row else None

    def record_discovery_review(self, discovery_candidate_id: str, reviewed_by: str, decision: str, linked_document_id: Optional[str]) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE discovery_candidates SET reviewed_by=?, reviewed_at=?, review_decision=?,"
                " linked_document_id=? WHERE discovery_candidate_id=?",
                (reviewed_by, _now(), decision, linked_document_id, discovery_candidate_id),
            )

    # ---- round 4: intake lifecycle ----
    _INTAKE_EXTRA_COLUMNS = {
        "source_format", "uploaded_by", "uploaded_at", "original_filename",
        "extracted_text", "intake_batch_id",
        "content_type", "source_mime_type", "source_content_hash",
        "source_document_ref", "extraction_status",
    }

    def set_intake_extras(self, document_id: str, **fields) -> None:
        cols = {k: v for k, v in fields.items() if k in self._INTAKE_EXTRA_COLUMNS}
        if not cols:
            return
        set_clause = ", ".join(f"{k} = ?" for k in cols)
        with self._conn:
            self._conn.execute(
                f"UPDATE documents SET {set_clause}, updated_at = ? WHERE document_id = ?",
                (*cols.values(), _now(), document_id),
            )

    def set_retry_count(self, document_id: str, retry_count: int) -> None:
        with self._conn:
            self._conn.execute(
                "UPDATE documents SET retry_count = ?, updated_at = ? WHERE document_id = ?",
                (retry_count, _now(), document_id),
            )

    _METADATA_PATCH_COLUMNS = {
        "title", "authors", "journal", "publication_date", "doi", "pmid",
        "trial_id", "source_tier", "source_url",
    }

    def update_document_metadata(self, document_id: str, fields: dict) -> DocumentRecord:
        cols = {k: v for k, v in fields.items() if k in self._METADATA_PATCH_COLUMNS}
        if cols:
            # authors/cmt_subtypes/genes are stored as JSON text, same as create_document
            if "authors" in cols and isinstance(cols["authors"], list):
                cols["authors"] = json.dumps(cols["authors"])
            set_clause = ", ".join(f"{k} = ?" for k in cols)
            with self._conn:
                self._conn.execute(
                    f"UPDATE documents SET {set_clause}, updated_at = ? WHERE document_id = ?",
                    (*cols.values(), _now(), document_id),
                )
        doc = self.get_document(document_id)
        if doc is None:
            raise ValueError(f"No such document: {document_id}")
        return doc

    def list_documents_page(self, state: Optional[IngestionState] = None,
                             source_type: Optional[str] = None,
                             limit: int = 50, offset: int = 0,
                             content_type: Optional[str] = None) -> list[DocumentRecord]:
        clauses, params = [], []
        if state is not None:
            clauses.append("approval_status = ?")
            params.append(state.value)
        if source_type is not None:
            # 'direct_upload' (brief's term) -> everything NOT 'discovery' in
            # the existing ingestion_method column; 'discovery' passes through.
            if source_type == "discovery":
                clauses.append("ingestion_method = 'discovery'")
            else:
                clauses.append("ingestion_method != 'discovery'")
        if content_type is not None:
            clauses.append("content_type = ?")
            params.append(content_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = self._conn.execute(
            f"SELECT * FROM documents {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (*params, limit, offset),
        ).fetchall()
        return [self._row_to_doc(r) for r in rows]

    def create_intake_batch(self, batch_id: str, batch_type: str, created_by: str, item_count: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO intake_batches (batch_id, batch_type, created_by, created_at, item_count)"
                " VALUES (?,?,?,?,?)",
                (batch_id, batch_type, created_by, _now(), item_count),
            )

    def record_batch_item(self, batch_id: str, document_id: Optional[str], status: str, error: Optional[str]) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO intake_batch_items (batch_id, document_id, status, error, created_at)"
                " VALUES (?,?,?,?,?)",
                (batch_id, document_id, status, error, _now()),
            )
            col = "success_count" if status == "success" else "failure_count"
            self._conn.execute(
                f"UPDATE intake_batches SET {col} = {col} + 1 WHERE batch_id = ?", (batch_id,)
            )

    def get_batch(self, batch_id: str) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM intake_batches WHERE batch_id = ?", (batch_id,)).fetchone()
        return dict(row) if row else None

    def list_batch_items(self, batch_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM intake_batch_items WHERE batch_id = ? ORDER BY id", (batch_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def create_processing_job(self, job_id: str, document_id: str, rq_job_id: str, attempt: int) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO processing_jobs (job_id, document_id, rq_job_id, attempt, status, enqueued_at)"
                " VALUES (?,?,?,?,'queued',?)",
                (job_id, document_id, rq_job_id, attempt, _now()),
            )

    def update_processing_job(self, rq_job_id: str, status: str, error: Optional[str] = None) -> None:
        now = _now()
        with self._conn:
            if status == "started":
                self._conn.execute(
                    "UPDATE processing_jobs SET status=?, started_at=? WHERE rq_job_id=?",
                    (status, now, rq_job_id),
                )
            elif status in ("finished", "failed"):
                self._conn.execute(
                    "UPDATE processing_jobs SET status=?, finished_at=?, error=? WHERE rq_job_id=?",
                    (status, now, error, rq_job_id),
                )
            else:
                self._conn.execute(
                    "UPDATE processing_jobs SET status=?, error=? WHERE rq_job_id=?",
                    (status, error, rq_job_id),
                )

    def get_latest_processing_job(self, document_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM processing_jobs WHERE document_id = ? ORDER BY enqueued_at DESC LIMIT 1",
            (document_id,),
        ).fetchone()
        return dict(row) if row else None

    # ---- audit ----
    def record_answer_audit(self, persona, query_scope, knowledge_version,
                             retrieved_chunk_ids, cited_source_ids, insufficient_evidence) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO answer_audit (asked_at, persona, query_scope, knowledge_version,"
                " retrieved_chunk_ids, cited_source_ids, insufficient_evidence) VALUES (?,?,?,?,?,?,?)",
                (_now(), persona, query_scope, knowledge_version,
                 json.dumps(retrieved_chunk_ids), json.dumps(cited_source_ids),
                 1 if insufficient_evidence else 0),
            )

    def audit_history(self, document_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM ingestion_audit WHERE document_id = ? ORDER BY id", (document_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    # ---- conversations ----
    def create_conversation(self, conversation_id: str, persona: str,
                             user_id: Optional[str] = None, title: Optional[str] = None) -> None:
        now = _now()
        with self._conn:
            # INSERT OR IGNORE: if the row already exists, this whole
            # statement (including `title`) is a no-op — see the
            # docstring on the abstract method for why that's load-bearing
            # (a resumed conversation's title must not drift with every
            # new question asked in it).
            self._conn.execute(
                "INSERT OR IGNORE INTO conversations"
                " (conversation_id, user_id, title, persona, created_at, last_activity_at)"
                " VALUES (?,?,?,?,?,?)",
                (conversation_id, user_id, title, persona, now, now),
            )

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        row = self._conn.execute(
            "SELECT * FROM conversations WHERE conversation_id = ?", (conversation_id,)
        ).fetchone()
        return dict(row) if row else None

    def list_conversations_for_user(self, user_id: str, limit: int = 20) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM conversations WHERE user_id = ? ORDER BY last_activity_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def add_turn(self, turn: TurnRecord) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO conversation_turns (turn_id, conversation_id, turn_order, question,"
                " rewritten_question, scope, answer, cited_source_ids, insufficient_evidence, created_at)"
                " VALUES (?,?,?,?,?,?,?,?,?,?)",
                (turn.turn_id, turn.conversation_id, turn.turn_order, turn.question,
                 turn.rewritten_question, turn.scope, turn.answer,
                 json.dumps(turn.cited_source_ids), 1 if turn.insufficient_evidence else 0,
                 turn.created_at),
            )
            self._conn.execute(
                "UPDATE conversations SET last_activity_at = ? WHERE conversation_id = ?",
                (turn.created_at, turn.conversation_id),
            )

    def get_turns(self, conversation_id: str, limit: Optional[int] = None) -> list[TurnRecord]:
        query = "SELECT * FROM conversation_turns WHERE conversation_id = ? ORDER BY turn_order"
        rows = self._conn.execute(query, (conversation_id,)).fetchall()
        turns = [
            TurnRecord(
                turn_id=r["turn_id"], conversation_id=r["conversation_id"],
                turn_order=r["turn_order"], question=r["question"],
                rewritten_question=r["rewritten_question"], scope=r["scope"] or "",
                answer=r["answer"], cited_source_ids=json.loads(r["cited_source_ids"] or "[]"),
                insufficient_evidence=bool(r["insufficient_evidence"]), created_at=r["created_at"],
            )
            for r in rows
        ]
        if limit is not None and len(turns) > limit:
            turns = turns[-limit:]
        return turns

    def delete_conversation(self, conversation_id: str) -> None:
        with self._conn:
            self._conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (conversation_id,))
            # conversation_turns rows cascade via ON DELETE CASCADE, but
            # SQLite only enforces that with foreign_keys=ON (set at
            # connection time in __init__) — kept explicit here anyway so
            # this method is correct even if that pragma setting changes.
            self._conn.execute("DELETE FROM conversation_turns WHERE conversation_id = ?", (conversation_id,))

    def delete_stale_conversations(self, older_than_days: int) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        with self._conn:
            stale_ids = [r["conversation_id"] for r in self._conn.execute(
                "SELECT conversation_id FROM conversations WHERE last_activity_at < ?", (cutoff,)
            )]
            for cid in stale_ids:
                self._conn.execute("DELETE FROM conversations WHERE conversation_id = ?", (cid,))
                self._conn.execute("DELETE FROM conversation_turns WHERE conversation_id = ?", (cid,))
        return len(stale_ids)


class PostgresBackend(DBBackend):
    """Real production backend against db/schema.sql. Requires `psycopg2`
    and a reachable Postgres instance — NEITHER is available in this
    sandbox (no network, package not installed, no server). This class is
    written to the same interface as SqliteBackend and follows the same
    SQL shape as schema.sql, but it has not been run anywhere. Status:
    IMPLEMENTED BUT NOT INTEGRATED — see DELIVERABLES.md section A.
    """

    def __init__(self, dsn: str):
        try:
            import psycopg2  # noqa: F401
            import psycopg2.extras  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "PostgresBackend requires psycopg2, which is not installed "
                "in this environment. Install it in your deployment venv "
                "(`pip install psycopg2-binary`) and run against a real "
                "Postgres instance before relying on this backend."
            ) from e
        import psycopg2
        import psycopg2.extras
        self._psycopg2 = psycopg2
        self._conn = psycopg2.connect(dsn)
        self._conn.autocommit = False

    def close(self):
        self._conn.close()

    @contextmanager
    def _cursor(self):
        cur = self._conn.cursor(cursor_factory=self._psycopg2.extras.RealDictCursor)
        try:
            yield cur
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise
        finally:
            cur.close()

    # Every DocumentRecord column that is physically stored, in INSERT order.
    # LIVE-BUG FIX: this INSERT used to list only the 21 round-3 columns, so
    # on PostgreSQL register_intake() silently dropped extracted_text,
    # source_format, uploaded_*, original_filename, retry_count and
    # intake_batch_id -- the document registered fine and then failed at
    # processing with "No extracted text stored". SQLite persisted them,
    # which is why the SQLite-backed suites never saw it. Columns 22+ require
    # migrations/0002 and 0003 to have been applied (deploy/migrate.sh does).
    _DOC_INSERT_COLUMNS = (
        "document_id", "source_id", "title", "authors", "journal", "publication_date",
        "doi", "pmid", "trial_id", "cmt_subtypes", "genes", "study_type", "source_tier",
        "source_url", "internal_storage_path", "discovery_candidate_id",
        "ingestion_method", "approval_status", "approved_by", "approved_at",
        "knowledge_version",
        "source_format", "uploaded_by", "uploaded_at", "original_filename",
        "extracted_text", "retry_count", "intake_batch_id",
        "content_type", "source_mime_type", "source_content_hash",
        "source_document_ref", "extraction_status",
    )

    @staticmethod
    def _doc_insert_values(doc: DocumentRecord) -> tuple:
        status = doc.approval_status.value if isinstance(doc.approval_status, IngestionState) else doc.approval_status
        return (
            doc.document_id, doc.source_id, doc.title, doc.authors, doc.journal,
            doc.publication_date, doc.doi, doc.pmid, doc.trial_id, doc.cmt_subtypes,
            doc.genes, doc.study_type, doc.source_tier, doc.source_url,
            doc.internal_storage_path, doc.discovery_candidate_id, doc.ingestion_method,
            status, doc.approved_by, doc.approved_at, doc.knowledge_version,
            doc.source_format, doc.uploaded_by, doc.uploaded_at, doc.original_filename,
            doc.extracted_text, doc.retry_count, doc.intake_batch_id,
            doc.content_type or "research_paper", doc.source_mime_type,
            doc.source_content_hash, doc.source_document_ref, doc.extraction_status,
        )

    def create_document(self, doc: DocumentRecord) -> None:
        cols = self._DOC_INSERT_COLUMNS
        sql = (
            f"INSERT INTO cmt_veda_rag.documents ({', '.join(cols)}) "
            f"VALUES ({', '.join(['%s'] * len(cols))})"
        )
        with self._cursor() as cur:
            cur.execute(sql, self._doc_insert_values(doc))

    @staticmethod
    def _row_to_document(row) -> DocumentRecord:
        """RealDictCursor row -> DocumentRecord. Unknown columns (e.g. a
        newer migration than this code) are ignored instead of raising
        TypeError, and the approval_status enum is converted."""
        data = dict(row)
        data["approval_status"] = IngestionState(data["approval_status"])
        known = {f.name for f in dataclasses.fields(DocumentRecord)}
        return DocumentRecord(**{k: v for k, v in data.items() if k in known})

    def get_document(self, document_id: str) -> Optional[DocumentRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM cmt_veda_rag.documents WHERE document_id = %s", (document_id,))
            row = cur.fetchone()
        if not row:
            return None
        return self._row_to_document(row)

    def find_duplicate(self, doi: str, pmid: str, title: str) -> Optional[DocumentRecord]:
        with self._cursor() as cur:
            if doi:
                cur.execute("SELECT * FROM cmt_veda_rag.documents WHERE doi = %s", (doi,))
                row = cur.fetchone()
                if row:
                    return self._row_to_document(row)
            if pmid:
                cur.execute("SELECT * FROM cmt_veda_rag.documents WHERE pmid = %s", (pmid,))
                row = cur.fetchone()
                if row:
                    return self._row_to_document(row)
        return None

    def transition_document_state(self, document_id: str, new_state: IngestionState, actor: str) -> DocumentRecord:
        doc = self.get_document(document_id)
        if doc is None:
            raise ValueError(f"No such document: {document_id}")
        validate_transition(doc.approval_status, new_state)
        with self._cursor() as cur:
            cur.execute(
                "UPDATE cmt_veda_rag.documents SET approval_status=%s, updated_at=now(),"
                " approved_by = COALESCE(%s, approved_by), approved_at = COALESCE(%s, approved_at)"
                " WHERE document_id=%s",
                (new_state.value,
                 actor if new_state == IngestionState.APPROVED else None,
                 _now() if new_state == IngestionState.APPROVED else None,
                 document_id),
            )
            cur.execute(
                "INSERT INTO cmt_veda_rag.ingestion_audit (document_id, from_state, to_state, actor)"
                " VALUES (%s,%s,%s,%s)",
                (document_id, doc.approval_status.value, new_state.value, actor),
            )
        return self.get_document(document_id)

    def list_documents(self, state: Optional[IngestionState] = None) -> list[DocumentRecord]:
        with self._cursor() as cur:
            if state is None:
                cur.execute("SELECT * FROM cmt_veda_rag.documents")
            else:
                cur.execute("SELECT * FROM cmt_veda_rag.documents WHERE approval_status=%s", (state.value,))
            rows = cur.fetchall()
        out = []
        for row in rows:
            out.append(self._row_to_document(row))
        return out

    def next_source_id(self) -> str:
        # Atomic via a real Postgres sequence — no MAX()+1 race condition
        # (fixed after review; the original draft here derived the next ID
        # from MAX(source_id), which is unsafe under concurrent inserts).
        with self._cursor() as cur:
            cur.execute("SELECT nextval('cmt_veda_rag.source_id_seq') AS n")
            n = cur.fetchone()["n"]
        return f"{SOURCE_ID_PREFIX}{n:0{SOURCE_ID_DIGITS}d}"

    def add_chunks(self, chunks: list[ChunkRecord]) -> None:
        with self._cursor() as cur:
            for c in chunks:
                cur.execute(
                    "INSERT INTO cmt_veda_rag.chunks (chunk_id, document_id, section, chunk_order, content)"
                    " VALUES (%s,%s,%s,%s,%s)",
                    (c.chunk_id, c.document_id, c.section, c.chunk_order, c.content),
                )

    def get_chunks_for_document(self, document_id: str) -> list[ChunkRecord]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.chunks WHERE document_id=%s ORDER BY chunk_order",
                (document_id,),
            )
            rows = cur.fetchall()
        return [ChunkRecord(chunk_id=r["chunk_id"], document_id=r["document_id"],
                             section=r["section"], chunk_order=r["chunk_order"],
                             content=r["content"]) for r in rows]

    def get_chunk(self, chunk_id: str) -> Optional[ChunkRecord]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM cmt_veda_rag.chunks WHERE chunk_id=%s", (chunk_id,))
            r = cur.fetchone()
        if not r:
            return None
        return ChunkRecord(chunk_id=r["chunk_id"], document_id=r["document_id"],
                            section=r["section"], chunk_order=r["chunk_order"], content=r["content"])

    def lexical_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        with self._cursor() as cur:
            cur.execute(
                """SELECT c.chunk_id AS chunk_id,
                          ts_rank_cd(c.content_tsv, plainto_tsquery('english', %s)) AS score
                   FROM cmt_veda_rag.chunks c
                   JOIN cmt_veda_rag.documents d ON d.document_id = c.document_id
                   WHERE d.approval_status = 'indexed'
                     AND c.content_tsv @@ plainto_tsquery('english', %s)
                   ORDER BY score DESC LIMIT %s""",
                (query, query, limit),
            )
            rows = cur.fetchall()
        return [(r["chunk_id"], float(r["score"])) for r in rows]

    def get_active_chunk_ids(self) -> set:
        with self._cursor() as cur:
            cur.execute(
                "SELECT c.chunk_id FROM cmt_veda_rag.chunks c"
                " JOIN cmt_veda_rag.documents d ON d.document_id=c.document_id"
                " WHERE d.approval_status = 'indexed'"
            )
            rows = cur.fetchall()
        return {r["chunk_id"] for r in rows}

    def get_document_for_chunk(self, chunk_id: str) -> Optional[str]:
        with self._cursor() as cur:
            cur.execute("SELECT document_id FROM cmt_veda_rag.chunks WHERE chunk_id=%s", (chunk_id,))
            row = cur.fetchone()
        return row["document_id"] if row else None
         # ---- batch fetch (performance) ----
    def get_document_ids_for_chunks(self, chunk_ids: list) -> dict:
        if not chunk_ids:
            return {}
        with self._cursor() as cur:
            cur.execute(
                "SELECT chunk_id, document_id FROM cmt_veda_rag.chunks "
                "WHERE chunk_id = ANY(%s::uuid[])",
                ([str(x) for x in chunk_ids],),
            )
            rows = cur.fetchall()
        return {str(r["chunk_id"]): str(r["document_id"]) for r in rows}

    def get_chunks_by_ids(self, chunk_ids: list) -> dict:
        if not chunk_ids:
            return {}
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.chunks "
                "WHERE chunk_id = ANY(%s::uuid[])",
                ([str(x) for x in chunk_ids],),
            )
            rows = cur.fetchall()
        return {
            str(r["chunk_id"]): ChunkRecord(
                chunk_id=str(r["chunk_id"]),
                document_id=str(r["document_id"]),
                section=r["section"],
                chunk_order=r["chunk_order"],
                content=r["content"],
            )
            for r in rows
        }

    def get_documents_by_ids(self, document_ids: list) -> dict:
        if not document_ids:
            return {}
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.documents "
                "WHERE document_id = ANY(%s::uuid[])",
                ([str(x) for x in document_ids],),
            )
            rows = cur.fetchall()
        out = {}
        for row in rows:
            doc = self._row_to_document(row)
            out[str(doc.document_id)] = doc
        return out

    def create_discovery_candidate(self, discovery_candidate_id, proposed_title, proposed_source_url) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO cmt_veda_rag.discovery_candidates"
                " (discovery_candidate_id, proposed_title, proposed_source_url)"
                " VALUES (%s,%s,%s)",
                (discovery_candidate_id, proposed_title, proposed_source_url),
            )

    def get_discovery_candidate(self, discovery_candidate_id: str) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.discovery_candidates WHERE discovery_candidate_id=%s",
                (discovery_candidate_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def record_discovery_review(self, discovery_candidate_id, reviewed_by, decision, linked_document_id) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE cmt_veda_rag.discovery_candidates SET reviewed_by=%s, reviewed_at=now(),"
                " review_decision=%s, linked_document_id=%s WHERE discovery_candidate_id=%s",
                (reviewed_by, decision, linked_document_id, discovery_candidate_id),
            )

    # ---- round 4: intake lifecycle ----
    # Written to the same interface/behavior as SqliteBackend's methods
    # above. NOT executed against a real server in this sandbox (no
    # psycopg2/network — same constraint as every other PostgresBackend
    # method; see the class docstring and DELIVERABLES_ROUND4.md). The SQL
    # itself (migrations/0002_intake_lifecycle.sql) WAS verified live via
    # `psql` directly against a local Postgres 16 instance — see that doc
    # for what "verified" means here vs. this Python code path.
    _INTAKE_EXTRA_COLUMNS = {
        "source_format", "uploaded_by", "uploaded_at", "original_filename",
        "extracted_text", "intake_batch_id",
        "content_type", "source_mime_type", "source_content_hash",
        "source_document_ref", "extraction_status",
    }

    def set_intake_extras(self, document_id: str, **fields) -> None:
        cols = {k: v for k, v in fields.items() if k in self._INTAKE_EXTRA_COLUMNS}
        if not cols:
            return
        set_clause = ", ".join(f"{k} = %s" for k in cols)
        with self._cursor() as cur:
            cur.execute(
                f"UPDATE cmt_veda_rag.documents SET {set_clause}, updated_at = now() WHERE document_id = %s",
                (*cols.values(), document_id),
            )

    def set_retry_count(self, document_id: str, retry_count: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                "UPDATE cmt_veda_rag.documents SET retry_count = %s, updated_at = now() WHERE document_id = %s",
                (retry_count, document_id),
            )

    _METADATA_PATCH_COLUMNS = {
        "title", "authors", "journal", "publication_date", "doi", "pmid",
        "trial_id", "source_tier", "source_url",
    }

    def update_document_metadata(self, document_id: str, fields: dict) -> DocumentRecord:
        cols = {k: v for k, v in fields.items() if k in self._METADATA_PATCH_COLUMNS}
        if cols:
            set_clause = ", ".join(f"{k} = %s" for k in cols)
            with self._cursor() as cur:
                cur.execute(
                    f"UPDATE cmt_veda_rag.documents SET {set_clause}, updated_at = now() WHERE document_id = %s",
                    (*cols.values(), document_id),
                )
        doc = self.get_document(document_id)
        if doc is None:
            raise ValueError(f"No such document: {document_id}")
        return doc

    def list_documents_page(self, state: Optional[IngestionState] = None,
                             source_type: Optional[str] = None,
                             limit: int = 50, offset: int = 0,
                             content_type: Optional[str] = None) -> list[DocumentRecord]:
        clauses, params = [], []
        if state is not None:
            clauses.append("approval_status = %s")
            params.append(state.value)
        if source_type is not None:
            if source_type == "discovery":
                clauses.append("ingestion_method = 'discovery'")
            else:
                clauses.append("ingestion_method != 'discovery'")
        if content_type is not None:
            clauses.append("content_type = %s")
            params.append(content_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._cursor() as cur:
            cur.execute(
                f"SELECT * FROM cmt_veda_rag.documents {where} ORDER BY created_at DESC LIMIT %s OFFSET %s",
                (*params, limit, offset),
            )
            rows = cur.fetchall()
        out = []
        for row in rows:
            out.append(self._row_to_document(row))
        return out

    def create_intake_batch(self, batch_id: str, batch_type: str, created_by: str, item_count: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO cmt_veda_rag.intake_batches (batch_id, batch_type, created_by, item_count)"
                " VALUES (%s,%s,%s,%s)",
                (batch_id, batch_type, created_by, item_count),
            )

    def record_batch_item(self, batch_id: str, document_id: Optional[str], status: str, error: Optional[str]) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO cmt_veda_rag.intake_batch_items (batch_id, document_id, status, error)"
                " VALUES (%s,%s,%s,%s)",
                (batch_id, document_id, status, error),
            )
            col = "success_count" if status == "success" else "failure_count"
            cur.execute(
                f"UPDATE cmt_veda_rag.intake_batches SET {col} = {col} + 1 WHERE batch_id = %s", (batch_id,)
            )

    def get_batch(self, batch_id: str) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute("SELECT * FROM cmt_veda_rag.intake_batches WHERE batch_id = %s", (batch_id,))
            row = cur.fetchone()
        return dict(row) if row else None

    def list_batch_items(self, batch_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.intake_batch_items WHERE batch_id = %s ORDER BY id", (batch_id,)
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def create_processing_job(self, job_id: str, document_id: str, rq_job_id: str, attempt: int) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO cmt_veda_rag.processing_jobs (job_id, document_id, rq_job_id, attempt, status)"
                " VALUES (%s,%s,%s,%s,'queued')",
                (job_id, document_id, rq_job_id, attempt),
            )

    def update_processing_job(self, rq_job_id: str, status: str, error: Optional[str] = None) -> None:
        with self._cursor() as cur:
            if status == "started":
                cur.execute(
                    "UPDATE cmt_veda_rag.processing_jobs SET status=%s, started_at=now() WHERE rq_job_id=%s",
                    (status, rq_job_id),
                )
            elif status in ("finished", "failed"):
                cur.execute(
                    "UPDATE cmt_veda_rag.processing_jobs SET status=%s, finished_at=now(), error=%s WHERE rq_job_id=%s",
                    (status, error, rq_job_id),
                )
            else:
                cur.execute(
                    "UPDATE cmt_veda_rag.processing_jobs SET status=%s, error=%s WHERE rq_job_id=%s",
                    (status, error, rq_job_id),
                )

    def get_latest_processing_job(self, document_id: str) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.processing_jobs WHERE document_id = %s"
                " ORDER BY enqueued_at DESC LIMIT 1",
                (document_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def record_answer_audit(self, persona, query_scope, knowledge_version,
                            retrieved_chunk_ids, cited_source_ids, insufficient_evidence) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO cmt_veda_rag.answer_audit (persona, query_scope, knowledge_version,"
                " retrieved_chunk_ids, cited_source_ids, insufficient_evidence)"
                " VALUES (%s,%s,%s,%s::uuid[],%s,%s)",
                (persona, query_scope, knowledge_version,
                 [str(x) for x in retrieved_chunk_ids],
                 cited_source_ids, insufficient_evidence),
            )

    def audit_history(self, document_id: str) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.ingestion_audit WHERE document_id=%s ORDER BY id",
                (document_id,),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    # ---- conversations ----
    # Written to the same interface as SqliteBackend's tested version
    # above; not itself executed anywhere (no Postgres in this sandbox —
    # same caveat as every other PostgresBackend method in this file).
    def create_conversation(self, conversation_id: str, persona: str,
                             user_id: Optional[str] = None, title: Optional[str] = None) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO cmt_veda_rag.conversations (conversation_id, user_id, title, persona)"
                " VALUES (%s,%s,%s,%s) ON CONFLICT (conversation_id) DO NOTHING",
                (conversation_id, user_id, title, persona),
            )

    def get_conversation(self, conversation_id: str) -> Optional[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.conversations WHERE conversation_id=%s",
                (conversation_id,),
            )
            row = cur.fetchone()
        return dict(row) if row else None

    def list_conversations_for_user(self, user_id: str, limit: int = 20) -> list[dict]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.conversations WHERE user_id=%s"
                " ORDER BY last_activity_at DESC LIMIT %s",
                (user_id, limit),
            )
            rows = cur.fetchall()
        return [dict(r) for r in rows]

    def add_turn(self, turn: TurnRecord) -> None:
        with self._cursor() as cur:
            cur.execute(
                "INSERT INTO cmt_veda_rag.conversation_turns (turn_id, conversation_id, turn_order,"
                " question, rewritten_question, scope, answer, cited_source_ids, insufficient_evidence)"
                " VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                (turn.turn_id, turn.conversation_id, turn.turn_order, turn.question,
                 turn.rewritten_question, turn.scope, turn.answer, turn.cited_source_ids,
                 turn.insufficient_evidence),
            )
            cur.execute(
                "UPDATE cmt_veda_rag.conversations SET last_activity_at = now() WHERE conversation_id=%s",
                (turn.conversation_id,),
            )

    def get_turns(self, conversation_id: str, limit: Optional[int] = None) -> list[TurnRecord]:
        with self._cursor() as cur:
            cur.execute(
                "SELECT * FROM cmt_veda_rag.conversation_turns WHERE conversation_id=%s ORDER BY turn_order",
                (conversation_id,),
            )
            rows = cur.fetchall()
        turns = [
            TurnRecord(
                turn_id=r["turn_id"], conversation_id=r["conversation_id"],
                turn_order=r["turn_order"], question=r["question"],
                rewritten_question=r["rewritten_question"], scope=r["scope"] or "",
                answer=r["answer"], cited_source_ids=list(r["cited_source_ids"] or []),
                insufficient_evidence=bool(r["insufficient_evidence"]),
                created_at=str(r["created_at"]),
            )
            for r in rows
        ]
        if limit is not None and len(turns) > limit:
            turns = turns[-limit:]
        return turns

    def delete_conversation(self, conversation_id: str) -> None:
        with self._cursor() as cur:
            cur.execute("DELETE FROM cmt_veda_rag.conversations WHERE conversation_id=%s", (conversation_id,))

    def delete_stale_conversations(self, older_than_days: int) -> int:
        with self._cursor() as cur:
            cur.execute(
                "DELETE FROM cmt_veda_rag.conversations"
                " WHERE last_activity_at < now() - (%s || ' days')::interval"
                " RETURNING conversation_id",
                (older_than_days,),
            )
            deleted = cur.fetchall()
        return len(deleted)


def get_backend(backend: str, sqlite_path: str = "", postgres_dsn: str = "") -> DBBackend:
    if backend == "sqlite":
        return SqliteBackend(sqlite_path)
    if backend == "postgres":
        return PostgresBackend(postgres_dsn)
    raise ValueError(f"Unknown DB backend: {backend}")
