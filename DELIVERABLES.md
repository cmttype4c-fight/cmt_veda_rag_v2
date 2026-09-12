# CMT Veda AI RAG v2 — Final Hardening Before VPS Staging: Status Report

This is round 3. Round 1 built the retrieval/grounding architecture.
Round 2 made persistence, ingestion, Discovery, and removal real and
tested (against a local substitute stack). This round closes the
remaining gaps the round-2 report flagged as open: real PDF ingestion,
idempotent Postgres migrations, a tested corpus-migration script, URL
sanitization for `public_reference_url`, a self-tested secrets scanner,
and a real (if unexecutable-here) Qwen validation script.

**Environment, unchanged and re-verified this round:** no network, no
Postgres server, no `psycopg2`/`fastapi`/`pydantic`/`faiss`/`llama_cpp`,
no `.gguf` model file anywhere in this sandbox. What IS available and new
this round: `reportlab` (used to generate real PDFs for testing) and
`psutil` (used by benchmark.py). Every claim below that says "tested" was
actually executed in this session; everything else says so explicitly.

## What changed this round

- `url_safety.py` (new) — rejects/strips direct PDF and object-storage
  URLs from `source_url`/`public_reference_url`, applied at both
  ingestion time and response time.
- Real PDF ingestion: `ingestion_pipeline.py` now validates PDF magic
  bytes/size/page-count, extracts text via `pypdf`, and — this session —
  was actually run against a `reportlab`-generated PDF through the full
  pipeline (chunk → embed → index → retrieve). A text-less/scanned PDF is
  correctly rejected rather than silently indexed empty.
- `main.py` gained a real multipart `/admin/ingest/manual/pdf` endpoint
  (`UploadFile` + `Form(...)`, not JSON+base64) using the same
  `ingest_document()` pipeline as everything else.
- `db/schema.sql`: fixed two real correctness issues — `CREATE TYPE`
  doesn't support `IF NOT EXISTS` in Postgres (now wrapped in the
  standard `DO $$ ... EXCEPTION WHEN duplicate_object ...` idiom, making
  the migration genuinely safe to rerun), and added `pgcrypto` for
  `gen_random_uuid()` portability to pre-13 Postgres.
- `db.py`: fixed a real bug where `SqliteBackend` would fail to open a DB
  file in a not-yet-created directory (bit immediately when testing the
  migration script against a fresh versioned path — exactly the scenario
  a real migration run would hit).
- `scripts/migrate_corpus.py` (new) — walks a document directory,
  ingests supported files (txt/md/pdf) into a **new** versioned
  index/DB, skips unsupported formats loudly, never touches the existing
  index. Actually run this session against a synthetic corpus (2 text
  files + 1 real PDF + 1 unsupported `.docx`) — all three supported
  files ingested and became queryable, the `.docx` was correctly skipped,
  and the original working directory was left untouched.
- `scripts/postgres_smoke_test.py` (new) — real, complete verification
  script for the Postgres path. Written, never run (no server).
- `scripts/run_qwen_validation.py` (new) — real, complete script to run
  the actual model against the 9 mandatory questions + 1 unsupported
  question × 3 personas, with automated citation/refusal checks and an
  explicit re-test of the section-48 SH3TC2/HDAC6-overgeneralization bug.
  Written, never run (no model).
- `scripts/secrets_scan.py` (new) — actually run against this repo. The
  first version had a real bug (a word-boundary regex issue meant it
  silently missed a deliberately-planted test secret in a variable named
  `RAG_API_KEY_TEST`); caught by self-testing against a planted secret
  before reporting it as working, fixed, reverified. Repo is clean.
- `.gitignore` (new) — keeps `.env`, `*.db`, vector-store files, and
  model weights out of version control.
- `benchmark.py`: added an explicit `estimated_queue_delay_s` metric per
  concurrency level (item 10 asked for "queueing" as its own reported
  number, not folded into latency).

All of round 2's tests were rerun after these changes; zero regressions
(`tests/test_pipeline_e2e.py` 10/10, plus the standalone unit-level logic
tests). Two new test files this round, both passing:
`tests/test_pdf_ingestion.py` (3/3) and `tests/test_corpus_migration.py`
(1/1).

---

## Against the 12 numbered items in this brief

**1. PostgreSQL** — Schema made idempotent (real fix, see above) and the
source-ID sequence race condition from round 2 remains fixed. The
persistence path itself is still **not verified against a live server**
— there is no Postgres, no `psycopg2`, and no network to get either in
this sandbox, and the brief explicitly says not to fake this. `db.py`'s
`PostgresBackend` is real code, written to the same interface as the
`SqliteBackend` that passes every test, and `scripts/postgres_smoke_test.py`
is ready to run and will tell you definitively — but nobody has run it.
**Implemented but not tested in sandbox.**

**2. Ingestion lifecycle** — Unchanged from round 2, re-verified this
round (no regressions): `DISCOVERED → PENDING_APPROVAL → APPROVED →
QUEUED → PROCESSING → INDEXED`, `PROCESSING → FAILED`,
`INDEXED → REMOVED`, illegal transitions rejected, only `INDEXED` is
retrievable. **Implemented + Tested.**

**3. Manual PDF ingestion** — This round's main upgrade. Real PDF →
validate → (approval already required upstream) → extract → chunk →
embed → index → INDEXED, run end-to-end against a real generated PDF.
The HTTP endpoint is written correctly (proper FastAPI multipart syntax)
but unexecuted (no `fastapi`/`python-multipart` here). PDF text
extraction fidelity against real publisher-formatted (often multi-column)
scientific PDFs remains unverified — `reportlab` produces simple
single-column output, so this proves the pipeline wiring, not extraction
robustness against your actual corpus's PDF layouts. **Pipeline:
Implemented + Tested. HTTP endpoint: Implemented but not tested in
sandbox. Real-publisher-PDF extraction fidelity: not verified, flagged
explicitly.**

**4. Discovery integration** — Unchanged from round 2, re-verified:
approve/reject/duplicate/inaccessible-source rejection all tested for
real against `SqliteBackend`. `discovery_candidate_id` preserved and
linked. **RAG-side contract: Implemented + Tested. Live call from an
actual Discovery Engine: Not implemented (no such system exists in this
sandbox to call).**

**5. Source protection** — `/files/*` remains absent from `main.py` (by
inspection; can't assert this programmatically without `fastapi`). New
this round: `url_safety.py` actually rejects/strips direct PDF and
object-storage URLs from `public_reference_url`, tested with 12 cases
including real PubMed/DOI/journal URLs (correctly kept) and PDF/S3/
localhost URLs (correctly stripped), applied at both write time and
response time. Persona-filtered response shape logic (Source ID only for
student; bibliographic metadata for clinician/researcher) is unchanged
from round 2 and still **unexecuted** — `main.py` cannot be imported or
run without `fastapi`/`pydantic`. **URL sanitization: Implemented +
Tested. Persona-filtered HTTP response shape: Implemented but not tested
in sandbox.**

**6. Stable Source IDs** — SQLite path tested across a simulated restart
(round 2, re-verified this round). Postgres path uses a real `SEQUENCE`
(fixed from a racy `MAX()+1` approach in round 2) — correct by
construction, not verified against a live server. **SQLite: Implemented
+ Tested. Postgres: Implemented but not tested in sandbox.**

**7. Existing corpus migration** — `scripts/migrate_corpus.py`, run for
real against a synthetic corpus this session (see above). Writes to a
new versioned path/knowledge_version, never touches the existing index —
rollback is simply "don't switch production to point at the new path
until it's validated." **Not run against your actual 11-document
corpus** (not available in this sandbox) — that run, and a review of the
filename-based title-inference fallback against your real files' naming,
is still an open step before cutover. **Script: Implemented + Tested
(against a synthetic corpus). Run against the real corpus: Not
implemented (nothing to run it against here).**

**8. Real Qwen validation** — `scripts/run_qwen_validation.py` is
complete: runs all 9 mandatory questions + 1 unsupported question across
student/clinician/researcher, auto-checks citation validity and refusal
behavior, and specifically flags (for human review, not auto-pass/fail)
whether the general "What is CMT?" answer is dominated by narrow
SH3TC2/HDAC6/aggresome/autophagy/CMT4C terms — the exact fingerprint of
the original bug. **This has not been run.** No `llama_cpp`, no model
weights, no network, anywhere in this sandbox. Nothing in this codebase
or report claims the real model's grounding behavior has been verified.
**Implemented but not tested in sandbox — and this is the single most
important thing to run before calling RAG v2 production-ready.**

**9. Keep current retrieval approach** — Unchanged: hybrid (semantic +
SQLite-FTS5/Postgres-tsvector lexical) + entity matching + metadata
weighting + deterministic (explicitly non-neural, documented as such
everywhere) reranking + diversification + multi-signal evidence gate. No
neural reranker introduced. **Implemented + Tested** (SQLite side);
Postgres lexical side **Implemented but not tested in sandbox**.

**10. Performance** — `benchmark.py` measures cold start, 10+ warm
sequential requests (min/avg/median/p95/max), concurrency at 1/2/5,
CPU/RAM (via `psutil` if present), failures, and now an explicit
`estimated_queue_delay_s` per concurrency level. **Never run** — no VPS,
no live server, no model. **Implemented but not tested in sandbox. No
performance numbers exist. Do not upgrade the VPS based on this report.**

**11. Security** — API-key auth, server-side-only secrets, and citation-
layer prompt-injection resistance are unchanged from round 2 and
re-verified. New this round: an actual secrets scan was run against this
repo (self-tested against a planted fake secret to confirm the scanner
isn't a rubber stamp — the first version had a real bug that silently
missed the planted secret, which was caught and fixed before reporting
this as clean). "No direct browser → RAG" and "patient data never enters
scientific RAG" remain architectural properties verified by inspection,
not runtime checks (there's no network topology or patient-data system
in this sandbox to test against). Real-model prompt-injection resistance
(does Qwen actually ignore injected instructions in document text, not
just fail to cite a fake source) is unverified — needs the real model.
**Secrets scan: Implemented + Tested. Citation-layer injection defense:
Implemented + Tested. Real-model injection resistance: Not implemented/
not tested (needs the real model). Architectural properties (patient-
data isolation, no-direct-browser-access): Implemented, verified by
inspection only.**

**12. Deliverable status** — see the table below.

---

## Capability status table

| Capability | Status |
|---|---|
| SQLite persistence (dev/test substitute) | Implemented + Tested |
| PostgreSQL persistence (production) | Implemented but not tested in sandbox |
| Postgres schema idempotency (rerunnable migrations) | Implemented but not tested in sandbox |
| Source ID stability across restart (SQLite) | Implemented + Tested |
| Source ID stability / atomic allocation (Postgres sequence) | Implemented but not tested in sandbox |
| Ingestion lifecycle state machine + illegal-transition rejection | Implemented + Tested |
| Manual text/markdown ingestion (full pipeline) | Implemented + Tested |
| Manual PDF ingestion (full pipeline: validate/extract/chunk/embed/index) | Implemented + Tested |
| Manual PDF ingestion HTTP endpoint (multipart) | Implemented but not tested in sandbox |
| PDF extraction fidelity vs. real publisher-formatted papers | Not implemented (unverified against real corpus PDFs) |
| Discovery RAG-side contract (register/approve/reject/duplicate/inaccessible) | Implemented + Tested |
| Discovery HTTP endpoints | Implemented but not tested in sandbox |
| Live Discovery Engine integration | Not implemented (no such system in sandbox) |
| Removal actually prevents retrieval (DB + active-filter + lexical index) | Implemented + Tested |
| Hybrid retrieval — SQLite FTS5 lexical + numpy semantic | Implemented + Tested |
| Hybrid retrieval — Postgres tsvector/GIN lexical | Implemented but not tested in sandbox |
| Hybrid retrieval — FAISS semantic (production vector store) | Implemented but not tested in sandbox |
| Entity-aware exact-terminology boosting | Implemented + Tested |
| Deterministic (non-neural) reranking, honestly labeled | Implemented + Tested |
| General-vs-specific retrieval diversification | Implemented + Tested |
| Multi-signal evidence-sufficiency gate | Implemented + Tested |
| URL safety (reject/strip PDF/storage URLs from public references) | Implemented + Tested |
| Citation validation (anti-hallucination) | Implemented + Tested |
| `/files/*` route absence | Implemented, verified by code inspection only (not programmatically — no fastapi here) |
| Persona-filtered HTTP response shape (Source ID vs. rich metadata) | Implemented but not tested in sandbox |
| Server-authoritative role → persona mapping | Implemented + Tested |
| Role-aware generation prompts (student/clinician/researcher instructions) | Implemented but not tested in sandbox (real-model output unverified) |
| Strict-grounding / no-forced-sections prompt rules | Implemented but not tested in sandbox (real-model behavior unverified) |
| Real Qwen model grounding/generalization behavior | Not implemented / not tested (no model available anywhere in this sandbox) |
| Section-48 regression (general CMT question not SH3TC2/HDAC6-dominated) — retrieval layer | Implemented + Tested |
| Section-48 regression — actual model output | Not tested (script ready: `scripts/run_qwen_validation.py`) |
| Corpus migration script | Implemented + Tested (against synthetic corpus) |
| Corpus migration — run against real 11-document corpus | Not implemented (corpus unavailable here) |
| Performance benchmark script (cold start/warm/concurrency/CPU/RAM/queueing) | Implemented but not tested in sandbox |
| Secrets scan | Implemented + Tested (self-verified against a planted secret) |
| Prompt-injection defense — citation layer | Implemented + Tested |
| Prompt-injection defense — real model | Not implemented / not tested |
| Patient-data separation (architectural) | Implemented, verified by inspection |
| No-direct-browser-access (architectural/deployment requirement) | Implemented, verified by inspection |
| Knowledge versioning (recording, not managing, a version per answer) | Placeholder |
| Contradiction detection across sources | Not implemented |
| Temporal-awareness-in-ranking ("latest research" guarding) | Not implemented |
| Table/figure structured extraction (beyond caption-level) | Placeholder |

---

## Required final output

- **Updated ZIP** — provided alongside this report.
- **Changed-file list** — new: `url_safety.py`, `scripts/migrate_corpus.py`,
  `scripts/postgres_smoke_test.py`, `scripts/run_qwen_validation.py`,
  `scripts/secrets_scan.py`, `tests/test_pdf_ingestion.py`,
  `tests/test_corpus_migration.py`, `.gitignore`. Modified:
  `ingestion_pipeline.py` (real PDF path), `main.py` (PDF endpoint +
  URL-safety wiring), `db.py` (Postgres source-ID sequence fix, SQLite
  parent-directory fix), `db/schema.sql` (idempotency fixes),
  `benchmark.py` (queueing metric).
- **PostgreSQL migrations** — `db/schema.sql`, now safe to rerun.
- **Discovery integration contract** — `discovery.py` (unchanged this
  round, re-verified).
- **Manual PDF ingestion implementation** — `ingestion_pipeline.py` +
  `main.py`'s `/admin/ingest/manual/pdf`, pipeline tested, endpoint
  untested (no fastapi here).
- **Source-protection tests** — `url_safety.py`'s 12-case test (run
  inline this session; consider promoting to a `tests/` file if you want
  it in CI), plus the round-2 citation-validation and persona-field-set
  tests (still valid, still require `fastapi` to execute the HTTP-layer
  half).
- **Real-model test instructions/results** — instructions and a complete
  script (`scripts/run_qwen_validation.py`) are provided. **No results**
  — the model has never been run. This is the top item to close before
  staging.
- **Benchmark script** — `benchmark.py`, complete, never run.
- **Updated DELIVERABLES.md** — this file.
- **Deployment instructions** — (1) apply `db/schema.sql` to real
  Postgres, rerun it once more to confirm idempotency; (2) run
  `scripts/postgres_smoke_test.py` against it; (3) install
  `psycopg2-binary`, `sentence-transformers`, `faiss-cpu`, `llama-cpp-
  python`, `python-multipart` in the deployment venv; (4) run
  `scripts/migrate_corpus.py` against the real 11-document corpus into a
  new versioned path; (5) run `scripts/run_qwen_validation.py` against
  the new index and manually review the flagged items; (6) run
  `benchmark.py` and get real numbers before any VPS-sizing decision; (7)
  only then point `RAG_DB_BACKEND=postgres` /
  `RAG_EMBEDDING_BACKEND=sentence_transformers` /
  `RAG_VECTOR_BACKEND=faiss` / `RAG_GENERATION_BACKEND=llama_cpp` at
  production and cut over.
- **Rollback instructions** — unchanged: v1's `main.py` has never been
  modified by any round of this work; reverting the systemd unit to it is
  a complete rollback. The new versioned Postgres `knowledge_version` /
  sqlite-vector-store path from a migration run can also simply not be
  switched to if validation fails — nothing is overwritten in place.
- **Remaining limitations** — Postgres, FAISS, sentence-transformers, and
  the real Qwen model remain completely unexecuted anywhere in this
  process; `main.py`'s HTTP layer has never been run or imported
  successfully; real-model generation-quality/grounding is entirely
  unverified; PDF extraction against real publisher-formatted papers is
  unverified; no performance numbers exist; the real 11-document corpus
  has not been migrated; knowledge-version management, contradiction
  detection, and temporal-awareness ranking remain unbuilt. None of this
  should be read as "close enough" — each is a specific, nameable step
  between this report and a defensible VPS-staging decision.

---

## Round 4: Multi-turn conversation support

Added after a direct question: "can a user ask a follow-up, and does the
RAG follow the previous chat?" The honest answer before this round was
no — every `/api/ask` call was fully stateless. This round adds real
multi-turn support, built and tested the same way as every other round:
actually implemented, actually run against the local substitute stack,
gaps stated plainly rather than implied away.

**Design decision, stated explicitly because it's the one that matters
most:** conversation history is used ONLY to rewrite an ambiguous
follow-up into a standalone question BEFORE retrieval. It is never fed
into the generation prompt. Feeding raw prior answers into generation
would let the model restate facts from an earlier turn that weren't in
THIS turn's freshly retrieved evidence — silently breaking the "the
supplied retrieved evidence is your EXCLUSIVE factual source" guarantee
that every other round of this work has protected. So: history changes
*what gets searched*, never *what the model is told to treat as fact*.

**What's new:**
- `db.py` / `db/schema.sql`: `conversations` and `conversation_turns`
  tables (SQLite tested, Postgres written-not-tested, same pattern as
  everything else). `delete_conversation()` for a real "clear my chat"
  action — flagged explicitly that there's no automatic retention/expiry
  job yet, which matters because conversation logs are a new
  sensitive-data surface (patient questions) that didn't exist before,
  separate from the PHI-free scientific corpus.
- `conversation.py` (new): `HeuristicQueryRewriter` — deterministic,
  no model call, splices the most recent turn's specific entity into a
  pronoun-bearing follow-up ("What causes **it**?" -> "What causes
  **CMT4C**?"). Real and tested, and a legitimate production component
  in its own right, not just a test substitute — skipping a second LLM
  call for the common case matters on a CPU-bound 2 vCPU box where a
  single generation call already costs ~4s. `LLMQueryRewriter` — real
  code sharing the same `Generator`/model as answer generation, for
  follow-ups the heuristic can't resolve; **not executed anywhere** (no
  model in this sandbox). `FallbackQueryRewriter` composes both: try free
  first, fall back to the model only if needed.
- `main.py`: `AskRequest` gains an **optional** `conversation_id` —
  omitting it (every existing integration) keeps the endpoint exactly as
  stateless as before; this is fully backward compatible, not a breaking
  change. When provided, the same id must be sent on every turn (the
  server never invents conversation ids). New
  `DELETE /api/conversations/{conversation_id}` endpoint. Like the rest
  of `main.py`, this is written but **not executed** — no `fastapi` in
  this sandbox.

**What's tested — 7/7 in `tests/test_conversation.py`, all real:**
persistence surviving a simulated restart, the idempotent
`create_conversation`, `get_turns(limit=...)`, `delete_conversation`
actually deleting; the heuristic rewriter resolving a pronoun, leaving
self-contained questions alone, refusing to guess with no history to
resolve against, and correctly preferring the *most recent* topic when a
conversation has moved on. And the one that answers the original
question directly: an explicit **before/after** test proving `"What
causes it?"` asked bare gets classified `general` and fails to find
CMT4C-specific evidence (confirming the bug is real), while the same
question run through the rewriter resolves to `"What causes CMT4C?"`,
gets classified `specific`, and correctly retrieves the SH3TC2 evidence.
A final test simulates two separate "requests" the way production
actually works — turn 2 reads turn 1's history back from the **database**,
not a variable still sitting in memory, which is the part that actually
has to work for this to function across real, independent HTTP calls.

**What's not done:** the LLM-fallback rewriter path (needs the real
model, same as every generation-quality claim in this whole project); the
retention/expiry job for conversation data; and — this is worth being
direct about — the heuristic rewriter is a simple pronoun-splice, not a
language model. It will not handle a genuinely complex rephrasing
("what about the other one we discussed earlier" with three prior
topics) gracefully; it'll either pick the most recent topic (possibly
wrong) or pass the question through unresolved. That's exactly the case
`LLMQueryRewriter` exists for, and exactly why it needs to actually be
run and evaluated before being relied on for anything beyond simple
pronoun follow-ups.

---

## Round 5: Save chat + resume later

Extended round 4's multi-turn support (which only lasted within a single
back-and-forth) into actual persistent, resumable chat history.

**What's new:**
- `conversations` table gains `user_id` (a TRUSTED value Veda's backend
  passes after authenticating the real end user — same trust boundary as
  `app_role`; this service still never authenticates end users itself)
  and `title` (auto-derived from the first question, truncated to 60
  chars, and — this is load-bearing, not incidental — stable for the
  life of the conversation: `create_conversation` is called on every
  turn but only actually writes the row on the FIRST call, via `INSERT
  OR IGNORE`, so a later question's title never overwrites the original).
- `GET /api/conversations?user_id=...` — lists a user's previous
  conversations, most-recently-active first, for a "continue where you
  left off" UI.
- `GET /api/conversations/{conversation_id}?user_id=...` — full
  transcript for resuming.
- Ownership enforcement (`conversation.is_authorized_for_conversation`,
  deliberately kept out of `main.py` so it's testable without `fastapi`):
  a conversation with a `user_id` attached can only be listed/fetched/
  deleted by that same `user_id`; mismatches return 404 rather than 403
  so an unauthorized caller can't even confirm the conversation exists.
  A conversation with no `user_id` (client chose not to pass one) stays
  accessible by anyone holding the `conversation_id`, matching round 4's
  original behavior.
- `db.delete_stale_conversations(older_than_days)` +
  `scripts/cleanup_stale_conversations.py` — round 4 flagged "no
  retention policy" as an open gap given patient questions are a new
  sensitive-data surface; this round actually closes it with a real,
  tested deletion mechanism. Scheduling it (cron/systemd timer) is still
  an operational step, not automatic — the script does the deletion, it
  doesn't run itself.

**Tested — 12/12 in `tests/test_conversation.py` (5 new since round 4),
all real:** title stability across repeated create-conversation calls;
per-user listing with confirmed cross-user isolation (a second user's
conversations never leak into the first user's list) and correct
recency ordering; the ownership-authorization logic directly, all four
cases (unowned/no-request-id, unowned/some-id, owned/matching,
owned/mismatched); stale-conversation deletion using a real backdated
timestamp (not mocked), confirming only conversations past the threshold
are removed; and the actual feature requested — a full "resume later"
test that closes the DB connection and vector store entirely, reopens
brand-new instances against the same files (as close to a real restart
as a single-process test gets), lists the user's conversations to find
the right one, renders the prior transcript, asks a new follow-up, and
confirms it still resolves correctly against the OLD persisted history
before being appended as turn 2.

**What's not done:** `main.py`'s three conversation endpoints (list/
detail/delete) are written but — same as every other round — never
executed, since `fastapi` isn't installed in this sandbox. The retention
job needs to actually be scheduled on the real deployment; nothing here
runs it automatically. And resumed transcripts return bare
`cited_source_ids` rather than re-expanded persona-filtered source
objects (title/authors/DOI/etc.) — kept simple deliberately rather than
adding a source_id-to-document lookup that doesn't exist elsewhere in
this codebase; if rich resumed-source display matters, that's a
follow-up, not silently assumed to already work.

---

## Round 6: VPS deployment diagnosis (Platform 1 response)

Direct response to a real deployment report: model loaded successfully,
but every basic question returned "insufficient evidence," and
validation then crashed with `ValueError: Requested tokens (16023)
exceed context window of 4096`. Found and fixed 5 real bugs; one
character-based stopgap (`CONTEXT_MAX_CHARS`) is explicitly replaced
with a token-based mechanism per the direct question asked.

### Bugs found and fixed

1. **`CONTEXT_CHARS_PER_CHUNK` was declared in `config.py` but never
   referenced in `generation.py`.** Confirmed with `grep` before touching
   anything — zero usages. Every candidate's full chunk text (up to
   1800 chars from `ingestion.py`'s chunker) was going into the prompt
   uncapped, across up to 12 chunks. This is the single largest
   contributor to the reported token overflow. Fixed: now applied as a
   cheap per-chunk cap in `build_messages()`.

2. **The real fix for the crash is token-based, not character-based** —
   directly answering the question asked ("whether the 8000-character
   context budget is the correct fix or whether the repository should
   instead calculate a token-based prompt budget"). Character-to-token
   ratio is not fixed: gene symbols, HGVS mutation notation
   ("p.Arg1109X"), and any residual PDF-extraction noise all tokenize
   worse than plain English prose, so no fixed character number is ever
   safe. Added `Generator.count_tokens()` (real tokenizer via
   `Llama.tokenize()` for the production model) and
   `fit_candidates_to_token_budget()` in `generation.py`, which drops
   least-relevant candidates one at a time until the prompt ACTUALLY fits
   `n_ctx` (measured, not estimated), then raises `ContextBudgetError`
   (rather than silently guessing) if even zero evidence doesn't fit.
   Wired into `main.py`'s `ask()` and `scripts/run_qwen_validation.py`.
   Tested for real (mock tokenizer, since no `llama_cpp`/model here):
   correctly drops from the least-relevant end first, correctly raises
   on an impossible budget, correctly leaves a well-fitting prompt alone.

3. **Most likely primary cause of "insufficient evidence" on every
   basic question: vector-store/backend mismatch between migration and
   validation.** `scripts/migrate_corpus.py` defaults to
   `--vector-backend numpy --embedding-backend hashing_tfidf`. If the
   real migration run didn't explicitly override BOTH to match
   validation's `--vector-backend faiss --embedding-backend
   sentence_transformers --vector-path /opt/cmtveda/rag-v2/runtime/
   faiss_index`, the FAISS index at that path was never populated —
   `FaissVectorStore.__init__` silently creates an empty index rather
   than erroring. Semantic retrieval then returns nothing for every
   query while Postgres still correctly reports 197 chunks — exactly
   matching the reported symptom. **Not verified against your actual
   deployment** (no Postgres/FAISS here) — this is a diagnosis to check,
   not a confirmed root cause. Built `scripts/diagnose_retrieval.py`
   specifically to check this FIRST, before running any questions:
   `vector_store.size()` printed immediately; tested against both a
   healthy and a deliberately-empty vector store to confirm the check
   actually distinguishes them.

4. **A second, more subtle real bug: lexical score scale mismatch
   between SQLite FTS5 and Postgres.** The confidence-blending formula
   divided the raw lexical score by a hardcoded `5.0`, calibrated
   against SQLite's `bm25()` scale. Postgres's `ts_rank_cd()` lives on a
   much smaller scale (typically well under 1.0 even for a strong
   match), so on Postgres specifically this made the lexical signal
   contribute almost nothing to confidence — compounding bug #3's
   effect. Fixed with a backend-conditional config default
   (`LEXICAL_SATURATION`, sqlite=5.0 unchanged / postgres=0.5 new),
   overridable via `RAG_LEXICAL_SATURATION`. The Postgres value is a
   **reasoned starting point, not empirically calibrated** — there is no
   Postgres instance in this sandbox to verify the real `ts_rank_cd`
   scale against; benchmark and tune against your real corpus.

5. **A related SQLite-specific bug found while testing the above (does
   NOT affect your Postgres deployment):** the SQLite FTS5 query
   sanitizer included stopwords ("what", "is") as OR terms, and the FTS5
   table had no stemming configured, so "supported" never matched
   "supportive". Postgres's `ts_rank_cd`/`plainto_tsquery` with the
   `'english'` config already does both (stopword removal AND stemming)
   automatically — this was purely a gap in the SQLite dev/test
   substitute's fidelity, not something present in the real deployment.
   Fixed anyway, for correctness of this repo's own test suite: stopword
   filtering added to `_sanitize_fts_query`, and the FTS5 table now uses
   `tokenize = 'porter unicode61'` (SQLite's built-in Porter stemmer).
   Fixing this uncovered that one existing test had been passing only
   because the stopword-pollution bug was inflating its score — the test
   corpus was enriched with more realistic document structure (matching
   the same "make the fixture structurally realistic" pattern used
   earlier in this project for the SH3TC2 near-miss) rather than
   reintroducing the bug to keep the test green.

### Also backported (matches fixes you'd already made independently)

- **`knowledge_version` propagation**: `IngestionInput` had no such field
  at all; `migrate_corpus.py`'s `build_ingestion_input()` accepted a
  `knowledge_version` parameter and never used it. Every migrated
  document silently got `knowledge_version=None` regardless of the
  migration run's `--knowledge-version` flag. Fixed in the source of
  truth (both `IngestionInput` and the `DocumentRecord` construction in
  `ingest_document()`), not left as a field-only hotfix. Verified: a
  document ingested with `knowledge_version="v2-2026-09-12"` now actually
  carries that value.
- **PDF NUL-character stripping**: your fix
  (`.replace("\x00", "")`) is correct and is now in
  `ingestion_pipeline.py`'s `extract_text_from_pdf_bytes()` directly,
  not just in a deployed hotfix.

### What this diagnosis does NOT do

- **Does not weaken the evidence gate.** `EVIDENCE_SCORE_FLOOR` (0.28) is
  untouched. Per your explicit instruction and my own read of the
  situation: the floor isn't the bug — an empty/mismatched vector store
  and a miscalibrated lexical-scale constant are. Once semantic
  retrieval is actually populated, real `sentence-transformers` cosine
  similarities for genuinely relevant passages are typically well above
  what the toy substitute embedder in this sandbox produces, so the
  floor should if anything become MORE easily clearable, not less —
  monitor after the fix, don't preemptively lower it.
- **Does not recommend a VPS upgrade.** Nothing in this diagnosis points
  at CPU/RAM as the bottleneck. `--n-ctx 8192` (if headroom is still
  tight after these fixes) is a software config change to `Llama()`'s
  init parameter, not a hardware change — Qwen2.5-1.5B's KV cache at
  8192 vs 4096 context adds a modest, bounded amount of RAM, not a
  VPS-tier change.
- **Does not touch Discovery Engine.** Out of scope per your instruction;
  nothing in `discovery.py` was modified this round.
- **Does not claim the primary hypothesis (vector-store mismatch) is
  confirmed.** It's the most likely explanation given how
  `migrate_corpus.py`'s defaults work, consistent with every symptom
  reported, and directly checkable in under a minute with
  `scripts/diagnose_retrieval.py`'s first preflight line — but it has
  not been verified against your actual Postgres/FAISS instance, because
  neither exists in this sandbox. Run the script; its first printed line
  will confirm or rule this out immediately.

### What to actually run, in order

1. `python3 scripts/diagnose_retrieval.py --db-backend postgres
   --postgres-dsn "$RAG_POSTGRES_DSN" --vector-path
   /opt/cmtveda/rag-v2/runtime/faiss_index --vector-backend faiss
   --embedding-backend sentence_transformers --model-path
   /opt/cmtveda/rag/runtime/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
   --n-ctx 4096 --n-threads 2` — read the three preflight lines first.
   If line 1 (`vector_store.size()`) is 0 or absurdly low relative to
   197 chunks, that's confirmed: re-run migration with
   `--vector-backend faiss --embedding-backend sentence_transformers
   --vector-path /opt/cmtveda/rag-v2/runtime/faiss_index` explicitly
   matching validation's flags exactly.
2. Rebuild the image with this round's fixes (`generation.py`, `main.py`,
   `retrieval.py`, `config.py`, `db.py`, `ingestion_pipeline.py`,
   `scripts/migrate_corpus.py`, `scripts/run_qwen_validation.py`).
3. Re-run `scripts/diagnose_retrieval.py` on the three reported
   questions — check `sufficient=True` and `fits=True` for each before
   re-running full validation.
4. Re-run `run_qwen_validation.py` for the full 9-question + persona +
   unsupported-question suite.
