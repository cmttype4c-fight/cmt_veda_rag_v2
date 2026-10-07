# CMT Veda AI RAG v2 — Round 4: Intake Lifecycle — Status Report

Round 1 built retrieval/grounding. Round 2 made persistence, ingestion,
Discovery, and removal real and tested. Round 3 closed the remaining
hardening gaps (real PDF ingestion, idempotent migrations, URL safety,
secrets scanning). This round implements the Intake Lifecycle brief and
your 11 confirmed decisions: splitting the old fused
`approve_and_ingest()`/`ingest_document()` into
`register_intake()` → `approve_intake()` (auto-queues) → `queue_intake()`
→ `process_intake()`, backed by Redis/RQ for async delivery and
PostgreSQL as the durable source of truth for state/retry/audit.

**Environment, re-verified this round:** still no network to install
packages. New and confirmed this round: `psycopg2`/`redis`/`rq`/`fastapi`
all fail to install (`pip install --break-system-packages <pkg>` →
"Could not find a version that satisfies the requirement ... (from
versions: none)"). **New finding this round:** `postgresql-16` and
`redis-server` are both pre-installed as OS-level service binaries, even
though their Python client libraries are not — this allowed real,
live verification of both the Postgres migration and basic Redis
connectivity, by talking to each server directly (`psql`, `redis-cli`)
instead of through the (uninstallable) Python drivers. Every claim below
that says "Tested" was actually executed in this session; everything
else says so explicitly, per your decision #10.

## What was built

- `config.py` — round-4 constants: `ALLOWED_SOURCE_TYPES`,
  `MIN_FULL_TEXT_CHARS`, `MAX_RETRIES=5`, `RETRY_BACKOFF_BASE_SECONDS`,
  `retry_backoff_seconds()`, `REDIS_URL`, `RQ_QUEUE_NAME`,
  `RQ_JOB_TIMEOUT_SECONDS`, `RAG_QUEUE_BACKEND` (`redis_rq` default /
  `inline` dev-CI fallback).
- `db.py` — `DocumentRecord` extended (`source_format`, `uploaded_by`,
  `uploaded_at`, `original_filename`, `extracted_text`, `retry_count`,
  `intake_batch_id`, plus a computed `source_type` property — see
  "Deliberate deviation" below); 11 new backend methods implemented in
  both `SqliteBackend` (tested) and `PostgresBackend` (SQL written,
  shaped identically, unexecuted — no `psycopg2` here).
- `migrations/0002_intake_lifecycle.sql` — new, idempotent Postgres
  migration (additive only: `ADD COLUMN IF NOT EXISTS`, `CREATE TABLE IF
  NOT EXISTS` for `intake_batches`, `intake_batch_items`,
  `processing_jobs`). **Run live, twice, against a real local Postgres
  16 server in this sandbox — zero errors on rerun, proper "already
  exists, skipping" notices.**
- `intake.py` (new) — the four lifecycle functions plus
  `patch_intake_metadata()`, `reject_intake()`, `bulk_register_intake()`,
  `bulk_approve_intake()`, and the `IntakeError`/`FullTextRuleViolation`/
  `InvalidSourceType`/`IntakeConflictError` exception types.
- `worker.py` (new) — real RQ/Redis wiring (`enqueue_processing`,
  `run_worker`, written but unexecuted — no `rq`/`redis` packages) plus
  `InlineQueueAdapter`, the tested synchronous substitute, and the
  shared `process_intake_job()` retry/backoff orchestration used by both.
- `main.py` — new `/admin/intake/*` router (register, PDF upload, list,
  get, patch, approve, reject, retry, bulk, bulk/approve) wired to
  `intake.py`/`worker.py`. **Legacy `/admin/discovery/*` and
  `/admin/ingest/*` routes left untouched** — see below.
- `test_intake_lifecycle.py` (new) — 11 tests, run against the real
  `SqliteBackend` + `HashingTfidfEmbeddings` + `NumpyVectorStore` +
  `InlineQueueAdapter` stack. **11/11 pass.**
- `docs/API_CONTRACTS_FINAL.md` (new) — the final, implementation-grounded
  API contract for Lovable/Veda-v1 (your confirmed decision #11),
  written after this implementation and cross-checked against it. See
  that document for the full endpoint-by-endpoint contract; this report
  stays the tested-vs-unverified status summary.

## Deliberate deviations from the reviewed proposal doc (disclosed, not silent)

1. **`source_type` is a computed property, not a physical column.** The
   proposal sketched a real rename/new enum. Implemented instead as
   `DocumentRecord.source_type` — a read-time alias over the existing
   `ingestion_method` column (`'discovery'` passes through,
   everything else reads as `'direct_upload'`). Same external
   vocabulary in every round-4 API response and filter; zero schema
   change, zero risk to round 3's already-passing tests that read/write
   `ingestion_method` directly. If you'd rather have the physical
   column for a future report that queries Postgres directly without
   going through this app, say so and I'll add it as a follow-up
   migration.

2. **Legacy routes keep their round-3 internals, not rewired to the new
   lifecycle.** Decision #6 said legacy routes "MAY" use the new
   implementation internally — not "must." Rewiring `discovery_approve`
   to `register_intake()`+`approve_intake()` would silently change its
   response from "already INDEXED" (synchronous, today) to "QUEUED, will
   index later" (asynchronous) — exactly the kind of behavioral change
   that should be a deliberate choice by whoever owns that caller
   (Discovery Engine), not something changed under them this round. So:
   legacy routes are **byte-for-byte unchanged** and still pass all 10
   round-3 tests. The new `/admin/intake/*` surface is the real fix, and
   is what Lovable/new callers should move to. Tell me if you'd rather
   I rewire the legacy internals anyway — it's a small, well-scoped
   change now that `intake.py` exists.

3. **A real robustness bug found and fixed while wiring this up:**
   `queue_intake()` originally transitioned the document to `QUEUED`
   *before* calling `enqueue_fn()`. If the enqueue call itself fails
   (Redis unreachable, which is genuinely true in this sandbox), the
   document was left stranded in `QUEUED` with no job record at all —
   nothing to retry, nothing for an admin to see why. Fixed: an
   `enqueue_fn` failure now records a `processing_jobs` row (status
   `failed`, with the error) and transitions the document to `FAILED` —
   the only other state `QUEUED` is legally allowed to reach — instead
   of lying about being queued. New test:
   `test_queue_intake_enqueue_failure_does_not_strand_document_in_queued`
   — **Tested**, passes.

4. **A second real bug found and fixed while writing `docs/API_CONTRACTS_FINAL.md`:**
   `config.validate_transition()` raised a bare `ValueError` for an
   illegal state transition (e.g. approving an already-`INDEXED`
   document, or retrying one that isn't `FAILED`), and `main.py`'s
   `_intake_exc_to_http()` mapped *any* `ValueError` to `404 "No such
   document"` — so a document that exists but is simply in the wrong
   state would have incorrectly been reported as not found. Fixed:
   added `config.IllegalStateTransitionError(ValueError)`, raised
   specifically for this case (every pre-existing `except ValueError`
   call site is unaffected, since it's a subclass), and `main.py` now
   maps it to `409 Conflict` before falling through to the generic
   `ValueError -> 404` branch. New test:
   `test_illegal_transition_is_a_distinct_error_type_not_generic_valueerror`
   — **Tested**, passes. This is the one change made after you said
   "do not make further implementation changes" — it qualified as the
   "genuine contract blocker" exception you allowed, since documenting
   the old behavior as correct would have shipped a wrong contract to
   Lovable.

## Confirmed decisions 1-11 — status

| # | Decision | Status |
|---|---|---|
| 1 | Auto-queue after approval; APPROVED stays conceptually distinct from INDEXED | **Implemented + Tested** (`test_register_approve_process_indexed_and_retrievable`) |
| 2 | `PATCH /admin/intake/{id}` — pending_approval only | **Implemented + Tested** (`test_metadata_patch_allowed_pending_rejected_after_approval`); HTTP layer: not tested (see below) |
| 3 | Bulk register/approve, no rollback, per-item audit | **Implemented + Tested** (`test_bulk_register_one_bad_item_does_not_block_others`, `test_bulk_approve_mixed_results`) |
| 4 | Retry policy: MAX_RETRIES=5, exponential backoff, Postgres authoritative | **Implemented + Tested** (`test_retry_policy_fails_five_times_then_stays_failed`); Postgres-side storage: schema live-verified, Python path untested (no psycopg2) |
| 5 | Redis+RQ; never in the sync retrieval path | **Implemented**; Redis connectivity live-verified (see below); `rq` Python integration unexecuted; retrieval.py confirmed to have zero Redis/RQ import or reference |
| 6 | Keep legacy routes; add canonical routes alongside | **Implemented + Tested** (round-3 suite still 10/10 unchanged) |
| 7 | Both source types through one lifecycle, provenance preserved | **Implemented + Tested** |
| 8 | Full-text-only; Newsletter/editorial never enters RAG | **Implemented + Tested** (`test_full_text_only_rule_rejects_short_text`, `test_disallowed_source_type_rejected_newsletter_isolation`) |
| 9 | Only INDEXED retrievable | **Implemented + Tested** (lexical_search assertion in the main flow test; unchanged SQL-level filter from round 3) |
| 10 | Live validation, tested vs. unverified reported separately | **This document** |
| 11 | Final API contract after implementation, not the proposal's assumed shapes | **Done — see `docs/API_CONTRACTS_FINAL.md`** |

## The 5 live-validation flows (your decision #10) — tested vs. unverified

1. **Discovery full-text flow → Ask Veda retrieval.** Tested at the
   `intake.py`/`db.py` level: `register_intake(..., source_type=
   "discovery")` → `approve_intake` (auto-queue) → worker →
   `INDEXED` → `db.lexical_search()` finds it. HTTP layer
   (`POST /admin/intake`, `/approve`) — **not tested**: `main.py` cannot
   be imported in this sandbox (`ModuleNotFoundError: No module named
   'fastapi'`, confirmed by a fresh `pip install` attempt this round),
   same constraint round 3's `DELIVERABLES.md` already documented for
   every HTTP endpoint in this file. `main.py`'s new routes are
   byte-compiled (`py_compile` clean) and manually traced against
   `intake.py`'s real signatures, not executed end-to-end as HTTP.
2. **Direct PDF flow** (upload → extraction → metadata correction →
   approval → auto-queue → processing → indexed → retrieval). Tested at
   the function level using text input (same code path
   `register_intake` takes for PDF once `_extract_text` has run — PDF
   byte extraction itself was already tested in round 3 against a real
   `reportlab`-generated PDF and is unchanged here). The new
   `POST /admin/intake/pdf` HTTP endpoint: **not tested** (same fastapi
   constraint as above).
3. **Bulk flow**, individual success/failure tracking. **Tested**
   (`test_bulk_register_one_bad_item_does_not_block_others`,
   `test_bulk_approve_mixed_results`) — confirms a bad item never
   blocks or mismarks the others, and per-item rows in
   `intake_batch_items` match reality.
4. **Failure/retry flow**, 5-attempt policy → FAILED. **Tested**
   (`test_retry_policy_fails_five_times_then_stays_failed`) — a document
   whose embedder always raises is driven through exactly 5 attempts by
   `InlineQueueAdapter` + `process_intake_job`'s backoff logic, lands in
   `FAILED` with `retry_count == 5`, and is confirmed to NOT auto-retry
   a 6th time.
5. **Newsletter-isolation flow.** **Tested**
   (`test_disallowed_source_type_rejected_newsletter_isolation`) —
   `register_intake(..., source_type="newsletter")` (and `"editorial"`,
   `"gemini_summary"`, `""`) all raise `InvalidSourceType` before any
   document row is created; `db.list_documents_page()` confirms zero
   rows exist afterward. This is enforced structurally (an allowlist, at
   the one function every intake path goes through), not just tested —
   see `config.ALLOWED_SOURCE_TYPES`'s docstring for the honest caveat on
   what this does and doesn't catch for genuinely mislabeled content.

## Redis — what was and wasn't actually verified this round

`redis-server` (v7.0.15) is pre-installed as an OS service, same as
Postgres. Started it live in this sandbox and confirmed:
- `redis-cli ping` → `PONG` (real connectivity, not asserted).
- `RPUSH`/`LRANGE` and `HSET`/`HGETALL` against real keys succeed —
  the two primitives RQ's queue mechanism is built on.

**Not verified:** the actual `redis`/`rq` Python packages (both fail to
install — no network), so `worker.enqueue_processing()` and
`worker.run_worker()` are written but never executed against this live
Redis. `InlineQueueAdapter` is what every test above actually runs
through. If you can run `pip install redis rq` in your own CI/deploy
environment (which has normal network access), I'd recommend doing a
real `enqueue_processing()` → `run_worker()` round-trip there before
first production use — everything is wired for it, but it has not been
executed by me.

## Tested vs. not-tested — full table

| Item | Status |
|---|---|
| `register_intake` (discovery + direct_upload) | Implemented + Tested |
| `approve_intake` (auto-queue, decision #1) | Implemented + Tested |
| `queue_intake` (incl. enqueue-failure handling) | Implemented + Tested |
| `process_intake` (worker-side pipeline) | Implemented + Tested |
| `patch_intake_metadata` (pre/post-approval lock) | Implemented + Tested |
| `reject_intake` | Implemented + Tested |
| `bulk_register_intake` / `bulk_approve_intake` | Implemented + Tested |
| Full-text-only floor (`MIN_FULL_TEXT_CHARS`) | Implemented + Tested |
| Source-type allowlist / Newsletter isolation | Implemented + Tested |
| Retry policy (5 attempts, backoff, stays FAILED) | Implemented + Tested |
| Duplicate detection (unchanged from round 3) | Implemented + Tested (regression-checked) |
| SQLite backend (11 new methods) | Implemented + Tested |
| PostgreSQL backend (11 new methods) | Implemented, not tested in sandbox (no psycopg2) |
| Postgres migration `0002_intake_lifecycle.sql` | Implemented + Tested live (run twice, idempotent) |
| `InlineQueueAdapter` | Implemented + Tested |
| Real RQ/Redis wiring (`worker.enqueue_processing`, `run_worker`) | Implemented, not tested in sandbox (no `rq`/`redis` packages) |
| Redis server connectivity (`PING`, list/hash primitives) | Tested live |
| `/admin/intake/*` route handlers (logic, via manual trace + py_compile) | Implemented, not tested in sandbox (no `fastapi`) |
| Legacy `/admin/discovery/*`, `/admin/ingest/*` routes | Unchanged; round-3 suite still 10/10 — Implemented + Tested |
| Round-3 regression (`test_pipeline_e2e.py`) | Re-run this round: 10/10 still pass |

## Recommended next step for a real deployment

Before first production use, in an environment with real network access:
1. `pip install psycopg2-binary redis rq fastapi python-multipart uvicorn`.
2. Run `migrations/0002_intake_lifecycle.sql` against the real Postgres
   (already proven idempotent here).
3. Start `python worker.py` against the real Redis, then exercise one
   full `POST /admin/intake` → `/approve` → poll `GET /admin/intake/{id}`
   round-trip over real HTTP, to close the one gap this sandbox
   genuinely cannot close (no `fastapi`/`rq`/`redis` Python packages,
   confirmed by a fresh install attempt this round).

## Final API contract for Lovable (decision #11) — done

`docs/API_CONTRACTS_FINAL.md` has the exact request/response JSON for
every `/admin/intake/*` endpoint, plus the legacy routes, `/api/ask`,
auth, lifecycle, Redis/RQ, and a full tested/unverified matrix — sourced
directly from the Pydantic models and functions actually implemented
in `main.py`/`intake.py`/`worker.py`/`db.py` (not the earlier proposal
doc's assumed shapes). Writing it surfaced the illegal-transition bug
fixed in deviation #4 above.
