# CMT Veda AI RAG v2 — Final API Contract (Round 4, as implemented)

This document describes **only what is actually implemented in code
today** (`main.py`, `intake.py`, `worker.py`, `config.py`, `db.py`,
`ingestion_pipeline.py`), as of Round 4. It supersedes the earlier
proposal doc ("Intake Lifecycle API & Schema Proposal") wherever the two
disagree — the proposal described intended shapes before implementation;
this document describes what was actually built, cross-checked against
the Pydantic models, routers, `intake.py`, `db.py`, and
`test_intake_lifecycle.py` (11/11 passing).

Every claim below is one of three states, stated explicitly wherever it
matters:
- **IMPLEMENTED + TESTED** — exercised directly in this sandbox (either
  via `test_intake_lifecycle.py`/`test_pipeline_e2e.py`, both currently
  11/11 and 10/10 passing, or via a live `psql`/`redis-cli` session).
- **IMPLEMENTED — RUNTIME UNVERIFIED** — the code exists and was
  reviewed line-by-line, but could not be executed as HTTP in this
  sandbox: `fastapi` cannot be installed here (`pip install
  --break-system-packages fastapi` → "Could not find a version that
  satisfies the requirement ... (from versions: none)"), so `main.py`
  cannot be imported or run. This is the same constraint Round 3's
  `DELIVERABLES.md` already documented for every HTTP endpoint in this
  file — it is not new or specific to Round 4.
- **NOT IMPLEMENTED** — no code exists for it; it was proposed/discussed
  but never built.

---

## 1. Authentication

Two independent, static shared-secret schemes — no JWT, no per-caller
identity, no scopes. Both fail CLOSED: if the relevant env var is unset
or empty, the endpoint returns `503`, never silently open.

### 1.1 End-user / Ask Veda API (`/api/*`)

Checked by `require_api_key()` (`main.py`):
- Header `X-API-Key: <key>`, **or**
- Header `Authorization: Bearer <key>`

Compared via `secrets.compare_digest()` against the `RAG_API_KEY`
environment variable. If `RAG_API_KEY` is unset/empty → `503
{"detail": "Knowledge service authentication is not configured."}`.
If the supplied key doesn't match → `401 {"detail": "Unauthorized."}`.

Applies to: `POST /api/ask`, `GET /api/stats`, `GET /api/conversations`,
`GET /api/conversations/{id}`, `DELETE /api/conversations/{id}`.
**`GET /api/health` requires no authentication at all** (by
inspection — it calls neither `require_api_key` nor `require_admin_key`).

### 1.2 Admin API (`/admin/*`, both legacy and canonical)

Checked by `require_admin_key()` (`main.py`):
- Header `X-Admin-API-Key: <key>` — **no Bearer fallback.**

Compared via `secrets.compare_digest()` against `RAG_ADMIN_API_KEY`. If
unset/empty → `503 {"detail": "Admin operations are not configured."}`.
If the supplied key doesn't match → `401 {"detail": "Unauthorized."}`.

Applies identically to **every** `/admin/intake/*` and every legacy
`/admin/discovery/*` / `/admin/ingest/*` route — same key, same check,
no per-endpoint variation. There is no separate "Discovery Engine
service account" vs. "admin user" distinction in code — both are
whoever holds `RAG_ADMIN_API_KEY`. **IMPLEMENTED — RUNTIME UNVERIFIED**
(logic verified by inspection; cannot execute an HTTP request against
it in this sandbox).

---

## 2. Lifecycle — states and transitions

`config.IngestionState` (str Enum), exact values:

```
discovered | pending_approval | approved | queued | processing | indexed | failed | removed
```

`config._ALLOWED_TRANSITIONS` (the single source of truth; enforced by
`db.transition_document_state()` on every write, not just suggested):

```
discovered       -> pending_approval
pending_approval -> approved | removed
approved         -> queued | removed
queued           -> processing | failed
processing       -> indexed | failed
indexed          -> removed
failed           -> queued | removed
removed          -> (terminal; no outbound transitions)
```

**No state ever moves backwards** except `failed -> queued` (an explicit
retry) and `indexed -> removed` (an explicit removal) — both one-way.
There is no `queued -> approved` or `processing -> queued` "un-advance."
A transition not in this table raises
`config.IllegalStateTransitionError` (a `ValueError` subclass — see §12
for why that distinction matters to the HTTP error contract).

**Confirmed: approval auto-queues.** `intake.approve_intake()` performs
`pending_approval -> approved` then immediately calls `queue_intake()`
(`approved -> queued`) in the same function call, before returning to
the caller. `approved` is written to the database and appears in the
`ingestion_audit` trail, but is never observable as the document's
*resting* state via any endpoint under normal operation — by the time
`POST /admin/intake/{id}/approve` returns, the state is already
`queued` (or further along, if running on the `inline` queue backend —
see §4). **IMPLEMENTED + TESTED**
(`test_register_approve_process_indexed_and_retrievable`).

**Confirmed: only `indexed` is retrievable.** `db.lexical_search()`'s
SQL restricts to documents whose `approval_status = 'indexed'` — this
is unchanged from Round 3 and was re-confirmed passing in
`test_pipeline_e2e.py` this round (`AN/removal` test: retrievable before
removal, not after) and in `test_intake_lifecycle.py`'s main flow test.
**IMPLEMENTED + TESTED.**

---

## 3. Source-type / provenance contract

`config.ALLOWED_SOURCE_TYPES = {"discovery", "direct_upload"}` — any
other value is rejected by `register_intake()` with
`InvalidSourceType` **before a document row is created** (not just
filtered out later). This is the Newsletter/editorial-isolation
mechanism: there is no value that lets Newsletter or Gemini-summary
content through, and nothing elsewhere in the codebase calls
`register_intake()` for that content in the first place. See §6 for
the honest caveat on what this allowlist does and doesn't catch.

**`source_type` is a computed property, not a physical column —
confirmed true in the current implementation.** `DocumentRecord.
source_type` (`db.py`) is:

```python
@property
def source_type(self) -> str:
    return "discovery" if self.ingestion_method == "discovery" else "direct_upload"
```

The actual stored column is still `ingestion_method`
(`'discovery'` / `'manual_upload'`), unchanged since Round 2/3. Every
Round-4 API request/response uses `source_type` with values
`'discovery'`/`'direct_upload'` — the translation happens in this one
property (reads) and in `register_intake()`'s
`ingestion_method = "discovery" if source_type == "discovery" else "manual_upload"`
line (writes). `db.list_documents_page(source_type=...)`'s filter does
the same translation at the SQL level. **IMPLEMENTED + TESTED**
(the computed property is exercised by every passing test that reads
`doc.source_type`).

Fields preserved per source type (`DocumentRecord`, all columns):

| Field | discovery | direct_upload |
|---|---|---|
| `discovery_candidate_id` | set | `null` |
| `uploaded_by` | `null` (unless explicitly passed) | set (actor, for PDF uploads) |
| `uploaded_at` | `null` unless `uploaded_by` set | set when `uploaded_by` set |
| `original_filename` | `""` unless provided | set for PDF uploads |
| `source_format` | extraction format (`text`/`markdown`/`pdf`) | extraction format |
| `extracted_text` | full extracted text, stored at registration | full extracted text, stored at registration |
| `source_url` | sanitized original URL | as provided, sanitized |

**Discovery's own internal IDs/metadata beyond `discovery_candidate_id`**
(whatever Discovery's payload shape looks like internally) are not
separately modeled — `migrations/0002_intake_lifecycle.sql` adds
`discovery_candidates.original_provenance JSONB` for this, but nothing
in `intake.py` currently writes to it. **NOT IMPLEMENTED** (column
exists; no code path populates it yet).

---

## 4. Redis / RQ contract

### 4.1 Configuration (`config.py`)

| Variable | Default | Purpose |
|---|---|---|
| `RAG_REDIS_URL` | `redis://localhost:6379/0` | Redis connection string |
| `RAG_RQ_QUEUE_NAME` | `cmt_veda_intake` | RQ queue name |
| `RAG_RQ_JOB_TIMEOUT_SECONDS` | `600` | Per-job timeout passed to `enqueue`/`enqueue_in` |
| `RAG_QUEUE_BACKEND` | `redis_rq` | `redis_rq` (production) or `inline` (dev/CI, no Redis) |
| `RAG_MAX_RETRIES` | `5` | Attempt ceiling (your confirmed decision #4) |
| `RAG_RETRY_BACKOFF_BASE_SECONDS` | `30` | Backoff base |

### 4.2 Enqueue behavior (`worker.enqueue_processing`)

- Attempt 1: `queue.enqueue(_rq_job_body, document_id, attempt, job_timeout=RQ_JOB_TIMEOUT_SECONDS)` — immediate.
- Attempt > 1: `queue.enqueue_in(timedelta(seconds=retry_backoff_seconds(attempt-1)), ...)` via rq-scheduler — delayed.
- `retry_backoff_seconds(attempt)` = `30 * 2^(attempt-1)` → **30s, 60s, 120s, 240s, 480s** for attempts 1–5.

### 4.3 Worker entry point

`python worker.py` → `run_worker()` → `rq.Worker([RQ_QUEUE_NAME]).work()`,
a separate OS process from the FastAPI app. The job body
(`_rq_job_body`) rebuilds `db`/`embedder`/`vector_store` from `config.py`
inside the worker process (never shares objects across the process
boundary with the API process) and calls the shared
`process_intake_job()` function.

### 4.4 Job payload

An RQ job carries exactly `(document_id: str, attempt: int)` — no other
data. Everything else (extracted text, metadata) is read fresh from
PostgreSQL by `process_intake()` inside the job, which is why Postgres
(not the job payload, not Redis) is the durable source of truth.

### 4.5 Retry orchestration (`worker.process_intake_job`)

```
try: intake.process_intake(db, embedder, vector_store, document_id, rq_job_id)
except Exception:
    if attempt < MAX_RETRIES:
        intake.queue_intake(db, document_id, actor="system:retry",
                             enqueue_fn=enqueue_fn, attempt=attempt + 1)
    # else: stays FAILED. Nothing further happens automatically.
```

This is identical whether running under real RQ or `InlineQueueAdapter`
— same function, same decision, same ceiling. **5th failed attempt
(`attempt == MAX_RETRIES == 5`) does not re-queue** — the document
stays `FAILED`, `documents.retry_count == 5`, and the only way forward
is an explicit admin action: `POST /admin/intake/{id}/retry`.
**IMPLEMENTED + TESTED** (`test_retry_policy_fails_five_times_then_stays_failed`).

### 4.6 What happens if Redis/RQ is unavailable

Fixed this round after being found while wiring up `main.py` (see
`intake.py`'s `queue_intake()`): if `enqueue_fn()` itself raises
(Redis unreachable, `rq` package missing, etc.), the document is
**not** left stranded in `queued` with no job record. `queue_intake()`
catches the exception, writes a `processing_jobs` row with a synthetic
job id and `status='failed'` (with the error message), transitions the
document to `failed` (the only legal exit from `queued` besides
`processing`), increments `retry_count`, and re-raises. The HTTP layer
(`main.py`'s `_enqueue_and_drain`) catches `(ImportError, ConnectionError,
OSError)` specifically and returns `503` with a message naming the
configured queue backend. **IMPLEMENTED + TESTED**
(`test_queue_intake_enqueue_failure_does_not_strand_document_in_queued`).

### 4.7 Tested vs. unverified, explicitly

| Claim | Status |
|---|---|
| Redis server reachable, responds to `PING` | **TESTED LIVE** — `redis-server` (v7.0.15) started in this sandbox, `redis-cli ping` → `PONG` |
| Redis `RPUSH`/`LRANGE`, `HSET`/`HGETALL` (the primitives RQ's queue is built on) | **TESTED LIVE** via `redis-cli` directly |
| The Python `redis` package | **NOT INSTALLABLE** in this sandbox (confirmed by a fresh `pip install` attempt this round — no network) |
| The Python `rq` package | **NOT INSTALLABLE**, same reason |
| `worker.enqueue_processing()` / `worker.run_worker()` (the real RQ code path) | **IMPLEMENTED — RUNTIME UNVERIFIED.** Written, reviewed, never executed against the live Redis above, because the Python client can't be installed here |
| `worker.InlineQueueAdapter` + `process_intake_job()` retry/backoff decision logic | **IMPLEMENTED + TESTED** — this is what every passing test actually runs through |
| Confirmed: Redis/RQ never touched in the synchronous retrieval path | **TESTED** by code inspection — `retrieval.py` has zero `redis`/`rq` import or reference; `POST /api/ask` never calls anything in `worker.py` or `intake.py` |

**Recommendation**, unchanged from the Round 4 status report: before
first production use, run `pip install redis rq` in a real
network-enabled environment and execute one real
`enqueue_processing()` → `run_worker()` round-trip. Nothing in the code
requires sandbox-specific behavior to do this — it's wired for it, just
never executed here.

---

## 5. RAG eligibility rules ("what can enter the index")

1. `source_type` must be `discovery` or `direct_upload` (§3) — anything
   else is rejected by `register_intake()` before a row exists.
2. Extracted text must be non-empty and ≥ 50 characters, or
   `ingestion_pipeline.IngestionValidationError` is raised ("suspiciously
   short — rejecting rather than registering likely-corrupted content").
   This check is unchanged from Round 3 and runs **before** the
   full-text-only check below.
3. **Full-text-only floor:** extracted text must be
   ≥ `config.MIN_FULL_TEXT_CHARS` (default **1000**, env
   `RAG_MIN_FULL_TEXT_CHARS`) or `intake.FullTextRuleViolation` is
   raised. **This is an honest length heuristic, not a real
   abstract-vs-full-text classifier** — the code's own docstring says
   so explicitly: a short genuine full-text note could theoretically be
   rejected; an unusually long abstract could theoretically pass. The
   structural guarantee against Newsletter/editorial content is rule 1
   (the allowlist), not this length check — nothing in the Newsletter
   pipeline calls `register_intake()` at all.
4. Only `indexed` documents are retrievable (§2) — enforced at the SQL
   level in `lexical_search()`, not just in application code.

All four are enforced **server-side**, inside `intake.register_intake()`
— there is no separate "has the admin UI already checked this" trust
boundary; calling the API directly with disallowed input hits the same
checks. **IMPLEMENTED + TESTED** (`test_full_text_only_rule_rejects_short_text`,
`test_disallowed_source_type_rejected_newsletter_isolation`).

---

## 6. Canonical endpoints — `/admin/intake/*`

All require `X-Admin-API-Key` (§1.2). All request/response bodies below
are the **exact** Pydantic models in `main.py` as implemented — field
names, types, and defaults are copied directly from the source, not
paraphrased. **Every endpoint in this section is IMPLEMENTED — RUNTIME
UNVERIFIED as HTTP** (no `fastapi` in this sandbox); the business logic
each one calls (`intake.py`) is **IMPLEMENTED + TESTED** directly.

### 6.1 `POST /admin/intake` — Register (text/markdown)

Content-Type: `application/json`. Not idempotent (each call creates a
new document; duplicate `(doi, pmid, title)` is rejected — see §6.11).

**Request** (`IntakeRegisterRequest`):
```json
{
  "source_type": "discovery",           // required: "discovery" | "direct_upload"
  "raw_text": "...",                    // required
  "format": "text",                     // optional, default "text"; "text" | "markdown"
  "title": "",                          // optional
  "authors": [],                        // optional, list
  "journal": "",
  "publication_date": null,             // optional string or null
  "doi": "",
  "pmid": "",
  "trial_id": "",
  "cmt_subtypes": [],
  "genes": [],
  "study_type": "",
  "source_tier": "unspecified",
  "source_url": "",
  "discovery_candidate_id": null,
  "knowledge_version": null,
  "uploaded_by": null,
  "original_filename": "",
  "actor": "admin1"                     // required
}
```

**Response `200`** (`IntakeDocumentResponse` — returned by every
`/admin/intake/*` endpoint except the bulk ones; see §6.9 for the full
field list, shown once here):
```json
{
  "document_id": "5e8f...-uuid",
  "source_id": "CMT-RAG-000042",
  "source_type": "discovery",
  "state": "pending_approval",
  "title": "...", "authors": [], "journal": "", "publication_date": null,
  "doi": "", "pmid": "", "trial_id": "", "cmt_subtypes": [], "genes": [],
  "study_type": "", "source_tier": "unspecified", "source_url": "",
  "discovery_candidate_id": null, "source_format": "text",
  "uploaded_by": null, "uploaded_at": null, "original_filename": "",
  "retry_count": 0, "intake_batch_id": null,
  "approved_by": null, "approved_at": null,
  "created_at": "2026-...", "updated_at": "2026-..."
}
```

**Errors:**
- `422` — `FullTextRuleViolation`, `InvalidSourceType`, or
  `IngestionValidationError` → `{"detail": "<message>"}`.
- `409` — `DuplicateDocumentError` → `{"detail": "<message>"}`.
- `422` (FastAPI's own, **different shape**) — malformed request body
  (missing `actor`, bad `source_type` literal, etc.):
  `{"detail": [{"loc": ["body", "source_type"], "msg": "...", "type": "..."}]}`
  — a **list**, not a string. Distinguishing these two 422 shapes
  matters for client error handling.

### 6.2 `POST /admin/intake/pdf` — Register (PDF upload)

Content-Type: `multipart/form-data`. Fields (all `Form(...)` except
`file`): `file` (required, the PDF), `source_type` (required),
`actor` (required), `title`, `authors_csv`, `journal`,
`publication_date`, `doi`, `pmid`, `trial_id`, `cmt_subtypes_csv`,
`genes_csv`, `study_type`, `source_tier` (default `"unspecified"`),
`source_url`, `discovery_candidate_id` — all optional strings, CSV
fields split on `,` and stripped. **Note:** unlike the legacy PDF
endpoint, this one does not take `x_admin_api_key` as anything other
than a header — same as every other admin route.

Response/errors: identical shape to §6.1. `uploaded_by` is set to
`actor` automatically when `source_type == "direct_upload"`;
`original_filename` is taken from the uploaded file's filename.

### 6.3 `GET /admin/intake` — List

Query parameters (note the exact names): `state_filter` (optional
string — one of the 8 `IngestionState` values, or `422` if unrecognized),
`source_type` (optional, `"discovery"` | `"direct_upload"`), `limit`
(default `50`), `offset` (default `0`). **There is no `state`
parameter** — it is named `state_filter` specifically to avoid
colliding with the app's internal `state` object; this is a real,
intentional naming detail Lovable needs to match exactly.

**Response `200`**: a JSON array of `IntakeDocumentResponse` (§6.1's
shape), most-recent-first (`ORDER BY created_at DESC`).

### 6.4 `GET /admin/intake/{document_id}` — Get one

**Response `200`**: one `IntakeDocumentResponse`. **`404`** if no such
document: `{"detail": "No such document."}`.

### 6.5 `PATCH /admin/intake/{document_id}` — Metadata correction

Content-Type: `application/json`. **Idempotent** in the HTTP sense (same
body → same result), but only legal while the document is
`pending_approval`.

**Request** (`IntakePatchRequest` — all fields optional; only fields
actually present in the request body are applied, via
`exclude_unset=True`):
```json
{
  "title": "Corrected Title",
  "authors": ["Smith J", "Doe A"],
  "journal": "...", "publication_date": "2025-01-01",
  "doi": "...", "pmid": "...", "trial_id": "...",
  "source_tier": "...", "source_url": "..."
}
```
`source_url`, if present and non-empty, is re-sanitized via
`url_safety.sanitize_reference_url()` before being stored.

**Response `200`**: the updated `IntakeDocumentResponse`.

**Errors:**
- `409` — document is not `pending_approval`
  (`IntakeConflictError`): `{"detail": "Cannot edit metadata: document ... is in state '...', not 'pending_approval'. ..."}`.
- `404` — no such document (`ValueError` from `db.update_document_metadata`/`get_document`).

Editing metadata **does not** change the document's state, and has no
effect on `source_type`/provenance fields (`uploaded_by`,
`discovery_candidate_id`, etc. are not patchable — only the fields
listed above are, per `db._METADATA_PATCH_COLUMNS` /
`db._INTAKE_EXTRA_COLUMNS`).

### 6.6 `POST /admin/intake/{document_id}/approve` — Approve (auto-queues)

**Request** (`IntakeActorRequest`): `{"actor": "admin1"}`.

**Response `200`** (`IntakeJobResponse` = `IntakeDocumentResponse` + `job_id`):
```json
{
  "job_id": "<rq job id, or inline-<uuid> on the inline backend>",
  "document_id": "...", "source_id": "...", "source_type": "...",
  "state": "queued",
  "...": "...all other IntakeDocumentResponse fields..."
}
```
**Important:** `state` reflects whatever the document's *actual* state
is by the time the handler responds — `"queued"` under the real
`redis_rq` backend, but potentially already `"indexed"` or `"failed"`
under the `inline` dev/CI backend, because `_enqueue_and_drain()`
synchronously drains the inline adapter before returning (§4). **Not
idempotent**: calling this twice on the same document raises `409`
the second time (§6.11/§2 — `pending_approval -> approved` is only
legal once).

**Errors:**
- `409` — `IllegalStateTransitionError` (e.g. document isn't
  `pending_approval`) or `IntakeConflictError` (already has an
  in-flight job).
- `404` — no such document.
- `503` — enqueue failed (Redis/RQ unreachable); the document has
  already been transitioned to `failed` server-side by the time this
  response is returned (§4.6) — the caller should `GET` the document
  to confirm, not assume it's still `pending_approval`.

### 6.7 `POST /admin/intake/{document_id}/reject`

**Request** (`IntakeRejectRequest`): `{"actor": "admin1", "reason": ""}` (`reason` optional, default `""`).

**Response `200`**: `IntakeDocumentResponse` with `"state": "removed"`.

**Errors:** `409` (illegal transition — e.g. already `indexed`), `404`.

### 6.8 `POST /admin/intake/{document_id}/retry`

The explicit admin retry decision #4 requires once a document has
exhausted `MAX_RETRIES` and sits `FAILED`. Legal from `failed` (also
technically legal from `approved`, since both map to the same
underlying `queue_intake()` call — but the admin UI's normal use case
is retrying a `failed` document).

**Request** (`IntakeActorRequest`): `{"actor": "admin1"}`.

**Response `200`** (`IntakeJobResponse`, same shape as §6.6). Starts a
fresh attempt-count cycle (`attempt=1`) for backoff purposes;
`documents.retry_count` itself is **cumulative and never reset** — it's
an audit counter across the document's whole life, not a per-cycle
budget.

**Errors:** same as §6.6 (`409` illegal transition / in-flight job,
`404`, `503` on enqueue failure).

### 6.9 `POST /admin/intake/bulk` — Bulk register

See §7 for the full schema. **Returns `200` even if every item fails**
— per-item failures are caught inside `intake.bulk_register_intake()`
and never raised as an HTTP error; only an unexpected infrastructure
failure (e.g. the DB connection itself dying) would surface as `500`.

### 6.10 `POST /admin/intake/bulk/approve` — Bulk approve

See §7. Same "always `200`, per-item status inside the body" behavior
as §6.9 — individual `approve_intake()` failures (including enqueue
failures) are caught per-item inside `intake.bulk_approve_intake()`.

### 6.11 Idempotency summary

| Endpoint | Idempotent? |
|---|---|
| `POST /admin/intake` | No — always creates a new document (unless duplicate-rejected) |
| `POST /admin/intake/pdf` | No, same reason |
| `GET /admin/intake`, `GET /admin/intake/{id}` | Yes (read-only) |
| `PATCH /admin/intake/{id}` | Yes (same body → same result), while `pending_approval` |
| `POST /admin/intake/{id}/approve` | **No** — second call on the same document is a `409` |
| `POST /admin/intake/{id}/reject` | No, same reason |
| `POST /admin/intake/{id}/retry` | **No** — refuses to double-queue a document with a non-terminal job (`IntakeConflictError`, `409`) |
| `POST /admin/intake/bulk`, `/bulk/approve` | No (each call processes its item list fresh; re-submitting the same list re-attempts each item, including items that already succeeded — there is no batch-level dedup) |

### 6.12 Endpoints proposed but NOT present in this router

- **No `GET /admin/intake/{id}/status`** as a separate endpoint —
  `GET /admin/intake/{id}` (§6.4) already returns `state` and
  `retry_count`, which is the status. **NOT IMPLEMENTED** as a distinct
  route.
- **No endpoint exposing `processing_jobs` rows** (attempt history,
  `rq_job_id`, timestamps, per-attempt errors). `db.get_latest_processing_job()`
  exists at the backend layer and is used internally
  (`_enqueue_and_drain`'s logging), but nothing serves it over HTTP.
  **NOT IMPLEMENTED.**
- **No `GET /admin/intake/batch/{batch_id}`** to re-fetch a batch's
  result later. `db.get_batch()` / `db.list_batch_items()` exist at the
  backend layer; the bulk endpoints (§6.9/6.10) return the batch result
  inline, once, at call time — there is no way to look it up again
  afterward. **NOT IMPLEMENTED.**

---

## 7. Bulk operations — exact schemas

### 7.1 `POST /admin/intake/bulk`

**Request** (`IntakeBulkRegisterRequest`):
```json
{
  "actor": "admin1",
  "items": [
    {
      "source_type": "direct_upload",
      "payload": { "raw_text": "...", "format": "text", "title": "Good One", "doi": "10.1/good1" },
      "uploaded_by": null,
      "original_filename": ""
    },
    {
      "source_type": "newsletter",
      "payload": { "raw_text": "...", "title": "Bad: wrong source_type" }
    }
  ]
}
```
`payload` is a free-form `dict` passed as `**kwargs` into
`IngestionInput(**item["payload"])` inside `intake.bulk_register_intake()`
— any key `IngestionInput` doesn't recognize raises a `TypeError`,
which is caught per-item and recorded as a failure (not a 422 — see
below).

**Response `200`** (`IntakeBulkResultResponse`):
```json
{
  "batch_id": "b3f1...-uuid",
  "batch_type": "intake",
  "item_count": 2,
  "success_count": 1,
  "failure_count": 1,
  "results": [
    {"document_id": "5e8f...-uuid", "status": "success"},
    {"status": "failed", "error": "source_type must be one of ['direct_upload', 'discovery'], got 'newsletter'. Rejected before any document row was created."}
  ]
}
```
`results` is a plain list of dicts in the **same order** as the
request's `items` — a successful item has `document_id`+`status`; a
failed item has `status`+`error` and no `document_id`. This exact
shape, including the per-item ordering guarantee and the absence of
`document_id` on failures, is confirmed by
`test_bulk_register_one_bad_item_does_not_block_others`
(**IMPLEMENTED + TESTED**).

A duplicate `(doi, pmid, title)` inside a bulk batch is caught exactly
like a single-item duplicate (`DuplicateDocumentError`, caught per-item,
recorded as `{"status": "failed", "error": "Duplicate of existing document ..."}`)
— it does not special-case bulk.

### 7.2 `POST /admin/intake/bulk/approve`

**Request** (`IntakeBulkApproveRequest`):
```json
{"actor": "admin1", "document_ids": ["5e8f...", "a1b2...", "00000000-0000-0000-0000-000000000000"]}
```

**Response `200`** (`IntakeBulkResultResponse`, `batch_type: "approve"`):
```json
{
  "batch_id": "c4a2...-uuid",
  "batch_type": "approve",
  "item_count": 3,
  "success_count": 2,
  "failure_count": 1,
  "results": [
    {"document_id": "5e8f...", "status": "success", "job_id": "<rq job id>"},
    {"document_id": "a1b2...", "status": "success", "job_id": "<rq job id>"},
    {"document_id": "00000000-0000-0000-0000-000000000000", "status": "failed", "error": "No such document: 00000000-0000-0000-0000-000000000000"}
  ]
}
```
A successful item includes `job_id`; a failed item does not.
**IMPLEMENTED + TESTED** (`test_bulk_approve_mixed_results`).

### 7.3 Partial-success guarantee

Both bulk functions loop with a **per-item `try`/`except`** — there is
no wrapping database transaction around the batch. One failing item
cannot roll back or mark any other item as failed/successful
incorrectly; this is structural (the code shape itself), not merely
observed in tests. The parent `intake_batches` row's `success_count`/
`failure_count` are plain counters incremented one at a time in
`db.record_batch_item()`, not derived after the fact.

---

## 8. Legacy routes — status

`/admin/discovery/register`, `/admin/discovery/approve`,
`/admin/discovery/reject`, `/admin/ingest/manual`,
`/admin/ingest/manual/pdf`, `/admin/ingest/remove`,
`/admin/ingest/status/{document_id}` — **all unchanged from Round 3,
byte-for-byte.** None of their handlers were edited this round; they
still call `discovery.approve_and_ingest()` /
`ingestion_pipeline.ingest_document()` internally, which remain
**synchronous** (the HTTP request blocks until indexing finishes, the
original Round-3 behavior).

**This was a deliberate choice, not an oversight or a partial
migration.** Rewiring, say, `discovery_approve` to call
`intake.register_intake()` + `intake.approve_intake()` internally
would silently change its response from "document is already
`indexed`" to "document is `queued`, will index later" — a real
behavioral change for whoever calls that endpoint today (Discovery
Engine), which should be a deliberate choice made by that caller's
owner, not something changed under them this round. **No legacy route
has been migrated or rewired to the new lifecycle.** All 10 of Round
3's own tests (`test_pipeline_e2e.py`) still pass unchanged, confirming
this. `/admin/intake/*` (§6) is the new canonical surface; new
integrations (Lovable, bulk tooling, PDF uploads) should target it.

---

## 9. Ask Veda / retrieval boundary

**Yes, a retrieval endpoint exists** — `POST /api/ask` (unchanged from
Round 2/3, not part of this round's work, documented here for
completeness since you asked explicitly):

- Auth: §1.1 (`X-API-Key` or `Authorization: Bearer`).
- Request (`AskRequest`): `question` (required), `user_type` (default
  `"patient"`), `app_role` (optional), `answer_length` (default
  `"detailed"`), `table_format` (default `"auto"`), `conversation_id`
  (optional, enables multi-turn), `user_id` (optional, trusted value
  from Veda's backend).
- Response (`AskResponse`): `answer`, `sources` (persona-filtered —
  bare `source_id` only for `student` persona, full metadata object for
  others), `knowledge_version`, `insufficient_evidence`,
  `conversation_id`, `rewritten_question`.
- Also present: `GET /api/stats`, `GET /api/health`,
  `GET/DELETE /api/conversations*`.

**Confirmed: Redis/RQ is not used anywhere in this path.** By direct
code inspection, `retrieval.py` has no `redis` or `rq` import, and
`POST /api/ask`'s handler never imports or calls anything from
`worker.py` or `intake.py`. Retrieval queries the already-`indexed`
corpus directly via `HybridRetriever`/`db.lexical_search()`/vector
store — exactly your confirmed decision #5. **IMPLEMENTED + TESTED**
(unchanged Round 3 behavior, re-confirmed passing this round via
`test_pipeline_e2e.py`'s 10/10).

---

## 10. Testing / verification matrix

| Component | Implemented | Function-level tested | HTTP-tested | Runtime-tested | Production-ready |
|---|---|---|---|---|---|
| Intake registration (`register_intake`) | YES | YES | NO | YES (SQLite) | NO — needs Postgres backend run for real |
| PDF upload (register path) | YES | YES (via text path; PDF byte-extraction tested in Round 3) | NO | UNVERIFIED (PDF→register_intake combo not re-run this round) | NO |
| Metadata PATCH | YES | YES | NO | YES (SQLite) | NO |
| Approval (auto-queue) | YES | YES | NO | YES (SQLite + InlineQueueAdapter) | NO |
| Automatic queueing (`queue_intake`) | YES | YES | NO | YES (SQLite + InlineQueueAdapter) | NO |
| Redis connectivity | YES (server) | N/A | N/A | YES (live `redis-cli`) | UNVERIFIED (Python client untested) |
| Python Redis integration (`redis` package) | YES (code) | NO | NO | NO — package not installable here | NO |
| RQ enqueue (`worker.enqueue_processing`) | YES (code) | NO | NO | NO — package not installable here | NO |
| RQ worker execution (`run_worker`) | YES (code) | NO | NO | NO | NO |
| PostgreSQL Python backend (`PostgresBackend`) | YES (code) | NO | NO | NO — `psycopg2` not installable here | NO |
| Postgres migration SQL | YES | N/A | N/A | YES (live `psql`, run twice, idempotent) | YES (schema itself) |
| Bulk registration | YES | YES | NO | YES (SQLite) | NO |
| Bulk approval | YES | YES | NO | YES (SQLite + InlineQueueAdapter) | NO |
| Rejection | YES | YES | NO | YES (SQLite) | NO |
| Retry (explicit, post-FAILED) | YES | YES | NO | YES (SQLite) | NO |
| Five-attempt limit | YES | YES | NO | YES (SQLite + InlineQueueAdapter) | NO |
| Enqueue-failure handling (no stranded QUEUED) | YES | YES | NO | YES (SQLite) | NO |
| Illegal-transition error mapping (409 vs 404) | YES | YES (at exception-type level) | UNVERIFIED (can't run FastAPI) | YES (at exception-type level) | NO |
| Indexed state / retrieval eligibility | YES | YES | NO | YES (SQLite) | NO |
| Newsletter / source-type isolation | YES | YES | NO | YES (SQLite) | NO |
| Authentication (`require_api_key`/`require_admin_key`) | YES (code) | N/A | NO — `fastapi` not installable here | NO | NO |
| FastAPI HTTP endpoints, all of `/admin/intake/*` | YES (code, `py_compile`-clean) | N/A | NO | NO | NO |
| Legacy `/admin/discovery/*`, `/admin/ingest/*` (regression) | YES (unchanged) | YES | NO | YES (SQLite, `test_pipeline_e2e.py` 10/10) | Same as Round 3 |
| `/api/ask` retrieval, Redis/RQ absence confirmed | YES (unchanged) | YES | NO | YES | Same as Round 3 |

---

## Summary for integration planning

### Safe for Veda-v1 / Lovable to integrate against now (contract-stable)

The **request/response schemas** in §6–§8 reflect the actual
implementation and are safe to build UI/integration code against —
field names, types, status codes, and error shapes will not change
without a new version of this document. This is **not** the same claim
as "safe to run in production": every `/admin/intake/*` endpoint is
runtime-unverified end-to-end (no `fastapi` in this sandbox), and the
real Postgres/Redis backends are unverified in Python. Integration code
(request builders, response parsers, TypeScript types, mock servers)
can be written against this contract now; it should not be pointed at
a live production deployment until the "Recommended next step" below
is done.

### Implemented but runtime-unverified (do this before first production use)

1. Run `pip install psycopg2-binary redis rq fastapi python-multipart uvicorn` in a real network-enabled environment (this sandbox genuinely cannot — confirmed by a fresh attempt this round).
2. Start the FastAPI app for real and exercise one full HTTP round-trip per endpoint in §6–§9 (not just the underlying Python functions, which are already tested).
3. Run `migrations/0002_intake_lifecycle.sql` against a real Postgres (already proven idempotent at the SQL level) and re-run `test_intake_lifecycle.py`'s equivalent assertions against `PostgresBackend` instead of `SqliteBackend`.
4. Start `python worker.py` against a real Redis and confirm one real `enqueue_processing()` → pickup → `process_intake()` round-trip, including a deliberate failure to confirm the 5-attempt/backoff timing for real (this sandbox only proved the *decision logic*, not real wall-clock backoff delays).

### No remaining contract blockers

The one real blocker found while writing this document — the
`IllegalStateTransitionError`/404-vs-409 mismatch (§2, §6.6–§6.8) — was
fixed in code this round (not merely documented as a known issue) and
is covered by a new passing test
(`test_illegal_transition_is_a_distinct_error_type_not_generic_valueerror`).
No other contradiction was found between router paths, Pydantic
models, `intake.py`, `db.py`, the migration, and the test suite.


---

## Addendum A — post-integration changes (additive; the Round 4 contract above is otherwise unchanged)

**A.1 Route order (bug fix).** `POST /admin/intake/bulk` and `POST /admin/intake/bulk/approve` are declared
before the dynamic `/{document_id}/...` routes, and `bulk` / `pdf` are reserved path segments: they can
never be interpreted as a `document_id` (a request that would fall through to a dynamic handler with a
reserved segment gets `404`). Previously `POST /admin/intake/bulk/approve` was captured by
`POST /admin/intake/{document_id}/approve` and failed with `500` on PostgreSQL (`invalid input syntax for
type uuid: "bulk"`). Request/response bodies of both endpoints are unchanged.

**A.2 Scientific content class (`content_type`) — new optional field, orthogonal to `source_type`.**
`source_type` (`discovery` | `direct_upload`) says how a document arrived; `content_type` says what kind of
evidence it is. Allowed: `research_paper` (default — existing clients are unaffected), `clinical_trial`,
`genetic_variant`, `guideline`, `consensus_statement`, `outcome_measure`. Anything else → `422`
(`newsletter_*` content is not a RAG source and has no content type).
- Accepted on `POST /admin/intake`, each bulk item's `payload`, and `POST /admin/intake/pdf` (form field).
- Returned on every intake response; filterable with `GET /admin/intake?content_type=...`.
- The full-text floor (`RAG_MIN_FULL_TEXT_CHARS`) applies to literature-style classes (`research_paper`,
  `guideline`, `consensus_statement`, `outcome_measure`). Structured records (`clinical_trial`,
  `genetic_variant`) are exempt — they are registry/database records, not the full text of a paper — but the
  50-character corrupted-content floor still applies.
- Ask Veda evidence blocks carry `source_kind: <content_type>` and the grounding rules require attributing
  claims to the right kind of source ("ClinVar classifies…" ≠ "a study reports…").

**A.3 Source provenance — new optional fields** on register requests (and returned on responses):
`source_format` (`pdf`|`xml`|`html`|`text`|`markdown` — the ORIGINAL representation; defaults to `format`),
`source_mime_type`, `source_content_hash`, `source_document_ref`, `extraction_status`. `source_url` and
`discovery_candidate_id` are unchanged. RAG indexes the **extracted text** whatever the origin:
XML- and HTML-derived scientific text is as valid as PDF, and `pdf_available` is never required.

**A.4 No content truncation in RAG.** `raw_text` is stored and chunked in full (no character cap other
than the 50 MB PDF-upload byte limit and normal HTTP body limits). Over-long paragraphs are split on
sentence/whitespace boundaries so no chunk exceeds `MAX_CHUNK_CHARS`. Entity extraction scans the whole text.

**A.5 Idempotent repair.** Re-registering a source whose existing duplicate is still `pending_approval`
with an empty `extracted_text` (left by the PostgreSQL insert bug below) fills the text/provenance in place
and returns the same `document_id`/`source_id`; no state change, no approval bypass. Any other duplicate is
still `409`.

**A.6 Migration `0003_scientific_taxonomy_provenance.sql`** (idempotent; applied by `deploy/migrate.sh`).
