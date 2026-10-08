# RAG v2 deployment

One image, three roles: **api** (uvicorn), **worker** (RQ, exactly one),
**migrate** (one-shot, idempotent). Redis is a fourth, internal-only service.
Nothing secret is stored in the repository or the image.

## Production environment variables (names only)

Supplied through the host env file `/opt/cmtveda/rag-v2/rag-v2.env`
(outside the repo, `chmod 600`; template: `deploy/rag-v2.env.example`).

| Variable | Required | Notes |
|---|---|---|
| `RAG_API_KEY` | yes | user API (`X-API-Key` / Bearer) |
| `RAG_ADMIN_API_KEY` | yes | admin API (`X-Admin-API-Key`). **The only admin-key name the app reads.** |
| `RAG_POSTGRES_DSN` | yes | URI or key=value. Host must be reachable from the container: use `host.docker.internal` for PostgreSQL on the VPS host |
| `RAG_GGUF_MODEL_PATH` | yes | absolute path under `/opt/cmtveda/rag/runtime/models/` |

Fixed by `docker-compose.yml` (not secrets, do not put placeholders in the env file):
`RAG_DB_BACKEND=postgres`, `RAG_EMBEDDING_BACKEND=sentence_transformers`,
`RAG_VECTOR_BACKEND=faiss`, `RAG_GENERATION_BACKEND=llama_cpp`,
`RAG_QUEUE_BACKEND=redis_rq`, `RAG_REDIS_URL`, `RAG_VECTOR_STORE_PATH`.

Optional tuning (see `config.py`): `RAG_KNOWLEDGE_VERSION`, `RAG_N_CTX`,
`RAG_N_THREADS`, `RAG_MAX_QUESTION_CHARS`, `RAG_RQ_JOB_TIMEOUT_SECONDS`,
`RAG_MAX_RETRIES`, `RAG_RETRY_BACKOFF_BASE_SECONDS`, `RAG_MIN_FULL_TEXT_CHARS`,
and the retrieval-tuning `RAG_*` variables.

### `RAG_ADMIN_Y`
Not read anywhere in the code or its history. It is obsolete/incorrect. Put
its value (if it was the admin key) under `RAG_ADMIN_API_KEY` and delete
`RAG_ADMIN_Y`. If `RAG_ADMIN_Y` is set while `RAG_ADMIN_API_KEY` is not,
startup fails with an explicit message.

## Startup validation
`validate_production_config()` runs first in the API lifespan and in the
worker. It rejects placeholders (e.g. `YOUR_..._DSN`), a malformed DSN, a
missing key, a missing GGUF file, and reports **all** problems at once. It
names variables and never prints values. `deploy/migrate.sh` also refuses a
placeholder DSN before touching the database.

## First-time setup on the VPS
```sh
sudo install -d -m 755 /opt/cmtveda/rag-v2/runtime/vector_store /opt/cmtveda/rag-v2/runtime/redis
sudo chown -R 10001:10001 /opt/cmtveda/rag-v2/runtime/vector_store   # container user
sudo install -m 600 deploy/rag-v2.env.example /opt/cmtveda/rag-v2/rag-v2.env
sudoedit /opt/cmtveda/rag-v2/rag-v2.env                              # replace every <...>
```
The DB role in `RAG_POSTGRES_DSN` must own the `cmt_veda_rag` schema objects
(the migration alters tables). PostgreSQL must accept connections from the
Docker bridge (`listen_addresses` and a `pg_hba.conf` entry for the docker subnet).

## Deploy
```sh
git pull                                  # corrected Platform1 commit
docker compose up -d --build              # builds, migrates, then starts api + worker + redis
./deploy/smoke_test.sh
```
Start order is enforced: `rag-v2-migrate` must exit 0 before the API/worker start.
Publishing is `127.0.0.1:8000 -> 8000`; mounts are
`/opt/cmtveda/rag-v2/runtime` (rw) and `/opt/cmtveda/rag/runtime/models` (ro).
Remove the old container first if its name collides: `docker rm -f rag-v2`.

## Constraints
- Run **one** worker (single writer of the FAISS index; RQ scheduler enabled for retries).
- Never `docker exec ... pip install`. Dependencies live in `requirements.txt`.

## Verification status
Not buildable in the development sandbox (no Docker daemon). Verified there:
validator and vector-store reload unit tests, `migrate.sh` against PostgreSQL 16,
the existing suites. **Unverified until the first VPS build:** the image build,
real FAISS / psycopg2 / RQ / llama.cpp paths, `/api/health` inside the container.
