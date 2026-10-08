#!/bin/sh
# Applies the PostgreSQL schema and migrations for RAG v2. Idempotent: safe
# to run on every deployment (schema.sql and migrations/*.sql are written
# with IF NOT EXISTS / duplicate_object guards).
#
#   * schema.sql is applied ONLY if cmt_veda_rag.documents does not exist yet
#     (fresh database). An existing production database is not re-run
#     against the base schema.
#   * Every migrations/*.sql file is then applied in filename order, in ONE
#     psql session holding an advisory lock, so two deployments can never
#     migrate concurrently.
#   * Stops at the first error (ON_ERROR_STOP).
#
# Needs: psql, and RAG_POSTGRES_DSN in the environment. The DSN is never
# printed. The connecting role must own the cmt_veda_rag objects (ALTER
# TABLE ... ADD COLUMN requires ownership).
set -eu

APP_DIR="${RAG_APP_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"

if [ -z "${RAG_POSTGRES_DSN:-}" ]; then
  echo "migrate: RAG_POSTGRES_DSN is not set." >&2
  exit 2
fi
case "$RAG_POSTGRES_DSN" in
  *YOUR_*|*CHANGE_ME*|*REPLACE_ME*|*'${'*)
    echo "migrate: RAG_POSTGRES_DSN contains an unresolved placeholder. Fix the env file; nothing was changed." >&2
    exit 2 ;;
esac

PSQL="psql -X -q -v ON_ERROR_STOP=1"

have_schema=$($PSQL "$RAG_POSTGRES_DSN" -tA -c "SELECT to_regclass('cmt_veda_rag.documents') IS NOT NULL")

{
  echo "SELECT pg_advisory_lock(727001);"
  if [ "$have_schema" != "t" ]; then
    echo "migrate: fresh database detected; applying schema.sql" >&2
    echo "\\i $APP_DIR/schema.sql"
  fi
  for f in $(ls "$APP_DIR"/migrations/*.sql 2>/dev/null | sort); do
    echo "migrate: applying $(basename "$f")" >&2
    echo "\\i $f"
  done
} | $PSQL "$RAG_POSTGRES_DSN" >/dev/null

echo "migrate: done." >&2
