#!/bin/sh
# Post-deploy smoke test. Run on the VPS after `docker compose up -d --build`.
#   ./deploy/smoke_test.sh            (uses http://127.0.0.1:8000)
# Reads RAG_ADMIN_API_KEY from the env file if present; never prints secrets.
set -eu
BASE="${RAG_BASE_URL:-http://127.0.0.1:8000}"
ENV_FILE="${RAG_ENV_FILE:-/opt/cmtveda/rag-v2/rag-v2.env}"
fail() { echo "SMOKE FAIL: $*" >&2; exit 1; }

echo "1/4 waiting for /api/health (up to 180s)"
i=0
until curl -fsS "$BASE/api/health" >/tmp/rag_health.json 2>/dev/null; do
  i=$((i+1)); [ "$i" -gt 36 ] && fail "health endpoint never answered; run: docker logs rag-v2 --tail 80"
  sleep 5
done
cat /tmp/rag_health.json; echo
grep -q '"ready": *true' /tmp/rag_health.json || fail "health answered but ready=false"

echo "2/4 worker container running"
[ "$(docker inspect -f '{{.State.Running}}' rag-v2-worker 2>/dev/null)" = "true" ] || fail "rag-v2-worker is not running"

echo "3/4 admin auth is configured (expects 401/403, not 503)"
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/admin/intake" -H 'X-Admin-API-Key: wrong-on-purpose' || true)
case "$code" in 401|403) echo "admin auth enforced ($code)";; 503) fail "admin key not configured (503): set RAG_ADMIN_API_KEY";; *) echo "note: /admin/intake returned $code";; esac

echo "4/4 no placeholder in container environment"
if docker exec rag-v2 env | grep -E '^RAG_[A-Z_]+=.*(YOUR_|CHANGE_?ME|REPLACE_?ME)' >/dev/null; then
  fail "a RAG_* variable still holds a placeholder"
fi
echo "SMOKE OK"
