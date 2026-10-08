#!/bin/sh
# Live lifecycle check for one RAG document (e.g. CMT-RAG-000034's document_id).
#   RAG_ADMIN_API_KEY=... RAG_API_KEY=... ./deploy/verify_document.sh <document_id> ["question"]
# Reads keys from the environment (never printed). Approves ONLY if the document is
# pending_approval -- this is a human-approval step you are choosing to run, not automation.
set -eu
BASE="${RAG_BASE_URL:-http://127.0.0.1:8000}"
ID="${1:?usage: verify_document.sh <document_id> [question]}"
Q="${2:-What did the study find about cardiovascular autonomic neuropathy in Charcot-Marie-Tooth disease?}"
: "${RAG_ADMIN_API_KEY:?set RAG_ADMIN_API_KEY}"; : "${RAG_API_KEY:?set RAG_API_KEY}"
adm() { curl -fsS -H "X-Admin-API-Key: $RAG_ADMIN_API_KEY" -H 'Content-Type: application/json' "$@"; }
state() { adm "$BASE/admin/intake/$ID" | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["state"], d.get("content_type"), d.get("source_format"), sep=" | ")'; }

echo "before : $(state)"
if state | grep -q '^pending_approval'; then
  adm -X POST "$BASE/admin/intake/$ID/approve" -d '{"actor":"live-verification"}' >/dev/null
  echo "approve: requested"
fi
i=0
until state | grep -qE '^(indexed|failed)'; do
  i=$((i+1)); [ "$i" -gt 60 ] && { echo "TIMEOUT waiting for indexed (is rag-v2-worker running?)"; exit 1; }
  sleep 5
done
echo "after  : $(state)"
state | grep -q '^indexed' || { echo "NOT INDEXED -- docker logs rag-v2-worker --tail 80"; exit 1; }
echo "ask    :"
curl -fsS -H "X-API-Key: $RAG_API_KEY" -H 'Content-Type: application/json' "$BASE/api/ask" \
  -d "$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1], "user_type": "clinician"}))' "$Q")" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(json.dumps({k:d.get(k) for k in ("answer","cited_source_ids","citations") if k in d}, indent=2)[:2500])'
