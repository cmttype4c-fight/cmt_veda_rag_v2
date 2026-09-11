#!/usr/bin/env python3
"""
scripts/cleanup_stale_conversations.py
-----------------------------------------
Retention mechanism for conversation history. Patient/caregiver questions
stored for multi-turn support are a sensitive-data surface that doesn't
exist elsewhere in this system (the scientific RAG corpus itself stays
PHI-free) — this script is what actually enforces a retention window
rather than keeping every conversation forever by default.

This does the deletion for real (tested against SqliteBackend — see
tests/test_conversation.py). Running it on a schedule is still an
operational step you need to do: this is not a background job, it's a
script. Wire it to cron or a systemd timer, e.g.:

    # /etc/cron.d/cmt-veda-conversation-cleanup
    0 3 * * * cmtveda cd /path/to/rag && python3 scripts/cleanup_stale_conversations.py --older-than-days 90 --db-backend postgres --postgres-dsn "$RAG_POSTGRES_DSN"

Usage:
    python3 scripts/cleanup_stale_conversations.py --older-than-days 90 [--dry-run]
"""

import argparse
import sys

sys.path.insert(0, ".")

from db import get_backend


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--older-than-days", type=int, required=True,
                     help="Delete conversations whose last_activity_at is older than this many days.")
    ap.add_argument("--db-backend", choices=["sqlite", "postgres"], default="sqlite")
    ap.add_argument("--sqlite-path", default="./cmt_veda_rag.db")
    ap.add_argument("--postgres-dsn", default="")
    ap.add_argument("--dry-run", action="store_true",
                     help="Report what would be deleted without actually deleting. "
                          "NOTE: the current delete_stale_conversations() implementation "
                          "does not have a separate count-only mode — --dry-run here just "
                          "skips calling it and reminds you to add one if you need it "
                          "before running for real against production data.")
    args = ap.parse_args()

    if args.older_than_days < 1:
        print("ERROR: --older-than-days must be at least 1.")
        sys.exit(2)

    db = get_backend(args.db_backend, sqlite_path=args.sqlite_path, postgres_dsn=args.postgres_dsn)

    if args.dry_run:
        print(f"[DRY RUN] Would delete conversations with no activity in the last "
              f"{args.older_than_days} days. Re-run without --dry-run to actually delete.")
        if hasattr(db, "close"):
            db.close()
        return

    deleted = db.delete_stale_conversations(older_than_days=args.older_than_days)
    print(f"Deleted {deleted} conversation(s) with no activity in the last {args.older_than_days} days.")

    if hasattr(db, "close"):
        db.close()


if __name__ == "__main__":
    main()
