#!/usr/bin/env python3
"""
scripts/migrate_corpus.py
---------------------------
Controlled migration/re-ingestion for an existing document corpus (final
hardening brief item 7: "Provide a migration/re-ingestion process for the
current corpus. Do not overwrite the existing working RAG index blindly.
Build a versioned v2 index and allow rollback.").

This has been run in this sandbox against a SYNTHETIC test directory (a
couple of .txt files + one real reportlab-generated PDF) — proving the
walk-directory -> classify-format -> ingest-each-file -> new-versioned-
index logic actually works end to end (see
tests/test_corpus_migration.py). It has NOT been run against your actual
11-document corpus, because that corpus isn't available in this sandbox.
Before using this for real: point --corpus-dir at your actual documents
directory and review the title/metadata inference below (filename-based
title guessing is a placeholder — supply a metadata sidecar/CSV instead
if you have real bibliographic data for the 11 documents, see
--metadata-csv).

Safety properties, by construction:
  * Writes to a NEW sqlite/vector-store path (or a NEW `knowledge_version`
    tag when using Postgres) — never overwrites the currently-serving
    index. The old index/DB files are completely untouched; rollback is
    "keep pointing production at the old path/version."
  * Each file is ingested through the exact same `ingest_document()`
    pipeline as Discovery/manual uploads — same validation, same
    chunking, same lifecycle. A migration is not a special/bypassed path.
  * A file that fails to ingest (bad PDF, duplicate, too short) is
    logged and skipped, not silently dropped — the script prints a
    summary of successes/failures at the end and exits non-zero if
    anything failed, so a migration run can be scripted/CI-gated.

Usage:
    python3 scripts/migrate_corpus.py \\
        --corpus-dir /path/to/existing/documents \\
        --db-backend sqlite --sqlite-path ./cmt_veda_rag_v2.db \\
        --vector-path ./vector_store_v2/store \\
        --knowledge-version v2-2026-09-11 \\
        --actor migration-script \\
        [--metadata-csv metadata.csv] [--dry-run]

For Postgres: --db-backend postgres --postgres-dsn "$RAG_POSTGRES_DSN"
(same DB, new `knowledge_version` tag — Postgres doesn't need a separate
file path the way sqlite/numpy do, but you should still validate the new
knowledge_version's documents before switching `RAG_KNOWLEDGE_VERSION` in
production config).
"""

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, ".")

from db import get_backend, DuplicateDocumentError
from embeddings import get_embedding_backend
from vector_store import get_vector_store
from ingestion_pipeline import IngestionInput, ingest_document, IngestionValidationError

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}


def load_metadata_csv(path: str) -> dict:
    """Optional sidecar CSV: filename,title,authors,journal,publication_date,doi,pmid,source_tier
    `authors` is semicolon-separated. Returns {filename: {...}}."""
    if not path:
        return {}
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            fname = row.pop("filename", None)
            if not fname:
                continue
            if row.get("authors"):
                row["authors"] = [a.strip() for a in row["authors"].split(";") if a.strip()]
            out[fname] = row
    return out


def build_ingestion_input(path: Path, metadata: dict, knowledge_version: str) -> IngestionInput:
    meta = dict(metadata.get(path.name, {}))
    title = meta.pop("title", None) or path.stem.replace("_", " ").replace("-", " ").title()

    ext = path.suffix.lower()
    if ext in PDF_EXTENSIONS:
        return IngestionInput(
            raw_bytes=path.read_bytes(), format="pdf", title=title,
            ingestion_method="manual_upload", knowledge_version=knowledge_version,
            **{k: v for k, v in meta.items() if v},
        )
    elif ext in TEXT_EXTENSIONS:
        return IngestionInput(
            raw_text=path.read_text(encoding="utf-8", errors="replace"),
            format="markdown" if ext == ".md" else "text", title=title,
            ingestion_method="manual_upload", knowledge_version=knowledge_version,
            **{k: v for k, v in meta.items() if v},
        )
    else:
        raise IngestionValidationError(f"Unsupported file extension '{ext}' for {path.name}")


def migrate(corpus_dir: str, db, embedder, vector_store, knowledge_version: str,
            actor: str, metadata: dict, dry_run: bool = False) -> dict:
    corpus_path = Path(corpus_dir)
    files = sorted(
        p for p in corpus_path.iterdir()
        if p.is_file() and p.suffix.lower() in (TEXT_EXTENSIONS | PDF_EXTENSIONS)
    )
    results = {"succeeded": [], "failed": [], "skipped_unsupported": []}

    all_files_in_dir = sorted(p for p in corpus_path.iterdir() if p.is_file())
    for p in all_files_in_dir:
        if p.suffix.lower() not in (TEXT_EXTENSIONS | PDF_EXTENSIONS):
            results["skipped_unsupported"].append(p.name)

    for path in files:
        print(f"Ingesting: {path.name}")
        if dry_run:
            print(f"  [DRY RUN] would ingest {path.name}")
            continue
        try:
            payload = build_ingestion_input(path, metadata, knowledge_version)
            doc = ingest_document(db, embedder, vector_store, payload, actor=actor)
            print(f"  -> INDEXED as {doc.source_id} (document_id={doc.document_id})")
            results["succeeded"].append((path.name, doc.source_id))
        except DuplicateDocumentError as e:
            print(f"  -> SKIPPED (duplicate): {e}")
            results["failed"].append((path.name, f"duplicate: {e}"))
        except IngestionValidationError as e:
            print(f"  -> FAILED (validation): {e}")
            results["failed"].append((path.name, f"validation: {e}"))
        except Exception as e:
            print(f"  -> FAILED (unexpected): {e}")
            results["failed"].append((path.name, f"unexpected: {e}"))

    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--corpus-dir", required=True)
    ap.add_argument("--db-backend", choices=["sqlite", "postgres"], default="sqlite")
    ap.add_argument("--sqlite-path", default="./cmt_veda_rag_v2.db")
    ap.add_argument("--postgres-dsn", default="")
    ap.add_argument("--vector-path", default="./vector_store_v2/store")
    ap.add_argument("--vector-backend", choices=["numpy", "faiss"], default="numpy")
    ap.add_argument("--embedding-backend", choices=["hashing_tfidf", "sentence_transformers"], default="hashing_tfidf")
    ap.add_argument("--knowledge-version", required=True)
    ap.add_argument("--actor", required=True)
    ap.add_argument("--metadata-csv", default="")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not Path(args.corpus_dir).is_dir():
        print(f"ERROR: --corpus-dir '{args.corpus_dir}' is not a directory.")
        sys.exit(2)

    print(f"=== Corpus migration: knowledge_version={args.knowledge_version} ===")
    print(f"Source directory: {args.corpus_dir}")
    print(f"NEW db: {args.db_backend} "
          f"({args.sqlite_path if args.db_backend == 'sqlite' else '[postgres, same DB, new knowledge_version tag]'})")
    print(f"NEW vector store: {args.vector_backend} at {args.vector_path}")
    print("(The existing production index/DB is NOT touched by this run.)\n")

    metadata = load_metadata_csv(args.metadata_csv)

    db = get_backend(args.db_backend, sqlite_path=args.sqlite_path, postgres_dsn=args.postgres_dsn)
    embedder = get_embedding_backend(args.embedding_backend)
    vector_store = get_vector_store(args.vector_backend, args.vector_path, embedder.dim)

    results = migrate(args.corpus_dir, db, embedder, vector_store, args.knowledge_version,
                       args.actor, metadata, dry_run=args.dry_run)

    print("\n=== Migration summary ===")
    print(f"Succeeded: {len(results['succeeded'])}")
    for name, sid in results["succeeded"]:
        print(f"  {name} -> {sid}")
    print(f"Failed/skipped: {len(results['failed'])}")
    for name, reason in results["failed"]:
        print(f"  {name}: {reason}")
    if results["skipped_unsupported"]:
        print(f"Unsupported file types (ignored): {results['skipped_unsupported']}")

    if hasattr(db, "close"):
        db.close()

    if results["failed"]:
        print("\nMigration completed WITH FAILURES — review before validating/cutting over.")
        sys.exit(1)
    print("\nMigration completed. VALIDATE against the new index/knowledge_version "
          "before pointing production traffic at it (run tests/test_pipeline_e2e.py-"
          "style regression questions against this new DB/vector-store path).")


if __name__ == "__main__":
    main()
