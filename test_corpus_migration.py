"""
tests/test_corpus_migration.py
---------------------------------
Automated version of the ad hoc migration test run during development
(final hardening brief, item 7). Builds a synthetic corpus directory (two
text files + one real reportlab-generated PDF + one unsupported .docx),
runs it through `scripts/migrate_corpus.py`'s `migrate()` function
directly, and confirms:
  - supported files (txt/md/pdf) are ingested and reach INDEXED;
  - the unsupported file is skipped, not silently mis-indexed;
  - the resulting NEW index is actually queryable;
  - nothing was written to the original repo's working directory.

Run with: python3 tests/test_corpus_migration.py
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from db import SqliteBackend
from embeddings import HashingTfidfEmbeddings
from vector_store import NumpyVectorStore
from retrieval import HybridRetriever, assess_evidence_sufficiency
from migrate_corpus import migrate


def _build_synthetic_corpus(corpus_dir: Path):
    (corpus_dir / "cmt_overview.txt").write_text(
        "Abstract\nCharcot-Marie-Tooth disease is a group of inherited "
        "peripheral neuropathies affecting motor and sensory nerves, "
        "causing progressive weakness and sensory loss. " * 4
    )
    (corpus_dir / "cmt_symptoms.md").write_text(
        "# Symptoms\nCommon symptoms of CMT include distal muscle "
        "weakness, foot deformities, hammertoes, and reduced tendon "
        "reflexes. " * 4
    )
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet
    doc = SimpleDocTemplate(str(corpus_dir / "cmt4c_paper.pdf"), pagesize=letter)
    styles = getSampleStyleSheet()
    story = [
        Paragraph("Abstract", styles["Normal"]), Spacer(1, 12),
        Paragraph("CMT4C is caused by biallelic mutations in SH3TC2, "
                  "most commonly p.Arg1109X.", styles["Normal"]),
    ]
    doc.build(story)
    (corpus_dir / "notes.docx").write_bytes(b"fake docx content, unsupported format")


def test_migration_end_to_end_with_real_pdf():
    tmpdir = tempfile.mkdtemp(prefix="cmt_veda_migration_test_")
    try:
        corpus_dir = Path(tmpdir) / "corpus"
        corpus_dir.mkdir()
        _build_synthetic_corpus(corpus_dir)

        repo_files_before = set(os.listdir("."))

        db = SqliteBackend(os.path.join(tmpdir, "v2.db"))
        embedder = HashingTfidfEmbeddings()
        vector_store = NumpyVectorStore(os.path.join(tmpdir, "vs", "store"), dim=embedder.dim)

        results = migrate(
            str(corpus_dir), db, embedder, vector_store,
            knowledge_version="v2-test", actor="test-migration", metadata={},
        )

        assert len(results["succeeded"]) == 3, f"expected 3 successes, got {results}"
        assert "notes.docx" in results["skipped_unsupported"]
        assert not results["failed"], f"unexpected failures: {results['failed']}"
        print(f"3 supported files migrated: {results['succeeded']}")
        print(f"Unsupported file correctly skipped: {results['skipped_unsupported']}")

        # the new index is actually queryable
        retriever = HybridRetriever(db, embedder, vector_store)
        final, scope, ents = retriever.retrieve_and_rank("What is CMT4C?")
        assessment = assess_evidence_sufficiency(final, question="What is CMT4C?")
        assert assessment.sufficient, "migrated PDF content should be retrievable"
        print("Migrated index is queryable: PASS")

        # the migration touched only the tmpdir, never the repo's own working directory
        repo_files_after = set(os.listdir("."))
        assert repo_files_before == repo_files_after, (
            f"migration left stray files in the repo root: "
            f"{repo_files_after - repo_files_before}"
        )
        print("Original repo/working directory untouched: PASS")

        db.close()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    try:
        test_migration_end_to_end_with_real_pdf()
        print("\n1/1 passed")
    except Exception as e:
        print(f"\nFAIL: {e}")
        sys.exit(1)
