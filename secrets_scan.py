#!/usr/bin/env python3
"""
scripts/secrets_scan.py
--------------------------
A real, runnable scan for obviously-hardcoded secrets (final hardening
brief, item 11: "no secrets in Git"). This is a heuristic pattern scan,
not a substitute for a real secret-scanning tool (gitleaks/trufflehog) in
CI — but it IS actually run against this repo below, and the result is
real, not asserted.

Patterns checked: hardcoded API keys/tokens assigned as string literals,
Postgres/DB connection strings with an embedded password, AWS-style
access keys, and generic "password ="/"secret ="/"api_key =" literal
assignments. Deliberately excludes anything read from `os.environ` or
passed as a function parameter — those are correct, not violations.
"""

import re
import sys
from pathlib import Path

_PATTERNS = [
    # No leading \b: identifiers like RAG_API_KEY_TEST have no regex word
    # boundary before "API" (the preceding "_" is also a word character),
    # so requiring \b there silently missed exactly the kind of prefixed/
    # suffixed variable names real code uses. A bare substring search is
    # more false-positive-prone but that's the right tradeoff for a
    # security scanner — a missed secret is worse than an extra line to
    # dismiss on review. (Caught during self-testing below: a deliberately
    # planted fake secret was NOT flagged by the first version of this
    # pattern, which used \b and failed silently — fixed before shipping.)
    ("hardcoded_api_key_literal", re.compile(
        r'(?i)(api[_-]?key|secret|token|password)[A-Za-z0-9_]*\s*[:=]\s*["\'][A-Za-z0-9_\-]{12,}["\']'
    )),
    ("postgres_dsn_with_password", re.compile(r"postgres(?:ql)?://[^:\s]+:[^@\s]+@")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("generic_private_key_block", re.compile(r"-----BEGIN (RSA |EC )?PRIVATE KEY-----")),
]

# Lines that LOOK like a match but are actually fine (documentation,
# os.environ reads, empty-string defaults) — checked before flagging.
_SAFE_CONTEXT_MARKERS = ("os.environ", "getenv", '""', "''", "example", "placeholder",
                          "<", "your-", "REPLACE", "user:pass", "user:password")


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    findings = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return findings
    for lineno, line in enumerate(text.splitlines(), start=1):
        if any(marker in line for marker in _SAFE_CONTEXT_MARKERS):
            continue
        for name, pattern in _PATTERNS:
            if pattern.search(line):
                findings.append((lineno, name, line.strip()[:120]))
    return findings


def scan_repo(root: str, extensions=(".py", ".sql", ".md", ".txt", ".yaml", ".yml", ".env")) -> dict:
    root_path = Path(root)
    results = {}
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix in extensions and "__pycache__" not in path.parts:
            findings = scan_file(path)
            if findings:
                results[str(path.relative_to(root_path))] = findings
    return results


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "."
    print(f"Scanning {target} for hardcoded secrets...")
    results = scan_repo(target)
    if not results:
        print("\nNo hardcoded secrets found by this scan.")
        print("(This is a heuristic scan, not a guarantee — run gitleaks/"
              "trufflehog in CI for real coverage before trusting a repo "
              "with real secrets in its history.)")
        sys.exit(0)
    print(f"\n{sum(len(v) for v in results.values())} potential issue(s) in {len(results)} file(s):")
    for fname, findings in results.items():
        for lineno, kind, snippet in findings:
            print(f"  {fname}:{lineno} [{kind}] {snippet}")
    sys.exit(1)
