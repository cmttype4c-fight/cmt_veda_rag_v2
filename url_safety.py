"""
url_safety.py
--------------
Validates URLs before they're allowed into `source_url` /
`public_reference_url` — the ONE field that's allowed to reach a
clinician/researcher response (section 25/Z of the original spec; item 5
of the final hardening brief: "Reject/strip direct PDF URLs from
public_reference_url").

A bibliographic landing page (e.g. a DOI resolver, journal abstract page,
PubMed page) is fine. A direct link to a PDF, or anything that looks like
it points at CMT Veda's own storage, is not — that would recreate the
exact document-download exposure the removal of `/files/*` was meant to
close, just via a different field.

This is defense in depth applied at TWO points:
  1. Ingestion time (`sanitize_reference_url`, called from
     ingestion_pipeline.py) — bad URLs never get persisted in the first
     place.
  2. Response time (`main.py`'s `_persona_filtered_source` also calls
     this) — so even if something bad already got into the database by
     another path (a manual DB edit, a future bug in another writer), it
     still can't reach a response.
"""

import re
from urllib.parse import urlparse

# Extensions/path patterns that mean "this points at a file, not a page".
_FILE_EXTENSION_RE = re.compile(r"\.(pdf|docx?|zip|tar|gz|rar|xlsx?)(\?|#|$)", re.IGNORECASE)
_SUSPICIOUS_PATH_TOKENS = (
    "/pdf/", "/download", "/attachment", "/files/", "/storage/",
    "/uploads/", "/raw/", "/blob/", "?download=", "&download=",
)
# Hosts that should never appear in a public reference URL — internal
# infrastructure leaking out would be its own exposure, separate from the
# PDF-download concern.
_BLOCKED_HOST_PATTERNS = (
    "localhost", "127.0.0.1", "internal", ".local",
    "s3.amazonaws.com", "storage.googleapis.com",  # generic object storage;
    # legitimate bibliographic hosts (pubmed, doi.org, journal sites) never
    # look like this — an object-storage URL showing up here is a strong
    # signal something upstream is pointing at the raw file, not a landing page.
)


def is_safe_reference_url(url: str) -> tuple[bool, str]:
    """Returns (is_safe, reason). A URL is safe only if it's a plausible
    bibliographic landing page: has a scheme+host, isn't a direct file
    link, and doesn't point at internal/object storage."""
    if not url or not url.strip():
        return True, "empty is fine (field is optional)"
    url = url.strip()

    try:
        parsed = urlparse(url)
    except ValueError:
        return False, "unparseable URL"

    if parsed.scheme not in ("http", "https"):
        return False, f"unsupported scheme '{parsed.scheme}'"
    if not parsed.netloc:
        return False, "no host"

    if _FILE_EXTENSION_RE.search(parsed.path or ""):
        return False, "URL points directly at a file (PDF/DOCX/etc.), not a landing page"

    full_lower = url.lower()
    for token in _SUSPICIOUS_PATH_TOKENS:
        if token in full_lower:
            return False, f"URL path looks like a file/download endpoint ('{token}')"

    host_lower = parsed.netloc.lower()
    for pattern in _BLOCKED_HOST_PATTERNS:
        if pattern in host_lower:
            return False, f"URL host looks like internal/object storage ('{pattern}')"

    return True, "ok"


def sanitize_reference_url(url: str) -> str:
    """Returns the URL unchanged if safe, or '' (stripped) if not. Never
    raises — a bad reference URL is a reason to drop the link, not to
    fail the whole ingestion/response."""
    safe, _ = is_safe_reference_url(url)
    return url if safe else ""
