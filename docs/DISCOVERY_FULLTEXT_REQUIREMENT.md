# Required Discovery change: do not truncate extracted full text at 20,000 characters

**Status: NOT implemented here.** This is the Discovery repository's logic; RAG v2 does not own it and was
deliberately not modified to work around it. RAG already accepts and indexes the complete text.

## Where the 20,000 comes from
`app/services/fulltext/extraction.py` (Discovery):

```python
MAX_EXTRACTED_CHARS = 20_000                       # line 48
...
truncated = len(text) > MAX_EXTRACTED_CHARS        # _clip(), line 65
clipped = text[:MAX_EXTRACTED_CHARS]               # line 66  <- the truncation
return ExtractionResult(success=True, text=clipped, char_count=len(clipped), truncated=truncated)
```

`service.py::acquire_full_text` stores that clipped string in `discovery_documents.extracted_text`
(and `extracted_char_count`), the document endpoint serves it, Veda forwards it unchanged (no slicing in
`discovery-api.server.ts`), and RAG indexes what it is given. The constant's own comment says it exists to
keep **Gemini editorial prompts** bounded — i.e. it is an editorial-prompt budget applied at the wrong layer.
The `truncated` flag is computed but never persisted, so the cut is invisible downstream.

## Required change (Discovery repo)
1. `extraction.py::_clip` → normalise whitespace only; **return the complete text**. Remove
   `MAX_EXTRACTED_CHARS` (or keep it only as a very large sanity bound that FAILS extraction loudly rather
   than silently clipping). `extracted_char_count` must be the real length.
2. `app/services/intelligence/editorial_service.py` (line ~91, `wrap_untrusted(document.extracted_text)`):
   apply the prompt-size budget **there**, for the Gemini call only (e.g. an `EDITORIAL_EXCERPT_CHARS`
   setting). The stored document and the RAG path must never see that excerpt limit.
3. `discovery_documents.extracted_text` is already `Text` (unbounded) — no migration needed.
4. **Re-extract already-acquired documents.** `storage.save()` kept the original bytes (the full XML/PDF/HTML)
   at `document_ref`; only the extracted text was clipped. Add a one-off re-extraction (read stored bytes →
   `extract_text` → update `extracted_text`/`extracted_char_count`) for rows with
   `extracted_char_count = 20000`, instead of re-downloading. `acquire_full_text` currently returns an
   existing `acquired` row untouched, so a normal re-run will not fix them.
5. Add a regression test: a >20,000-character XML/HTML/PDF fixture round-trips with
   `extracted_char_count == len(extracted_text) > 20000`.

## Until that ships
Documents registered in RAG from Discovery carry only the first 20,000 characters of the paper. They are genuine
original text, correctly marked, but incomplete; after the Discovery fix, re-send them (RAG's duplicate check
will report 409 for non-empty duplicates — remove/reject the incomplete record first, or ask for a
"replace text while pending_approval" option if you prefer that workflow).
