"""
entities.py
-----------
Lightweight, dependency-free entity extraction (section 5) used to:
  * improve retrieval (lexical boosts, section 10)
  * decide whether a question is "general" vs "specific" (sections 9, 15)
  * power the specificity guard in generation.py

This is intentionally a regex/lookup-table approach, not an NLP/NER model —
no new infrastructure, no model download, and it is fully auditable/testable.
It is used to IMPROVE retrieval, never to manufacture facts (section 5).
"""

import re
from dataclasses import dataclass, field

# Known CMT-relevant genes. Extend this list as the corpus grows; it does not
# need to be exhaustive to be useful as a retrieval signal.
KNOWN_GENES = {
    "PMP22", "MPZ", "GJB1", "MFN2", "SH3TC2", "GDAP1", "EGR2", "NEFL",
    "LITAF", "RAB7A", "HSPB1", "HSPB8", "DNM2", "YARS1", "GARS1", "BSCL2",
    "FGD4", "PRX", "FIG4", "SBF2", "NDRG1", "HK1", "AARS1", "MED25",
    "HDAC6", "DHTKD1", "TRPV4", "IGHMBP2",
}

# Known/likely CMT subtype tokens (case-insensitive match, word-boundary safe).
KNOWN_SUBTYPES = {
    "CMT1A", "CMT1B", "CMT1C", "CMT1D", "CMT1E", "CMT1F",
    "CMT2A", "CMT2A1", "CMT2A2", "CMT2B", "CMT2C", "CMT2D", "CMT2E",
    "CMT2F", "CMT2I", "CMT2J", "CMT2K", "CMT2L", "CMT2N", "CMT2P",
    "CMT4A", "CMT4B1", "CMT4B2", "CMT4C", "CMT4D", "CMT4E", "CMT4F",
    "CMT4H", "CMT4J", "CMTX1", "CMTX5", "CMTDIB", "CMTDIC", "CMTDID",
    "HNPP", "DSN", "CHN",
}

_GENE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(KNOWN_GENES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
_SUBTYPE_PATTERN = re.compile(
    r"\b(" + "|".join(sorted(KNOWN_SUBTYPES, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)
# HGVS-ish protein/coding mutation notation, e.g. p.Arg1109X, c.1234A>G
_MUTATION_PATTERN = re.compile(
    r"\bp\.[A-Za-z]{3}\d+[A-Za-z\*]{1,3}\b|\bc\.\d+[+-]?\d*[ACGT]>[ACGT]\b",
    re.IGNORECASE,
)
_PMID_PATTERN = re.compile(r"\bPMID:?\s*(\d{5,9})\b", re.IGNORECASE)
_DOI_PATTERN = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>]+\b")
_TRIAL_PATTERN = re.compile(r"\bNCT\d{8}\b", re.IGNORECASE)


@dataclass
class ExtractedEntities:
    genes: set = field(default_factory=set)
    subtypes: set = field(default_factory=set)
    mutations: set = field(default_factory=set)
    pmids: set = field(default_factory=set)
    dois: set = field(default_factory=set)
    trial_ids: set = field(default_factory=set)

    def is_empty(self) -> bool:
        return not (
            self.genes or self.subtypes or self.mutations
            or self.pmids or self.dois or self.trial_ids
        )

    def specific_terms(self) -> set:
        """Any of these being present means the query is 'specific' rather
        than a general CMT question (sections 9, 15)."""
        return self.genes | self.subtypes | self.mutations | self.trial_ids


def extract_entities(text: str) -> ExtractedEntities:
    if not text:
        return ExtractedEntities()
    return ExtractedEntities(
        genes={m.upper() for m in _GENE_PATTERN.findall(text)},
        subtypes={m.upper() for m in _SUBTYPE_PATTERN.findall(text)},
        mutations=set(_MUTATION_PATTERN.findall(text)),
        pmids=set(_PMID_PATTERN.findall(text)),
        dois=set(_DOI_PATTERN.findall(text)),
        trial_ids={m.upper() for m in _TRIAL_PATTERN.findall(text)},
    )


# Very small, keyword-based intent classifier (section 5). This only steers
# retrieval (source diversification, evidence gating) — it never overrides
# evidence, per the spec's explicit instruction.
_GENERAL_MARKERS = (
    "what is cmt", "what is charcot", "overview of cmt", "cmt disease",
    "causes of cmt", "symptoms of cmt", "how is cmt diagnosed",
    "what causes cmt", "cmt in general",
)


def classify_query_scope(question: str, ents: ExtractedEntities) -> str:
    """Returns 'general' or 'specific'.

    A question is 'specific' the moment it names a gene, subtype, mutation,
    or trial ID explicitly. Otherwise, broad/definitional phrasing without
    any named entity is treated as 'general'. This is the mechanism behind
    the specificity-protection rule in section 15.
    """
    if ents.specific_terms():
        return "specific"
    q = question.lower().strip()
    if any(marker in q for marker in _GENERAL_MARKERS):
        return "general"
    # Short, entity-free questions default to general; longer entity-free
    # questions are ambiguous but still treated as general since there is no
    # signal to narrow retrieval.
    return "general"
