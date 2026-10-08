"""
embeddings.py
-------------
Embedding backend abstraction.

  * `SentenceTransformerEmbeddings` — the real production embedder
    (`sentence-transformers/all-MiniLM-L6-v2`, matching the model already
    named in config.py / the original v1 service). Requires the
    `sentence-transformers` package and a network connection the first
    time it downloads model weights. **Not available in this sandbox** —
    no network. Status: IMPLEMENTED BUT NOT INTEGRATED.

  * `HashingTfidfEmbeddings` — a dependency-free, deterministic substitute
    used to actually exercise the ingestion/retrieval pipeline in this
    sandbox. It is built on scikit-learn's `HashingVectorizer`, which
    IS installed here and needs no network/model download. Be precise
    about what this is: a weighted bag-of-words vector (with a small
    dependency-free suffix-stripping stemmer applied before hashing, so
    "cause"/"causes"/"caused" aren't three unrelated tokens), not a
    learned semantic embedding. It captures lexical/term overlap well (so
    it's genuinely useful for testing retrieval logic, ranking, and
    diversification) but it will NOT capture synonymy or paraphrase the
    way MiniLM does. Do not deploy this to production as a replacement
    for the real embedding model — it exists purely so the pipeline in
    this repo is provably executable without network access.
"""

import re
from abc import ABC, abstractmethod
import numpy as np


_SUFFIX_RULES_LONG = ("ational", "ization", "fulness", "iveness", "ousness", "ing", "ment", "tion", "sion", "ance", "ence")
_SUFFIX_RULES_SHORT = ("ed", "es", "al", "ic")


def _light_stem_token(tok: str) -> str:
    """Crude, dependency-free suffix stripping — not a real Porter
    stemmer, just enough to stop "causes"/"caused"/"causing" from being
    three unrelated tokens to a bag-of-words vectorizer with no morphology
    awareness. Only used by HashingTfidfEmbeddings (the sandbox-only
    substitute embedder); irrelevant to the real production embedder.
    Strips at most ONE suffix — chaining rules (e.g. stripping "es" then
    stripping the resulting trailing "s" again) over-stems short words."""
    stripped = False
    if len(tok) > 6:
        for suf in _SUFFIX_RULES_LONG:
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                tok = tok[: -len(suf)]
                stripped = True
                break
    if not stripped and len(tok) > 4:
        for suf in _SUFFIX_RULES_SHORT:
            if tok.endswith(suf) and len(tok) - len(suf) >= 3:
                tok = tok[: -len(suf)]
                stripped = True
                break
    if not stripped and len(tok) > 3 and tok.endswith("s") and not tok.endswith("ss"):
        tok = tok[:-1]
    return tok


_WORD_RE = re.compile(r"[A-Za-z0-9]+")


_STOPWORDS = frozenset("""
the a an is are was were be been being of in on at to for and or but with
by from as that this these those it its also though which who whom what
where when how why not no do does did have has had will would can could
may might must shall should than then there their them they he she we you
your i
""".split())


def _stem_preprocessor(text: str) -> str:
    words = _WORD_RE.findall(text.lower())
    # Stopword removal before stemming: standard practice for TF-based
    # retrieval, and directly relevant here — without it, common function
    # words dilute the cosine similarity between a short query and a long
    # document even when the actual content words align well. This is a
    # general fidelity improvement to the substitute embedder, not a fix
    # tailored to any one test question.
    return " ".join(_light_stem_token(w) for w in words if w not in _STOPWORDS)


class EmbeddingBackend(ABC):
    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Returns an (n_texts, dim) L2-normalized float32 array."""
        ...

    @property
    @abstractmethod
    def dim(self) -> int: ...


class HashingTfidfEmbeddings(EmbeddingBackend):
    def __init__(self, n_features: int = 4096):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._vectorizer = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm=None,
            ngram_range=(1, 2), preprocessor=_stem_preprocessor,
        )
        self._n_features = n_features

    @property
    def dim(self) -> int:
        return self._n_features

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._n_features), dtype=np.float32)
        mat = self._vectorizer.transform(texts).toarray().astype(np.float32)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return mat / norms


class SentenceTransformerEmbeddings(EmbeddingBackend):
    """Real production embedder. Requires `sentence-transformers` +
    network access to fetch model weights on first use. Untested in this
    sandbox — see module docstring."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise RuntimeError(
                "SentenceTransformerEmbeddings requires the "
                "`sentence-transformers` package, which is not installed "
                "in this environment. Install it in your deployment venv."
            ) from e
        self._model = SentenceTransformer(model_name)
        import config
        if config.RAG_EMBEDDING_MAX_SEQ_LENGTH > 0:
            self._model.max_seq_length = config.RAG_EMBEDDING_MAX_SEQ_LENGTH
        self._dim = self._model.get_sentence_embedding_dimension()

    @property
    def dim(self) -> int:
        return self._dim

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._dim), dtype=np.float32)
        return self._model.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).astype(np.float32)


def get_embedding_backend(name: str) -> EmbeddingBackend:
    if name == "hashing_tfidf":
        return HashingTfidfEmbeddings()
    if name == "sentence_transformers":
        return SentenceTransformerEmbeddings()
    raise ValueError(f"Unknown embedding backend: {name}")
