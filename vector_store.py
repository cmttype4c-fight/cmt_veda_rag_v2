"""
vector_store.py
----------------
Vector index abstraction.

  * `FaissVectorStore` — the real production vector index. Requires the
    `faiss` package, which is **not installed in this sandbox** and
    cannot be installed (no network). Status: IMPLEMENTED BUT NOT
    INTEGRATED. FAISS natively supports incremental addition
    (`IndexIDMap.add_with_ids`), which is what section D ("ingest one
    document without rebuilding the entire corpus") requires — this
    wrapper is written to use that, not a full-corpus rebuild.

  * `NumpyVectorStore` — a real, persisted, incrementally-updatable
    vector store built on plain numpy (which IS installed here), used to
    actually run the ingestion/retrieval pipeline in this sandbox. It
    persists vectors + id list to disk (`.npy` / `.json`) so a query
    after a simulated "restart" (new process, same files) still works.
    Cosine similarity is computed with a single matrix-vector product —
    fine at the corpus sizes this repo can actually test with, but this
    is a brute-force search, NOT an ANN index. It should not be presented
    as a scalability answer for "thousands of papers / tens of thousands
    of chunks" (see DELIVERABLES.md section I) — that's what FAISS (or
    pgvector, if later benchmarked and justified) is for in production.

Both stores are intentionally "dumb" about lifecycle/removal — the
authoritative "is this chunk still allowed to surface" check lives in
db.py's `get_active_chunk_ids()` (section G), applied as a post-filter
after the vector search returns candidates. That way removal correctness
never depends on which vector backend is in use.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np


class VectorStore(ABC):
    @abstractmethod
    def add(self, ids: list[str], vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Returns [(id, similarity), ...], higher similarity = better."""
        ...

    @abstractmethod
    def size(self) -> int: ...


class NumpyVectorStore(VectorStore):
    def __init__(self, path: str, dim: int):
        self.path = Path(path)
        self.dim = dim
        self._ids_path = self.path.with_suffix(".ids.json")
        self._vecs_path = self.path.with_suffix(".vecs.npy")
        self.ids: list[str] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self._load()

    def _load(self):
        if self._ids_path.exists() and self._vecs_path.exists():
            self.ids = json.loads(self._ids_path.read_text())
            self.vectors = np.load(self._vecs_path)
            if self.vectors.shape[0] != len(self.ids):
                raise RuntimeError(
                    f"Corrupt vector store at {self.path}: "
                    f"{self.vectors.shape[0]} vectors but {len(self.ids)} ids."
                )

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._ids_path.write_text(json.dumps(self.ids))
        np.save(self._vecs_path, self.vectors)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        if vectors.shape[0] == 0:
            return
        # Incremental append — does NOT rebuild the existing rows, which is
        # the section-D requirement ("ingest one document without
        # rebuilding the entire corpus").
        self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
        self.ids.extend(ids)
        self._persist()

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self.vectors.shape[0] == 0:
            return []
        q = query_vector.astype(np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        sims = self.vectors @ q  # vectors are pre-normalized at add() time by the caller
        top_idx = np.argsort(-sims)[:k]
        return [(self.ids[i], float(sims[i])) for i in top_idx]

    def size(self) -> int:
        return len(self.ids)


class FaissVectorStore(VectorStore):
    """Real production vector store. Requires `faiss`, not installed in
    this sandbox. See module docstring. Untested here."""

    def __init__(self, path: str, dim: int):
        try:
            import faiss  # noqa: F401
        except ImportError as e:
            raise RuntimeError(
                "FaissVectorStore requires the `faiss` (or `faiss-cpu`) "
                "package, which is not installed in this environment."
            ) from e
        import faiss
        self._faiss = faiss
        self.path = Path(path)
        self.dim = dim
        index_file = self.path.with_suffix(".faiss")
        if index_file.exists():
            self._index = faiss.read_index(str(index_file))
        else:
            base = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine
            self._index = faiss.IndexIDMap2(base)
        self._id_map_path = self.path.with_suffix(".idmap.json")
        self._str_to_int: dict[str, int] = {}
        self._int_to_str: dict[int, str] = {}
        if self._id_map_path.exists():
            data = json.loads(self._id_map_path.read_text())
            self._str_to_int = data["str_to_int"]
            self._int_to_str = {int(k): v for k, v in data["int_to_str"].items()}

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._faiss.write_index(self._index, str(self.path.with_suffix(".faiss")))
        self._id_map_path.write_text(json.dumps({
            "str_to_int": self._str_to_int, "int_to_str": self._int_to_str,
        }))

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.shape[0] == 0:
            return
        int_ids = []
        for sid in ids:
            iid = len(self._str_to_int) if sid not in self._str_to_int else self._str_to_int[sid]
            self._str_to_int[sid] = iid
            self._int_to_str[iid] = sid
            int_ids.append(iid)
        # add_with_ids is an incremental op — no full-index rebuild needed
        # (section D).
        self._index.add_with_ids(vectors.astype("float32"), np.array(int_ids, dtype="int64"))
        self._persist()

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._index.ntotal == 0:
            return []
        q = query_vector.astype("float32").reshape(1, -1)
        sims, idxs = self._index.search(q, min(k, self._index.ntotal))
        out = []
        for sim, iid in zip(sims[0], idxs[0]):
            if iid == -1:
                continue
            out.append((self._int_to_str.get(int(iid), str(iid)), float(sim)))
        return out

    def size(self) -> int:
        return self._index.ntotal


def get_vector_store(backend: str, path: str, dim: int) -> VectorStore:
    if backend == "numpy":
        return NumpyVectorStore(path, dim)
    if backend == "faiss":
        return FaissVectorStore(path, dim)
    raise ValueError(f"Unknown vector backend: {backend}")
