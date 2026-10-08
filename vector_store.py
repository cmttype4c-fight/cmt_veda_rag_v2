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
import os
import threading
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


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def _mtimes(*paths: Path) -> tuple:
    """Change signature for a set of files; (-1) for a missing file."""
    out = []
    for p in paths:
        try:
            st = p.stat()
            out.append((st.st_mtime_ns, st.st_size))
        except FileNotFoundError:
            out.append((-1, -1))
    return tuple(out)


# Cross-process note (deployment): in production the API process and the
# RQ worker process are separate OS processes sharing one index directory.
# The worker appends (add -> persist); the API only searches. Without the
# reload-on-change logic below, the API would keep serving its start-up
# snapshot and newly INDEXED documents would not be retrievable until the
# API restarted. Writes are atomic (temp file + os.replace) so a reader
# never sees a half-written file. Run ONE worker: concurrent writers are
# not coordinated.

class NumpyVectorStore(VectorStore):
    def __init__(self, path: str, dim: int):
        self.path = Path(path)
        self.dim = dim
        self._ids_path = self.path.with_suffix(".ids.json")
        self._vecs_path = self.path.with_suffix(".vecs.npy")
        self.ids: list[str] = []
        self.vectors = np.zeros((0, dim), dtype=np.float32)
        self._lock = threading.Lock()
        self._sig = None
        self._load()

    def _load(self):
        if self._ids_path.exists() and self._vecs_path.exists():
            ids = json.loads(self._ids_path.read_text())
            vectors = np.load(self._vecs_path)
            if vectors.shape[0] != len(ids):
                # Writer is between its two file replacements; keep the
                # previous snapshot and try again on the next call.
                if self._sig is None:
                    raise RuntimeError(
                        f"Corrupt vector store at {self.path}: "
                        f"{vectors.shape[0]} vectors but {len(ids)} ids."
                    )
                return
            self.ids, self.vectors = ids, vectors
        self._sig = _mtimes(self._ids_path, self._vecs_path)

    def _reload_if_changed(self):
        if _mtimes(self._ids_path, self._vecs_path) != self._sig:
            self._load()

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_vecs = self._vecs_path.with_name(self._vecs_path.name + ".tmp")
        with open(tmp_vecs, "wb") as fh:
            np.save(fh, self.vectors)
        # Write vectors first, ids last: a reader that sees the new ids file
        # always finds vectors at least as new.
        os.replace(tmp_vecs, self._vecs_path)
        _atomic_write_text(self._ids_path, json.dumps(self.ids))
        self._sig = _mtimes(self._ids_path, self._vecs_path)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if len(ids) != vectors.shape[0]:
            raise ValueError("ids and vectors length mismatch")
        if vectors.shape[0] == 0:
            return
        # Incremental append — does NOT rebuild the existing rows, which is
        # the section-D requirement ("ingest one document without
        # rebuilding the entire corpus").
        with self._lock:
            self._reload_if_changed()  # append on top of the latest on-disk state
            self.vectors = np.vstack([self.vectors, vectors.astype(np.float32)])
            self.ids.extend(ids)
            self._persist()

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        with self._lock:
            self._reload_if_changed()
            vectors, ids = self.vectors, self.ids
        if vectors.shape[0] == 0:
            return []
        q = query_vector.astype(np.float32)
        qn = np.linalg.norm(q)
        if qn > 0:
            q = q / qn
        sims = vectors @ q  # vectors are pre-normalized at add() time by the caller
        top_idx = np.argsort(-sims)[:k]
        return [(ids[i], float(sims[i])) for i in top_idx]

    def size(self) -> int:
        with self._lock:
            self._reload_if_changed()
            return len(self.ids)


class FaissVectorStore(VectorStore):
    """Real production vector store. Requires `faiss-cpu`. Persists to
    <path>.faiss + <path>.idmap.json; see the cross-process note above
    (API reads, single RQ worker writes). Real `faiss` is not importable
    in the development sandbox; the reload/persist ordering logic is
    exercised in test_vector_store_reload.py against a stand-in module,
    NOT against real FAISS."""

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
        self._index_path = self.path.with_suffix(".faiss")
        self._id_map_path = self.path.with_suffix(".idmap.json")
        self._lock = threading.Lock()
        self._sig = None
        self._index = None
        self._str_to_int: dict[str, int] = {}
        self._int_to_str: dict[int, str] = {}
        self._load()

    def _load(self):
        faiss = self._faiss
        # Read the id map BEFORE the index. The writer replaces the id map
        # first and the index second, so an index that is newer than the id
        # map we read can never be observed.
        str_to_int, int_to_str = {}, {}
        if self._id_map_path.exists():
            data = json.loads(self._id_map_path.read_text())
            str_to_int = data["str_to_int"]
            int_to_str = {int(k): v for k, v in data["int_to_str"].items()}
        if self._index_path.exists():
            index = faiss.read_index(str(self._index_path))
        else:
            index = faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim))  # inner product on normalized vectors == cosine
        self._index, self._str_to_int, self._int_to_str = index, str_to_int, int_to_str
        self._sig = _mtimes(self._index_path, self._id_map_path)

    def _reload_if_changed(self):
        if _mtimes(self._index_path, self._id_map_path) != self._sig:
            self._load()

    def _persist(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Id map first (a superset mapping is harmless to a reader holding
        # the older index), then the index, each via temp file + os.replace.
        _atomic_write_text(self._id_map_path, json.dumps({
            "str_to_int": self._str_to_int, "int_to_str": self._int_to_str,
        }))
        tmp_index = self._index_path.with_name(self._index_path.name + ".tmp")
        self._faiss.write_index(self._index, str(tmp_index))
        os.replace(tmp_index, self._index_path)
        self._sig = _mtimes(self._index_path, self._id_map_path)

    def add(self, ids: list[str], vectors: np.ndarray) -> None:
        if vectors.shape[0] == 0:
            return
        with self._lock:
            self._reload_if_changed()  # append on top of the latest on-disk state
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
        with self._lock:
            self._reload_if_changed()
            index, int_to_str = self._index, self._int_to_str
            if index.ntotal == 0:
                return []
            q = query_vector.astype("float32").reshape(1, -1)
            sims, idxs = index.search(q, min(k, index.ntotal))
        out = []
        for sim, iid in zip(sims[0], idxs[0]):
            if iid == -1:
                continue
            out.append((int_to_str.get(int(iid), str(iid)), float(sim)))
        return out

    def size(self) -> int:
        with self._lock:
            self._reload_if_changed()
            return self._index.ntotal


def get_vector_store(backend: str, path: str, dim: int) -> VectorStore:
    if backend == "numpy":
        return NumpyVectorStore(path, dim)
    if backend == "faiss":
        return FaissVectorStore(path, dim)
    raise ValueError(f"Unknown vector backend: {backend}")
