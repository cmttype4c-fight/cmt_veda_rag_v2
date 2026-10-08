"""
test_vector_store_reload.py
---------------------------
Deployment-fix test: the API process and the RQ worker process share one
index directory. A reader instance ("API") must see vectors a separate
writer instance ("worker") persisted after the reader started, and a stale
instance that later appends must not discard the other writer's rows.

  * NumpyVectorStore: tested directly (real code, real files).
  * FaissVectorStore: tested against a minimal STAND-IN `faiss` module
    (brute-force inner product, numpy persistence) injected into
    sys.modules, because real faiss-cpu cannot be installed in the
    development sandbox. This exercises FaissVectorStore's own
    reload / atomic-persist / id-map logic, NOT FAISS itself. Real FAISS is
    unverified until run in the container image.

Run: python3 test_vector_store_reload.py   (or pytest)
"""

import os
import shutil
import sys
import tempfile
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vector_store
from vector_store import NumpyVectorStore, FaissVectorStore

DIM = 8


def _vec(i):
    v = np.zeros((1, DIM), dtype=np.float32)
    v[0, i % DIM] = 1.0
    return v


def _install_fake_faiss():
    class IndexFlatIP:
        def __init__(self, dim):
            self.dim = dim

    class IndexIDMap2:
        def __init__(self, base):
            self.dim = base.dim
            self.ids = np.zeros((0,), dtype="int64")
            self.vecs = np.zeros((0, base.dim), dtype="float32")

        @property
        def ntotal(self):
            return len(self.ids)

        def add_with_ids(self, vecs, ids):
            self.vecs = np.vstack([self.vecs, vecs])
            self.ids = np.concatenate([self.ids, ids])

        def search(self, q, k):
            sims = (self.vecs @ q.T).ravel()
            order = np.argsort(-sims)[:k]
            return sims[order].reshape(1, -1), self.ids[order].reshape(1, -1)

    def write_index(index, path):
        with open(path, "wb") as fh:
            np.savez(fh, vecs=index.vecs, ids=index.ids)

    def read_index(path):
        data = np.load(path)
        index = IndexIDMap2(IndexFlatIP(data["vecs"].shape[1]))
        index.vecs, index.ids = data["vecs"], data["ids"]
        return index

    mod = types.ModuleType("faiss")
    mod.IndexFlatIP, mod.IndexIDMap2 = IndexFlatIP, IndexIDMap2
    mod.write_index, mod.read_index = write_index, read_index
    sys.modules["faiss"] = mod


def _reader_sees_writer(make):
    d = tempfile.mkdtemp(prefix="cmt_vs_")
    try:
        path = os.path.join(d, "store")
        reader = make(path)            # API process: starts with an empty index
        assert reader.search(_vec(0)[0], 3) == []
        writer = make(path)            # worker process: separate instance
        writer.add(["chunk-a"], _vec(0))
        hits = reader.search(_vec(0)[0], 3)
        assert hits and hits[0][0] == "chunk-a", hits
        assert reader.size() == 1
        writer.add(["chunk-b"], _vec(1))
        assert {h[0] for h in reader.search(_vec(1)[0], 5)} == {"chunk-a", "chunk-b"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _stale_writer_keeps_other_rows(make):
    d = tempfile.mkdtemp(prefix="cmt_vs_")
    try:
        path = os.path.join(d, "store")
        stale = make(path)             # loaded while the index was empty
        worker = make(path)
        worker.add(["from-worker"], _vec(0))
        stale.add(["from-api-legacy-route"], _vec(1))   # must append on top, not overwrite
        fresh = make(path)
        assert fresh.size() == 2, fresh.size()
        assert {h[0] for h in fresh.search(_vec(0)[0], 5)} == {"from-worker", "from-api-legacy-route"}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_numpy_reader_sees_writer():
    _reader_sees_writer(lambda p: NumpyVectorStore(p, DIM))
    print("PASS: NumpyVectorStore reader sees a separate writer's rows")


def test_numpy_stale_writer_keeps_other_rows():
    _stale_writer_keeps_other_rows(lambda p: NumpyVectorStore(p, DIM))
    print("PASS: NumpyVectorStore stale instance appends without discarding other rows")


def test_faiss_logic_reader_sees_writer_standin():
    _install_fake_faiss()
    _reader_sees_writer(lambda p: FaissVectorStore(p, DIM))
    print("PASS: FaissVectorStore reload logic (stand-in faiss, not real FAISS)")


def test_faiss_logic_stale_writer_keeps_rows_standin():
    _install_fake_faiss()
    _stale_writer_keeps_other_rows(lambda p: FaissVectorStore(p, DIM))
    print("PASS: FaissVectorStore stale-writer logic (stand-in faiss, not real FAISS)")


_TESTS = [
    test_numpy_reader_sees_writer, test_numpy_stale_writer_keeps_other_rows,
    test_faiss_logic_reader_sees_writer_standin, test_faiss_logic_stale_writer_keeps_rows_standin,
]

if __name__ == "__main__":
    failures = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:
            failures += 1
            print(f"FAIL: {t.__name__}: {e!r}")
    print(f"\n{len(_TESTS) - failures}/{len(_TESTS)} passed")
    sys.exit(1 if failures else 0)
