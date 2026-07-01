"""
VectorDB wrapper for ann-benchmarks (https://github.com/erikbern/ann-benchmarks).

To use:
  1. Clone ann-benchmarks: git clone https://github.com/erikbern/ann-benchmarks
  2. Copy this file to ann-benchmarks/ann_benchmarks/algorithms/vectordb/module.py
  3. Copy config.yml  to ann-benchmarks/ann_benchmarks/algorithms/vectordb/config.yml
  4. Install VectorDB: pip install /path/to/VectorDB --no-build-isolation
  5. Run: python run.py --algorithm vectordb --dataset sift-128-euclidean
"""
import os
import shutil
import tempfile

import numpy as np

# ann-benchmarks base class
try:
    from ann_benchmarks.algorithms.base import BaseANN
except ImportError:
    class BaseANN:  # stub for local testing
        pass

import vectordb as vdb


class VectorDB(BaseANN):
    def __init__(self, metric: str, method_param: dict):
        self._metric = metric
        self._M = method_param.get("M", 16)
        self._ef_construction = method_param.get("efConstruction", 200)
        self._ef_search = 64
        self._tmpdir = None
        self._db = None

    def fit(self, X: np.ndarray):
        """Build the index from the training set."""
        N, dim = X.shape
        self._tmpdir = tempfile.mkdtemp(prefix="vectordb_ann_")
        self._db = vdb.open(self._tmpdir)

        metric = {"euclidean": "l2", "angular": "cosine", "ip": "ip"}.get(
            self._metric, "l2")
        self._db.create_collection("bench", dimension=dim, metric=metric)

        BATCH = 10_000
        for start in range(0, N, BATCH):
            end = min(start + BATCH, N)
            self._db.insert("bench",
                            ids=list(range(start, end)),
                            vectors=X[start:end].astype(np.float32))

    def set_query_arguments(self, ef_search: int):
        self._ef_search = ef_search

    def query(self, q: np.ndarray, n: int) -> list[int]:
        results = self._db.search("bench",
                                  query=q.astype(np.float32),
                                  top_k=n,
                                  ef_search=self._ef_search)
        return [int(r["id"]) for r in results]

    def batch_query(self, X: np.ndarray, n: int):
        self.res = [self.query(q, n) for q in X]

    def get_batch_results(self) -> list[list[int]]:
        return self.res

    def __str__(self):
        return (f"VectorDB(M={self._M}, ef_construction={self._ef_construction}, "
                f"ef_search={self._ef_search})")

    def __del__(self):
        if self._tmpdir and os.path.exists(self._tmpdir):
            shutil.rmtree(self._tmpdir, ignore_errors=True)
