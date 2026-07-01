"""Pythonic wrapper around the C++ Engine (_vectordb).

Key differences from the raw Engine:
- IDs are strings. The Engine manages the internal NodeId mapping.
- insert() accepts a single vector (1-D array) or a batch (2-D array, one
  row per vector), matched by a parallel list of ids.
- search() returns user_id strings in results.
"""
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ._vectordb import Engine as _Engine


class VectorDB:
    def __init__(self, data_dir: str):
        self._engine = _Engine(str(data_dir))

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    def create_collection(self, name: str, dimension: int, metric: str = "l2"):
        self._engine.create_collection(name, dim=dimension, metric=metric)

    def drop_collection(self, name: str):
        self._engine.drop_collection(name)

    def list_collections(self) -> List[Dict]:
        return self._engine.list_collections()

    # ------------------------------------------------------------------
    # Insert — single or batch
    #
    # insert("col", vectors=vec)                       # auto-assign IDs
    # insert("col", vectors=vec, ids="a")              # single with ID
    # insert("col", vectors=vecs, ids=["a","b"])       # batch with IDs
    # insert("col", vectors=vecs, metadata=[{...},...])
    #
    # Returns: list of effective IDs (user-provided or auto-assigned strings).
    # ------------------------------------------------------------------
    def insert(self, collection: str, *, vectors,
               ids=None,
               metadata: Optional[Union[Dict, List[Dict]]] = None) -> List[str]:
        # Normalise vectors to 2-D float32 array
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]
        n = vectors.shape[0]

        # Normalise ids: None → list of empty strings (engine auto-assigns)
        if ids is None:
            ids_list = [""] * n
        elif isinstance(ids, (str, int)):
            ids_list = [str(ids)]
        else:
            ids_list = [str(i) for i in ids]

        if len(ids_list) != n:
            raise ValueError(
                f"len(ids)={len(ids_list)} does not match vectors.shape[0]={n}")

        # Build metadata list (None entries mean no metadata for that vector).
        if metadata is None:
            metas = None
        elif isinstance(metadata, list):
            metas = [m if isinstance(m, dict) else None for m in metadata]
        else:
            metas = [metadata] * n

        # Single batch call: one fdatasync for the whole batch instead of one per vector.
        assigned = self._engine.insert_batch(
            collection, ids_list, vectors, metas)
        return assigned

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------
    def remove(self, collection: str, id: Union[str, int]):
        self._engine.remove(collection, str(id))

    # ------------------------------------------------------------------
    # Search — returns user_id strings
    # ------------------------------------------------------------------
    def search_batch(self, collection: str, *, queries,
                     top_k: int = 10, ef_search: int = 64,
                     num_threads: int = 0,
                     filters: Optional[Dict] = None) -> List[List[Dict]]:
        """Parallel batch search. queries is a 2-D float32 array (N × dim).
        Returns a list of N result lists, each sorted by ascending distance.
        num_threads=0 means hardware_concurrency()."""
        queries = np.asarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[np.newaxis, :]
        batch = self._engine.search_batch(
            collection, queries, top_k=top_k, ef_search=ef_search,
            num_threads=num_threads, filters=filters)
        return [[{"id": r["user_id"], "distance": r["distance"]} for r in row]
                for row in batch]

    def search(self, collection: str, *, query, top_k: int = 10,
               ef_search: int = 64,
               filters: Optional[Dict] = None) -> List[Dict]:
        query = np.asarray(query, dtype=np.float32)
        results = self._engine.search(collection, query, top_k=top_k,
                                      ef_search=ef_search, filters=filters)
        return [{"id": r["user_id"], "distance": r["distance"]}
                for r in results]

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def checkpoint(self, collection: str):
        self._engine.checkpoint(collection)


def open(data_dir: str) -> VectorDB:
    """Open (or create) a VortexDB database. Returns a VectorDB instance."""
    return VectorDB(data_dir)
