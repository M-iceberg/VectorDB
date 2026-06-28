"""Pythonic wrapper around the C++ Engine (_vectordb).

Key differences from the raw Engine:
- IDs can be strings or ints. String IDs are mapped to internal uint32_t
  and the mapping is persisted to data_dir/id_map.json so it survives restarts.
- insert() accepts a single vector (1-D array) or a batch (2-D array, one
  row per vector), matched by a parallel list of ids.
- search() returns the original string/int IDs, not the internal uint32_t.
"""
import json
import pathlib
from typing import Any, Dict, List, Optional, Union

import numpy as np

from ._vectordb import Engine as _Engine


class VectorDB:
    def __init__(self, data_dir: str):
        self._data_dir = pathlib.Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._engine = _Engine(str(data_dir))
        self._map_path = self._data_dir / "id_map.json"
        # {collection: {str_id: int_id}}
        self._str_to_int: Dict[str, Dict[str, int]] = {}
        # {collection: {int_id: original_id}}   original_id may be str or int
        self._int_to_orig: Dict[str, Dict[int, Any]] = {}
        self._next_id: Dict[str, int] = {}
        self._load_map()

    # ------------------------------------------------------------------
    # ID mapping (only used for string IDs; int IDs pass through directly)
    # ------------------------------------------------------------------
    def _load_map(self):
        if self._map_path.exists():
            data = json.loads(self._map_path.read_text())
            for col, mapping in data.items():
                self._str_to_int[col] = {k: v for k, v in mapping.items()}
                self._int_to_orig[col] = {v: k for k, v in mapping.items()}
                self._next_id[col] = max(mapping.values(), default=-1) + 1

    def _save_map(self):
        self._map_path.write_text(json.dumps(self._str_to_int, indent=None))

    def _resolve_id(self, collection: str, user_id: Union[str, int]) -> int:
        if isinstance(user_id, int):
            return user_id
        col_map = self._str_to_int.setdefault(collection, {})
        rev_map = self._int_to_orig.setdefault(collection, {})
        if user_id not in col_map:
            int_id = self._next_id.get(collection, 0)
            col_map[user_id] = int_id
            rev_map[int_id] = user_id
            self._next_id[collection] = int_id + 1
            self._save_map()
        return col_map[user_id]

    def _original_id(self, collection: str, int_id: int) -> Any:
        return self._int_to_orig.get(collection, {}).get(int_id, int_id)

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------
    def create_collection(self, name: str, dimension: int, metric: str = "l2"):
        self._engine.create_collection(name, dim=dimension, metric=metric)

    def drop_collection(self, name: str):
        self._engine.drop_collection(name)
        self._str_to_int.pop(name, None)
        self._int_to_orig.pop(name, None)
        self._next_id.pop(name, None)
        self._save_map()

    def list_collections(self) -> List[Dict]:
        return self._engine.list_collections()

    # ------------------------------------------------------------------
    # Insert — single or batch
    #
    # insert("col", ids="a", vectors=vec)
    # insert("col", ids=["a","b"], vectors=np.array([[...],[...]]))
    # insert("col", ids=["a","b"], vectors=..., metadata=[{...},{...}])
    # ------------------------------------------------------------------
    def insert(self, collection: str, *, ids, vectors,
               metadata: Optional[Union[Dict, List[Dict]]] = None):
        # Normalise ids to a list
        if isinstance(ids, (str, int)):
            ids = [ids]

        # Normalise vectors to 2-D float32 array
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim == 1:
            vectors = vectors[np.newaxis, :]

        n = len(ids)
        if vectors.shape[0] != n:
            raise ValueError(
                f"len(ids)={n} does not match vectors.shape[0]={vectors.shape[0]}")

        for i, user_id in enumerate(ids):
            int_id = self._resolve_id(collection, user_id)
            if metadata is None:
                meta = None
            elif isinstance(metadata, list):
                meta = metadata[i]
            else:
                meta = metadata  # same dict for every vector
            self._engine.insert(collection, int_id, vectors[i], meta)

    # ------------------------------------------------------------------
    # Remove
    # ------------------------------------------------------------------
    def remove(self, collection: str, id: Union[str, int]):
        int_id = self._resolve_id(collection, id)
        self._engine.remove(collection, int_id)

    # ------------------------------------------------------------------
    # Search — returns original IDs
    # ------------------------------------------------------------------
    def search(self, collection: str, *, query, top_k: int = 10,
               ef_search: int = 64,
               filters: Optional[Dict] = None) -> List[Dict]:
        query = np.asarray(query, dtype=np.float32)
        results = self._engine.search(collection, query, top_k=top_k,
                                      ef_search=ef_search, filters=filters)
        return [{"id": self._original_id(collection, r["id"]),
                 "distance": r["distance"]}
                for r in results]

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------
    def checkpoint(self, collection: str):
        self._engine.checkpoint(collection)


def open(data_dir: str) -> VectorDB:
    """Open (or create) a VortexDB database. Returns a VectorDB instance."""
    return VectorDB(data_dir)
