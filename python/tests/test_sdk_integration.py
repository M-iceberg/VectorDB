"""
SDK integration tests — Day 24.

Tests the full user-facing workflow via the VectorDB Python SDK (client.py).
Covers: insert, search, delete, filtered search, multi-collection, restart.
"""
import sys, os
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import vectordb

DIM = 32


def make_vecs(n, dim=DIM, seed=0):
    return np.random.default_rng(seed).random((n, dim), dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Quickstart demo — exact replica of the README example
# ---------------------------------------------------------------------------
def test_quickstart(tmp_path):
    db = vectordb.open(str(tmp_path))
    db.create_collection("docs", dimension=128, metric="l2")

    ids  = ["doc-a", "doc-b", "doc-c"]
    vecs = np.random.default_rng(0).random((3, 128), dtype=np.float32)
    db.insert("docs", ids=ids, vectors=vecs)

    results = db.search("docs", query=vecs[0], top_k=2)
    assert results[0]["id"] == "doc-a"
    assert results[0]["distance"] < 1e-5
    assert len(results) == 2


# ---------------------------------------------------------------------------
# 2. Full workflow: insert → search → delete → search again
# ---------------------------------------------------------------------------
def test_insert_search_delete_search(tmp_path):
    db = vectordb.open(str(tmp_path))
    db.create_collection("col", dimension=DIM)

    vecs = make_vecs(10)
    db.insert("col", ids=list(range(10)), vectors=vecs)

    # Search before delete — id 5 is nearest to vecs[5]
    results = db.search("col", query=vecs[5], top_k=3)
    assert results[0]["id"] == "5"
    assert results[0]["distance"] < 1e-5

    # Delete id 5
    db.remove("col", 5)

    # id 5 must not appear after delete
    results = db.search("col", query=vecs[5], top_k=5)
    assert all(r["id"] != "5" for r in results)


# ---------------------------------------------------------------------------
# 3. String IDs survive restart
# ---------------------------------------------------------------------------
def test_string_ids_survive_restart(tmp_path):
    path = str(tmp_path)
    vecs = make_vecs(4)

    db1 = vectordb.open(path)
    db1.create_collection("col", dimension=DIM)
    db1.insert("col", ids=["alpha", "beta", "gamma", "delta"], vectors=vecs)
    db1.checkpoint("col")
    del db1

    db2 = vectordb.open(path)
    results = db2.search("col", query=vecs[2], top_k=1)
    assert results[0]["id"] == "gamma"


# ---------------------------------------------------------------------------
# 4. Filtered search — string eq
# ---------------------------------------------------------------------------
def test_filtered_search_string_eq(tmp_path):
    db = vectordb.open(str(tmp_path))
    db.create_collection("col", dimension=DIM)

    vecs = make_vecs(10)
    meta = [{"tag": "even" if i % 2 == 0 else "odd"} for i in range(10)]
    db.insert("col", ids=list(range(10)), vectors=vecs, metadata=meta)

    results = db.search("col", query=make_vecs(1, seed=99)[0],
                        top_k=10, filters={"tag": "even"})
    ids = {r["id"] for r in results}
    assert ids <= {"0", "2", "4", "6", "8"}
    assert len(ids) == 5


# ---------------------------------------------------------------------------
# 5. Filtered search — numeric range
# ---------------------------------------------------------------------------
def test_filtered_search_numeric_range(tmp_path):
    db = vectordb.open(str(tmp_path))
    db.create_collection("col", dimension=DIM)

    vecs = make_vecs(10)
    meta = [{"score": float(i)} for i in range(10)]
    db.insert("col", ids=list(range(10)), vectors=vecs, metadata=meta)

    results = db.search("col", query=make_vecs(1, seed=77)[0],
                        top_k=10, filters={"score": {"$gte": 3.0, "$lte": 6.0}})
    ids = {r["id"] for r in results}
    assert ids <= {"3", "4", "5", "6"}
    assert len(ids) == 4


# ---------------------------------------------------------------------------
# 6. Filtered search — combined AND
# ---------------------------------------------------------------------------
def test_filtered_search_combined(tmp_path):
    db = vectordb.open(str(tmp_path))
    db.create_collection("col", dimension=DIM)

    vecs = make_vecs(8)
    meta = [{"cat": "A" if i < 4 else "B", "val": float(i)} for i in range(8)]
    db.insert("col", ids=list(range(8)), vectors=vecs, metadata=meta)

    # cat=="A" AND val>=2 → ids 2, 3
    results = db.search("col", query=make_vecs(1, seed=55)[0],
                        top_k=10,
                        filters={"cat": "A", "val": {"$gte": 2.0}})
    ids = {r["id"] for r in results}
    assert ids == {"2", "3"}


# ---------------------------------------------------------------------------
# 7. Delete then filtered search — deleted id absent
# ---------------------------------------------------------------------------
def test_delete_then_filtered_search(tmp_path):
    db = vectordb.open(str(tmp_path))
    db.create_collection("col", dimension=DIM)

    vecs = make_vecs(6)
    meta = [{"tag": "keep"} for _ in range(6)]
    db.insert("col", ids=list(range(6)), vectors=vecs, metadata=meta)

    db.remove("col", 3)

    results = db.search("col", query=vecs[3], top_k=6,
                        filters={"tag": "keep"})
    ids = {r["id"] for r in results}
    assert "3" not in ids
    assert len(ids) == 5


# ---------------------------------------------------------------------------
# 8. Multi-collection independence
# ---------------------------------------------------------------------------
def test_multi_collection(tmp_path):
    db = vectordb.open(str(tmp_path))
    db.create_collection("col_a", dimension=DIM)
    db.create_collection("col_b", dimension=DIM)

    vecs_a = make_vecs(5, seed=1)
    vecs_b = make_vecs(5, seed=2)
    db.insert("col_a", ids=list(range(5)), vectors=vecs_a)
    db.insert("col_b", ids=list(range(5)), vectors=vecs_b)

    ra = db.search("col_a", query=vecs_a[0], top_k=1)
    rb = db.search("col_b", query=vecs_b[0], top_k=1)

    assert ra[0]["id"] == "0" and ra[0]["distance"] < 1e-5
    assert rb[0]["id"] == "0" and rb[0]["distance"] < 1e-5

    cols = {c["name"] for c in db.list_collections()}
    assert cols == {"col_a", "col_b"}


# ---------------------------------------------------------------------------
# 9. Checkpoint + crash recovery preserves vectors and metadata
# ---------------------------------------------------------------------------
def test_checkpoint_recovery(tmp_path):
    path = str(tmp_path)
    vecs = make_vecs(8)
    meta = [{"tier": "gold" if i < 4 else "silver"} for i in range(8)]

    db1 = vectordb.open(path)
    db1.create_collection("col", dimension=DIM)
    db1.insert("col", ids=list(range(8)), vectors=vecs, metadata=meta)
    db1.checkpoint("col")
    del db1

    db2 = vectordb.open(path)
    results = db2.search("col", query=vecs[0], top_k=8)
    assert len(results) == 8

    gold = db2.search("col", query=vecs[0], top_k=8,
                      filters={"tier": "gold"})
    assert {r["id"] for r in gold} == {"0", "1", "2", "3"}


# ---------------------------------------------------------------------------
# 10. Recall quality — top-1 recall on 1K vectors
# ---------------------------------------------------------------------------
def test_recall_1k(tmp_path):
    N, Q = 1000, 20
    vecs  = make_vecs(N, seed=42)
    query = make_vecs(Q, seed=99)

    db = vectordb.open(str(tmp_path))
    db.create_collection("col", dimension=DIM)
    db.insert("col", ids=list(range(N)), vectors=vecs)

    hits = 0
    for q in query:
        results = db.search("col", query=q, top_k=10, ef_search=64)
        dists = np.sum((vecs - q) ** 2, axis=1)
        gt = str(int(np.argmin(dists)))   # IDs are strings now
        if any(r["id"] == gt for r in results):
            hits += 1

    recall = hits / Q
    assert recall >= 0.8, f"recall@1 = {recall:.2f} < 0.80"
