"""Day 22 pytest: metadata insert + filtered search."""
import sys, os, shutil, tempfile
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
import vectordb

DIM = 16
RNG = np.random.default_rng(42)


@pytest.fixture
def db(tmp_path):
    engine = vectordb.open(str(tmp_path))
    engine.create_collection("col", dim=DIM, metric="l2")
    yield engine


def make_vec(seed=None):
    r = np.random.default_rng(seed)
    return r.random(DIM, dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. insert without metadata still works (Day 21 regression)
# ---------------------------------------------------------------------------
def test_insert_no_metadata(db):
    v = make_vec(0)
    db.insert("col", 0, v)
    results = db.search("col", v, top_k=1)
    assert results[0]["id"] == 0
    assert results[0]["distance"] < 1e-5


# ---------------------------------------------------------------------------
# 2. string eq filter returns only matching nodes
# ---------------------------------------------------------------------------
def test_string_eq_filter(db):
    vecs = [make_vec(i) for i in range(6)]
    for i in range(3):
        db.insert("col", i,     vecs[i],   metadata={"category": "ml"})
    for i in range(3, 6):
        db.insert("col", i, vecs[i],   metadata={"category": "db"})

    query = make_vec(99)
    results = db.search("col", query, top_k=10, ef_search=50,
                        filters={"category": "ml"})
    ids = {r["id"] for r in results}
    assert ids <= {0, 1, 2}, f"filter leaked db nodes: {ids}"
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# 3. numeric $gte filter
# ---------------------------------------------------------------------------
def test_numeric_gte_filter(db):
    vecs = [make_vec(i) for i in range(5)]
    for i, v in enumerate(vecs):
        db.insert("col", i, v, metadata={"year": float(2020 + i)})

    results = db.search("col", make_vec(7), top_k=10, ef_search=50,
                        filters={"year": {"$gte": 2022.0}})
    ids = {r["id"] for r in results}
    # ids 2, 3, 4 have year 2022, 2023, 2024
    assert ids <= {2, 3, 4}, f"unexpected ids: {ids}"
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# 4. numeric $lte filter
# ---------------------------------------------------------------------------
def test_numeric_lte_filter(db):
    vecs = [make_vec(i) for i in range(5)]
    for i, v in enumerate(vecs):
        db.insert("col", i, v, metadata={"score": float(i * 10)})

    results = db.search("col", make_vec(8), top_k=10, ef_search=50,
                        filters={"score": {"$lte": 20.0}})
    ids = {r["id"] for r in results}
    # ids 0, 1, 2 have score 0, 10, 20
    assert ids <= {0, 1, 2}, f"unexpected ids: {ids}"
    assert len(ids) == 3


# ---------------------------------------------------------------------------
# 5. range filter $gte + $lte combined
# ---------------------------------------------------------------------------
def test_numeric_range_filter(db):
    vecs = [make_vec(i) for i in range(10)]
    for i, v in enumerate(vecs):
        db.insert("col", i, v, metadata={"price": float(i)})

    results = db.search("col", make_vec(55), top_k=10, ef_search=80,
                        filters={"price": {"$gte": 3.0, "$lte": 6.0}})
    ids = {r["id"] for r in results}
    assert ids <= {3, 4, 5, 6}, f"unexpected ids: {ids}"
    assert len(ids) == 4


# ---------------------------------------------------------------------------
# 6. combined string + numeric filter (AND semantics)
# ---------------------------------------------------------------------------
def test_combined_filter(db):
    vecs = [make_vec(i) for i in range(8)]
    for i, v in enumerate(vecs):
        cat = "A" if i < 4 else "B"
        db.insert("col", i, v, metadata={"cat": cat, "val": float(i)})

    # cat=="A" AND val>=2  → ids 2, 3
    results = db.search("col", make_vec(77), top_k=10, ef_search=80,
                        filters={"cat": "A", "val": {"$gte": 2.0}})
    ids = {r["id"] for r in results}
    assert ids == {2, 3}, f"expected {{2,3}}, got {ids}"


# ---------------------------------------------------------------------------
# 7. filter that matches nothing returns empty list
# ---------------------------------------------------------------------------
def test_filter_no_match(db):
    for i in range(5):
        db.insert("col", i, make_vec(i), metadata={"tag": "x"})

    results = db.search("col", make_vec(0), top_k=5,
                        filters={"tag": "nonexistent"})
    assert results == []


# ---------------------------------------------------------------------------
# 8. no filter still works after metadata inserts (regression)
# ---------------------------------------------------------------------------
def test_no_filter_after_metadata_inserts(db):
    vecs = [make_vec(i) for i in range(10)]
    for i, v in enumerate(vecs):
        db.insert("col", i, v, metadata={"x": float(i)})

    results = db.search("col", vecs[5], top_k=3, ef_search=50)
    assert results[0]["id"] == 5
    assert results[0]["distance"] < 1e-5


# ---------------------------------------------------------------------------
# 9. metadata survives checkpoint + reopen
# ---------------------------------------------------------------------------
def test_metadata_survives_restart(tmp_path):
    path = str(tmp_path)
    vecs = [make_vec(i) for i in range(6)]

    db1 = vectordb.open(path)
    db1.create_collection("col", dim=DIM)
    for i in range(3):
        db1.insert("col", i, vecs[i], metadata={"tag": "keep"})
    for i in range(3, 6):
        db1.insert("col", i, vecs[i], metadata={"tag": "drop"})
    db1.checkpoint("col")
    del db1

    db2 = vectordb.open(path)
    results = db2.search("col", make_vec(99), top_k=10, ef_search=50,
                         filters={"tag": "keep"})
    ids = {r["id"] for r in results}
    assert ids == {0, 1, 2}, f"expected {{0,1,2}} after restart, got {ids}"


# ---------------------------------------------------------------------------
# 10. unknown filter op raises ValueError
# ---------------------------------------------------------------------------
def test_unknown_filter_op_raises(db):
    db.insert("col", 0, make_vec(0), metadata={"x": 1.0})
    with pytest.raises((ValueError, RuntimeError)):
        db.search("col", make_vec(0), filters={"x": {"$ne": 1.0}})


# ---------------------------------------------------------------------------
# 11. bool metadata stored as "1"/"0", queryable as string
# ---------------------------------------------------------------------------
def test_bool_metadata(db):
    vecs = [make_vec(i) for i in range(4)]
    for i in range(2):
        db.insert("col", i,     vecs[i],   metadata={"active": True})
    for i in range(2, 4):
        db.insert("col", i, vecs[i],   metadata={"active": False})

    results = db.search("col", make_vec(99), top_k=10, ef_search=50,
                        filters={"active": "1"})
    ids = {r["id"] for r in results}
    assert ids == {0, 1}
