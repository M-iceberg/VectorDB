# Metadata Index Design

## What is MetadataIndex?

MetadataIndex is an in-memory index that maps metadata field values to node IDs. It enables filtered vector search: instead of returning the raw HNSW nearest neighbors, the engine can restrict results to nodes that match a predicate (e.g., `price < 100`, `category == "electronics"`).

MetadataIndex does not replace HNSW — it sits alongside it. HNSW finds approximate nearest neighbors by vector distance; MetadataIndex enforces metadata predicates on top of those candidates.

## Why Not Store Metadata in HNSW?

HNSW's graph structure is purpose-built for vector proximity. It knows nothing about scalar fields. Adding filter logic inside HNSW would require touching every layer of the graph traversal and break the clean separation between "find similar vectors" and "apply business rules."

Keeping them separate also means the two indexes can evolve independently. MetadataIndex can be replaced with a more sophisticated structure (bitmap index, column store) without touching HNSW.

## The Post-Filter Approach

Filtered search works in two phases:

```
1. HNSW search with inflated ef_search
       ↓
   candidates (more than top_k, to account for filtered-out nodes)
       ↓
2. MetadataIndex filter
       ↓
   final top_k results that satisfy the predicate
```

**Why inflate ef_search?** If you ask for top_k=10 but half the candidates fail the filter, you would get fewer than 10 results. Inflating ef_search (e.g., 3–10× the requested top_k) gives the filter enough candidates to work with. The exact multiplier depends on selectivity.

**Trade-off:** For very selective filters (few nodes pass), even a large ef_search might not yield enough candidates. A future improvement is a pre-filter path: intersect the MetadataIndex result set with HNSW traversal rather than filtering after the fact. For now, post-filter is simpler and correct for moderate selectivity.

## Three Field Types

### String fields

Stored as an inverted index: `field → value → list of NodeIds`.

```
string_idx["category"]["electronics"] = [0, 3, 7]
string_idx["category"]["furniture"]   = [1, 4]
string_idx["status"]["active"]        = [0, 1, 3]
```

Supports:
- **eq**: `query_eq("category", "electronics")` → direct hash lookup, O(1)
- **in**: call `query_eq` for each value and union the results at the caller level

### Numeric fields

Stored as a sorted list of `(value, NodeId)` pairs per field:

```
numeric_idx["price"] = [(9.99, 2), (19.99, 0), (49.99, 5), (99.99, 1)]
                          sorted by value ──────────────────────────────►
```

Supports:
- **gte / lte / range**: `query_range(lo, hi)` uses `lower_bound` + `upper_bound` to find the window in O(log n + k) where k is the number of results.

Both bounds are inclusive. `query_range("price", 10.0, 50.0)` returns nodes with `10.0 ≤ price ≤ 50.0`.

### Boolean fields

Stored as string fields with values `"1"` (true) and `"0"` (false). No separate structure needed.

## Data Structures

```
// Forward indices (query path)
unordered_map<field, unordered_map<value, vector<NodeId>>>  string_idx
unordered_map<field, sorted vector<(double, NodeId)>>       numeric_idx

// Reverse indices (remove path)
unordered_map<NodeId, vector<(field, value)>>  string_rev
unordered_map<NodeId, vector<(field, double)>> numeric_rev
```

The reverse indices exist to make `remove()` efficient. To understand why, consider what happens without them.

Say you inserted three nodes:

```
insert_string("color", "red",  id=0)
insert_string("color", "blue", id=1)
insert_string("color", "red",  id=2)
insert_numeric("price", 99.0,  id=0)
insert_numeric("price", 49.0,  id=2)
```

The forward index (string_idx / numeric_idx) looks like:

```
color → red  → [0, 2]
color → blue → [1]
price → [(49.0, 2), (99.0, 0)]
```

Now you want to remove id=0. You need to erase 0 from `color → red` and from `price`. But without a reverse index, you don't know which fields and values id=0 appears in — you'd have to scan every field and every value list looking for 0. With 100 fields that's 100 scans.

The reverse index solves this by recording, at insert time, exactly where each id was placed:

```
string_rev[0] = [("color", "red")]
numeric_rev[0] = [("price", 99.0)]
```

`remove(0)` then looks up these two entries directly and erases only the affected slots. Cost is O(k) where k is the number of metadata fields on that node — typically a small constant.

## remove() Walk-Through

```
remove(id=3):
  1. string_rev[3] = [("category", "electronics"), ("status", "active")]
     → erase 3 from string_idx["category"]["electronics"]
     → erase 3 from string_idx["status"]["active"]
     → delete string_rev[3]

  2. numeric_rev[3] = [("price", 49.99)]
     → binary search numeric_idx["price"] for value=49.99
     → linear scan for id=3, erase the entry
     → delete numeric_rev[3]
```

The binary search in step 2 is needed because multiple nodes can have the same numeric value. After locating the first entry with the target value, a short linear scan finds the exact (value, id) pair to remove.

## insert_numeric Ordering

`insert_numeric` maintains sorted order by inserting at the position returned by `lower_bound((value, id))`. This makes every insert O(n) due to the vector shift, but MetadataIndex is optimized for read-heavy workloads (many queries per insert). A future improvement for write-heavy workloads is a skip list or B-tree per field.

## What MetadataIndex Does NOT Do

- **Persistence**: MetadataIndex is currently rebuilt from WAL replay on every restart (Day 17). There is no snapshot format yet.
- **Compound predicates (AND/OR/NOT)**: the index returns candidate sets per predicate; the engine combines them. AND = intersection, OR = union, NOT = set difference against all known IDs.
- **Pre-filter path**: for very selective filters, scanning HNSW with post-filter wastes traversal time. A pre-filter path (intersect candidate set first, then navigate only to those nodes) is a future optimization.
- **Cardinality statistics**: there is no query planner. Choosing between pre-filter and post-filter based on estimated selectivity is future work.
