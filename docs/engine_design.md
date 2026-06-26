# Engine Design

## What is the Engine?

The Engine is the single entry point for all database operations. It owns one collection's worth of storage components — WAL, VectorFile, HnswIndex — and is responsible for keeping them in sync, making writes crash-safe, and recovering the correct state on restart.

Every operation a client sends (insert, remove, search) goes through the Engine. The Engine knows the ordering rules: write to the WAL first, then apply to memory. It also runs the checkpoint/recovery cycle that keeps WAL growth bounded.

## The Three Runtime Components

Each collection has three components that are always open and in use:

```
                   ┌─────────────────────────────────────────┐
                   │            Engine                        │
                   │                                         │
  insert(id, vec)──►  WAL          on disk, append-only log  │
                   │   ↓                                     │
                   │  HnswIndex    in memory, graph topology  │
                   │   ↓                                     │
                   │  VectorFile   on disk, mmap float array  │
                   └─────────────────────────────────────────┘

  checkpoint() calls GraphSerializer once to write graph.bin, then returns
```

**WAL** (`wal.log`) — every write is recorded here first, before anything else. Provides durability: if the process crashes, the WAL is the source of truth for what operations happened since the last checkpoint.

**HnswIndex** — the in-memory HNSW graph. Fast to search, volatile. Rebuilt from the checkpoint snapshot + WAL replay on every restart.

**VectorFile** (`vectors.vdb`) — mmap-backed flat array of float32 vectors. Persists raw vector data across restarts without any special recovery logic — mmap writes go through the OS page cache and survive crashes as long as the OS flushes them.

**GraphSerializer** is not a runtime component. It is a utility called exactly once per `checkpoint()` to snapshot the HnswIndex to `graph.bin`. It is not open between checkpoints.

## On-Disk Layout

Each collection lives in its own directory under `data_dir`:

```
data_dir/
  {collection_name}/
    schema.bin      — dim(8B) + metric(4B) + pad(4B) = 16 bytes
    wal.log         — append-only WAL records
    wal.log.base    — WAL base LSN sidecar (managed by Wal internally)
    vectors.vdb     — mmap vector storage
    graph.bin       — latest HNSW graph checkpoint (absent before first checkpoint)
```

`schema.bin` is written once at `create_collection()` time and never modified. On restart, the Engine scans `data_dir` for subdirectories, reads `schema.bin` in each to discover what collections exist, and recovers them.

## Write Path

For every insert:

```
1. Build WAL payload: [id: 4B][vec: dim × 4B]
2. wal.append(Insert, payload)
3. wal.sync()                   ← force to disk; client success only after this
4. index.insert(id, vec)        ← update in-memory graph
5. vf.append(vec)               ← persist to mmap file
```

The critical ordering: **WAL before memory, sync before success.**

If the process crashes after step 3 but before step 4, the WAL has the record and recovery will replay it. If it crashes after step 4 but before step 5, the WAL still has the record; VectorFile will catch up on next open (its own mmap persistence handles most cases, and the WAL covers the gap).

If it crashes before step 3 completes — meaning `sync()` never returned — the record is not guaranteed to be on disk. The client did not receive a success response, so it knows to retry.

For removes:

```
1. Build WAL payload: [id: 4B]
2. wal.append(Delete, payload)
3. wal.sync()
4. index.remove(id)             ← soft-delete (tombstone)
```

No VectorFile operation — vectors are never physically deleted (the slot is just orphaned).

## Checkpoint

A checkpoint captures the complete current state of the index to disk, then truncates the WAL. After a checkpoint, recovery only needs to replay the WAL records that arrived after the checkpoint — not the entire history.

```
checkpoint():
  1. GraphSerializer::serialize(index, "graph.bin")
     ← snapshot entire HnswIndex (nodes, neighbor lists, vectors, entry_point)
  2. lsn = wal.current_lsn()
     ← the LSN of the next record to be written (= "we've captured everything before this")
  3. wal.truncate_before(lsn)
     ← remove records with LSN < lsn; base_lsn sidecar updated atomically
```

After step 3, `wal.log` contains only records written after the checkpoint. The graph file captures everything before. The two together represent the complete state of the collection.

**Why serialize before truncate, not the other way around?**

The ordering is crash-safe because the snapshot is written before the WAL is shortened:

```
crash here (after serialize, before truncate):
  graph.bin has 0..3 ✓
  wal.log still has 0..3 ✓  (truncate never ran)
  recovery: load graph.bin, replay WAL — 0..3 skipped by idempotency check ✓

crash here (after truncate, before serialize) — WRONG ORDER, never done:
  graph.bin is missing or incomplete ✗
  wal.log has already dropped 0..3 ✗
  recovery: nothing to load → data loss ✗
```

Snapshot first means a crash during checkpoint always leaves a recoverable state: either the old WAL is still intact, or the new graph.bin is complete. The only cost of a crash between the two steps is that the WAL is not truncated — it will be truncated on the next checkpoint instead.

**Why hold the write lock during checkpoint?**

A write arriving between steps 1 and 3 would be in the WAL but not in graph.bin. That is fine — it will be replayed. But a write arriving between steps 1 and 2 that then gets assigned an LSN below `current_lsn()` would be truncated from the WAL and missing from graph.bin. The lock prevents this window.

## Recovery

On startup, for each collection:

```
1. Open WAL and VectorFile (both handle existing files gracefully)
2. If graph.bin exists:
     GraphSerializer::deserialize("graph.bin") → HnswIndex
   Else:
     Create empty HnswIndex
3. wal.replay(0, on_insert, on_delete)
   ← replays all records in the current WAL file
   ← "0" works because truncate_before already removed records below checkpoint LSN;
      the file only contains post-checkpoint records
```

**on_insert callback:**
```cpp
[&](uint32_t id, const float* vec, size_t dim) {
    if (index.node_layer(id) < 0)   // not already in index
        index.insert(id, vec);
}
```

**on_delete callback:**
```cpp
[&](uint32_t id) {
    index.remove(id);
}
```

The existence check in on_insert makes recovery idempotent. If recovery itself crashes halfway through WAL replay and restarts, the same WAL records are replayed again. Nodes already in the index (from the first partial replay) are skipped rather than double-inserted.

## The Checkpoint + WAL Lifecycle

```
time ────────────────────────────────────────────────────────►

inserts: 0  1  2  3  4  5  6  7  8  9  10  11  12  ...
                   ↑                    ↑
            checkpoint A          checkpoint B

graph.bin:  [captures 0..3]       [captures 0..10]
wal.log:    [4,5,...] then [11,12,...] (truncated at each checkpoint)
```

After checkpoint A: graph.bin has operations 0..3, WAL has 4, 5, 6, ...
After checkpoint B: graph.bin has operations 0..10, WAL has 11, 12, ...

On crash between checkpoints A and B:
- graph.bin has 0..3
- WAL has 4..whatever was written before crash
- Recovery: load graph.bin (gives 0..3) + replay WAL (gives 4..crash point) = full state

## What the Engine Does NOT Do (Yet)

- **Metadata filtering** (Day 16–17): the Engine has no MetadataIndex yet. `search()` returns raw HNSW results without filtering.
- **Thread-safe concurrent access** (Day 19): a `shared_mutex` stub exists but the locking strategy is coarse (one lock for all collections).
- **VectorFile sync on recovery**: during WAL replay, the Engine only updates HnswIndex. VectorFile may be missing vectors from the crash window (written to WAL but not flushed). Since HnswIndex stores its own vector copies, search works correctly. Full VectorFile sync on recovery is a future improvement.
- **Multiple collections per engine**: the current implementation supports it in principle (separate subdirectory per collection) but has not been load-tested with many collections.

## API

### `Engine(data_dir)`

Opens the engine. Scans `data_dir` for collection subdirectories, reads each `schema.bin`, and runs the recovery sequence for each collection found. Creates `data_dir` if it does not exist.

---

### `create_collection(schema)`

Creates a new collection with the given name, dim, and metric. Writes `schema.bin`, creates `wal.log` and `vectors.vdb`, and initializes an empty in-memory `HnswIndex`.

**Throws** if a collection with that name already exists or if `dim == 0`.

---

### `drop_collection(name)`

Removes the collection from memory and deletes its directory from disk.

---

### `insert(collection, id, vec)`

Writes a WAL Insert record, syncs to disk, inserts into HnswIndex, and appends to VectorFile. The operation is durable after `sync()` — if the process crashes after this call returns, the insert will be recovered on restart.

---

### `remove(collection, id)`

Writes a WAL Delete record, syncs to disk, and soft-deletes the node (tombstone). The node remains in the graph as a traversal intermediary but is excluded from search results.

---

### `search(req)`

Runs HNSW beam search. Returns up to `req.top_k` results sorted by ascending distance. Read-only — does not touch the WAL or VectorFile.

---

### `checkpoint(collection)`

Snapshots the collection's HnswIndex to `graph.bin` and truncates the WAL up to the current LSN. After this call, recovery needs to replay only the WAL records written after this point.

Should be called periodically to bound WAL growth and reduce recovery time on restart.
