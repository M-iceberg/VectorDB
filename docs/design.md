# VortexDB — System Design

A vector database built from scratch. This document covers the architecture, key design decisions, and trade-offs.

---

## What it does

Stores float32 vectors with optional metadata. Answers approximate nearest-neighbor (ANN) queries: given a query vector, return the k most similar vectors by L2, cosine, or inner-product distance. Supports metadata filters (string equality, numeric range) and crash-safe writes.

**Non-goals (deliberate):** no distributed coordination, no replication, no horizontal sharding. Single-node, single-process.

---

## Architecture

```
Python SDK (client.py)
    │ pybind11 zero-copy numpy
    ▼
Engine  ── shared_mutex (search=read, writes=exclusive)
    ├── HnswIndex     in-memory HNSW graph + SIMD distance
    ├── VectorFile    mmap-backed flat float32 array
    ├── WAL           append-only durability log
    └── MetadataIndex in-memory inverted + sorted index
```

Each collection is independent. All state lives under `data_dir/{collection_name}/`.

---

## Key design decisions

### 1. HNSW for the index

HNSW (Hierarchical Navigable Small World) is a proximity graph. Insert builds a multi-layer graph; search does beam search starting from the top layer. O(log N) average query time, tunable recall via `ef_search`.

Alternatives considered:
- **Flat (brute force):** O(N) query, 100% recall. Fine at N<10K, unusable at 1M.
- **IVF (inverted file):** faster to build, weaker recall at low `nprobe`. Requires training a k-means quantizer.
- **LSH:** simpler, lower recall, harder to tune. Not used in any major production system today.

HNSW is the industry standard (hnswlib, faiss-HNSW, all cloud vector DBs). Chosen because recall/speed trade-off is well-studied and graph structure maps naturally to cache-friendly flat arrays.

### 2. Flat arrays for graph storage

Original: `std::vector<std::vector<float>>` for vectors, `std::vector<std::vector<NodeId>>` for neighbor lists.

Problem: each `vector<float>` is a heap allocation. 100K vectors = 100K separate allocations scattered in memory. HNSW search jumps through the graph randomly — each pointer dereference is a potential cache miss (~100ns DRAM latency).

Fix: one flat `float* vecs_flat_` with stride `dim`, one flat `uint32_t* adj0_` for layer-0 adjacency in CSR format. Address of vector `i` = `vecs_flat_ + i * dim` (one multiply-add, ~1ns). Prefetch all neighbor vectors before the distance loop.

Result: **+164% QPS** from this change alone (1,669 → 4,407).

### 3. Write-ahead log for crash safety

Every insert/delete is written to `wal.log` and `fdatasync()`'d before touching any in-memory state. If the process crashes mid-operation, recovery replays the WAL.

WAL record: `[crc32 | payload_length | type | timestamp_us | payload]`. CRC catches corruption; incomplete tail records (partial write at crash) are silently discarded.

Checkpoint: serialize HNSW graph + metadata index to disk, write checkpoint LSN, truncate WAL before that LSN. Bounds WAL size and recovery time. Recovery = load snapshot + replay WAL records since snapshot LSN.

Recovery time is O(M × ef_construction × log N) per WAL record — the same as the original insert — because replay re-inserts into the live HNSW graph. At dim=128, replay speed is ~4–11 Kv/s depending on current index size.

### 4. Metadata as post-filter

Metadata (string fields, numeric fields) stored in a separate in-memory `MetadataIndex`:
- String fields: inverted index `map<string, set<NodeId>>`
- Numeric fields: sorted array of `(value, NodeId)` pairs, binary search for range queries

Applied *after* HNSW beam search, not during graph traversal. HNSW returns `ef_search` candidates; candidates not matching the filter are discarded; top-k of survivors returned.

**Trade-off:** simple to implement. Breaks down at very low selectivity. At 1% selectivity (1% of vectors pass the filter), recall drops to ~70% because ef_search candidates mostly don't pass the filter and the true nearest neighbors among the eligible set are never reached. Production systems address this with per-segment indexes or hybrid HNSW+IVF approaches.

### 5. SIMD distance computation

Distance function auto-detected at compile time via CMake:
- x86: AVX2 (`distance_avx2.cpp`) — 8 floats/cycle, 256-bit registers
- ARM: NEON (`distance_neon.cpp`) — 4 floats/cycle, 128-bit registers
- Fallback: scalar (`distance_naive.cpp`)

After flat array optimization: distance compute is 51% (ARM) to 61% (x86) of search time. The higher x86 fraction reflects AVX2's efficiency — each instruction processes more data, so graph traversal overhead shrinks proportionally.

---

## Write path

```
insert(id, vec, meta)
  1. make_insert_payload(id, vec, meta)      → binary blob
  2. wal.append(Insert, payload)             → write to wal.log
  3. wal.sync()                              → fdatasync()
  4. hnsw_index.insert(vec)                 → HNSW graph update
  5. vector_file.write(node_id, vec)        → mmap write
  6. metadata_index.insert(node_id, meta)   → in-memory update
```

If the process crashes after step 3 but before step 4, WAL replay re-applies the insert. If it crashes before step 3, the operation is lost — client must retry.

## Read path

```
search(query, top_k, ef_search, filters)
  1. Build filter allowlist from MetadataIndex (if filters provided)
  2. hnsw_index.search(query, ef_search) → candidate list (up to ef_search results)
  3. Filter candidates against allowlist
  4. Sort survivors by distance, return top_k
```

## Recovery path

```
Engine(data_dir)  [constructor, called on startup]
  for each collection directory:
    1. load schema.bin
    2. if graph.bin exists: GraphSerializer::deserialize() → HnswIndex
    3. if metadata.bin exists: MetadataSerializer::deserialize() → MetadataIndex
    4. read checkpoint LSN from wal.log.base
    5. wal.replay(checkpoint_lsn, on_insert, on_delete)
       → re-insert/delete each WAL record into HnswIndex and MetadataIndex
```

---

## Thread safety

Single `std::shared_mutex` per engine. `search()` acquires a shared lock (multiple concurrent reads allowed). `insert()`, `remove()`, `create_collection()`, `drop_collection()`, `checkpoint()` acquire exclusive locks. Coarse-grained but correct. Bottleneck at high write concurrency — see Phase 2 below.

---

## What's missing (deliberate)

**No distributed coordination.** Single node only. A distributed system would need:
- Consistent hashing or range partitioning to shard vectors across nodes
- A consensus layer (Raft/Paxos) for metadata operations (create/drop collection)
- Read replicas with async WAL streaming for horizontal read scale
- A global ID space or vector-clock-based conflict resolution

**No HNSW persistence between writes.** The graph is rebuilt on every checkpoint. A production system (e.g. hnswlib) can mmap the graph and update it in place, avoiding the O(N) rebuild cost at checkpoint.

**No query-time quantization.** All distance computation is exact float32. Production systems use product quantization (PQ) to compress vectors 4–32× and compute approximate distances on compressed data, trading recall for memory and speed.

**No per-filter graph.** Post-filtering breaks at low selectivity. Weaviate and Qdrant build per-segment HNSW graphs and use roaring bitmaps to intersect with filter results, maintaining recall at 1% selectivity.

See [`docs/api.md`](docs/api.md) for the Phase 2 distributed design sketch.
