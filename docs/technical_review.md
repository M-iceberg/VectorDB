# VectorDB Technical Review

This document covers four engineering challenges encountered during development: build time optimization, QPS optimization, multi-threaded insert correctness, and the memory/speed tradeoff. Each section explains what the problem was, how it was diagnosed, and what the fix was.

---

## 1. Build Time: 4,285s → 23s

### The problem

The first sign of trouble was running `bench/bench_ann_compare.py`, which compares VectorDB and hnswlib on SIFT-1M (1M vectors, dim=128). Build results:

```
VectorDB:  4,285s  (234 vec/s)
hnswlib:      31s  (31,000 vec/s)
```

136× slower. hnswlib is a pure in-memory library with no persistence. VectorDB writes a WAL and persists vectors to disk — but a 136× gap is not explained by I/O alone.

### Diagnosis: fdatasync per insert

`bench/bench_profile.cpp` profiled the C++ insert path. The call tree showed `fdatasync` at the top — not HNSW graph construction. The WAL was calling `fdatasync()` after every single insert record.

For 1M vectors, that is 1,000,000 `fdatasync` syscalls. On macOS, each takes ~4ms. Total: ~4,000s. The HNSW graph itself builds in ~20s.

### Fix 1: Batch WAL sync

Changed the WAL to sync only once per batch insert, not once per record. The Python API already accepted batches (`db.insert("col", ids=[...], vectors=np.array(...))`), so the change was entirely in the C++ engine — accumulate all records for the batch, then call `fdatasync` once at the end.

Result: build time dropped from 4,285s to **260s** (~16× faster). Still 8× behind hnswlib.

### Remaining gap: single-threaded insert

After eliminating the I/O bottleneck, profiling showed the remaining time was pure HNSW graph construction. The `insert()` call was single-threaded — each of the 1M vectors was inserted sequentially, each requiring a beam search through the graph to find good neighbors.

hnswlib's `add_items()` is multi-threaded by default, using all hardware threads.

### Fix 2: Multi-threaded batch insert (`insert_batch_mt`)

Implemented parallel insert at the C++ level. The key insight is that multiple threads can insert into HNSW concurrently if they use fine-grained locking:

- **Pre-assign all NodeIds** atomically before spawning threads (prevents ID collisions)
- **Pre-grow all arrays** to full size before threads start (prevents resize races)
- **256-stripe mutex** on NodeId: lock `id % 256` when modifying a node's adjacency list (two nodes in different stripes never block each other)
- **Read-only beam search**: `search_layer` reads the graph without locks — ANN tolerates stale edges

This is the same concurrency model as hnswlib.

Result: build time dropped from 260s to **23s** — now **1.4× faster** than hnswlib despite writing a WAL and persisting all vectors to disk.

---

## 2. QPS Optimization: 4,016 → 42,137

### The problem

After fixing build time, QPS was still 6.8× behind hnswlib:

```
 ef    VectorDB QPS    hnswlib QPS    gap
200          4,016         27,410    6.8×
```

### Diagnosis: it was not an algorithmic gap

Before optimizing, the per-query CPU cost was calculated:

- **VectorDB**: 4,016 QPS single-threaded → **249 µs per query**
- **hnswlib**: 27,410 QPS, using 18 hardware threads → wall time per 10K queries = 0.365s → CPU time = 0.365 × 18 = 6.57s → **657 µs per query**

VectorDB's per-query CPU cost was **2.6× lower** than hnswlib. The gap was not because VectorDB's search was slower — it was because hnswlib ran 18 queries in parallel while VectorDB ran 1.

The benchmark was calling `db.search(query=q)` in a Python loop (one at a time), while hnswlib used `knn_query(all_10K_queries)` — a single C++ call that dispatches across all threads internally. An unfair comparison.

### Fix 1: NEON 4-accumulator loop unrolling

Before addressing parallelism, the L2 distance kernel was also a bottleneck (51% of search time per `sample` profiling). The original implementation used a single NEON accumulator register:

```cpp
float32x4_t vsum = vdupq_n_f32(0.0f);
for (; i + 4 <= dim; i += 4) {
    float32x4_t diff = vsubq_f32(vld1q_f32(a+i), vld1q_f32(b+i));
    vsum = vmlaq_f32(vsum, diff, diff);
}
```

The problem: `vmlaq_f32(vsum, ...)` reads the result of the previous iteration's `vmlaq_f32`. On Apple M-series, FMA latency is ~3 cycles. The CPU cannot start the next FMA until the previous one finishes — limited to 1 iteration per 3 cycles regardless of available execution units.

Apple M-series performance cores have 4 NEON execution units. With one accumulator, 3 of the 4 sit idle every cycle.

Fix: use 4 independent accumulators. No dependency between them, so the CPU issues all 4 FMAs per cycle:

```cpp
float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
for (; i + 16 <= dim; i += 16) {
    float32x4_t d0 = vsubq_f32(vld1q_f32(a+i),    vld1q_f32(b+i));
    float32x4_t d1 = vsubq_f32(vld1q_f32(a+i+4),  vld1q_f32(b+i+4));
    float32x4_t d2 = vsubq_f32(vld1q_f32(a+i+8),  vld1q_f32(b+i+8));
    float32x4_t d3 = vsubq_f32(vld1q_f32(a+i+12), vld1q_f32(b+i+12));
    s0 = vmlaq_f32(s0, d0, d0);
    s1 = vmlaq_f32(s1, d1, d1);
    s2 = vmlaq_f32(s2, d2, d2);
    s3 = vmlaq_f32(s3, d3, d3);
}
float32x4_t vsum = vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3));
```

Result: single-threaded QPS improved ~30%.

### Fix 2: Parallel batch search (`search_batch`)

HNSW search is trivially parallelizable across queries: each query only reads the graph (no writes), and the visited-node tracking (`tl_visited`) is `thread_local` — two threads searching simultaneously never share mutable state.

Added `HnswIndex::search_batch()` at the C++ level, which splits N queries across `hardware_concurrency()` threads. The Python binding accepts a 2D numpy array and returns a list of lists — one call replaces a Python loop of N single-query calls.

Result:

```
 ef    VectorDB QPS    hnswlib QPS    faiss QPS
200         42,137         27,288       18,499
```

VectorDB now beats hnswlib by 54% and faiss by 2.3× at ef=200. At ef=800, VectorDB beats faiss by 3.8×.

---

## 3. Multi-threaded Insert: 4 Race Conditions

Enabling parallel insert introduced data races. The symptom was intermittent SIGSEGV (exit code 139) under stress testing. Four distinct races were found and fixed.

### Race 1: `neighbors.assign()` vs `search_layer` reading `size()`

`insert_with_id` called `node.neighbors.assign(assigned_layer + 1, {})` to initialize a new node's neighbor vectors. Concurrently, `search_layer` on another thread called `node.neighbors.size()` to check how many layers the node has.

`std::vector::assign` and `std::vector::size` on the same vector from two threads is undefined behavior — `assign` may reallocate, invalidating the size read mid-operation.

Fix: wrap the `neighbors.assign()` call under `stripe_lock(id)`. `search_layer` tolerates stale reads (it's an approximation algorithm), so it does not need to hold the lock — but the writer must.

### Race 2: Iterator invalidation during back-edge insertion

When inserting node `id` at level 0, the code iterated over `id`'s new neighbor list with a range-for loop while concurrently inserting back-edges:

```cpp
for (NodeId nbr : neighbors) {  // iterating node id's vector
    add_directed(nbr, id, 0, M_max);  // may push_back into id's vector
}
```

`add_directed(nbr, id, ...)` calls `add_edge` which calls `push_back` on `nodes_flat_[id].neighbors[0]`. A `push_back` that causes reallocation invalidates all iterators into that vector — the range-for loop's internal pointer becomes a dangling pointer.

Fix: copy the neighbor list into a local `std::vector<NodeId>` under the lock before iterating. The iteration then operates on the copy, which cannot be invalidated by concurrent writes to the original.

### Race 3: `entry_point_` set before `max_layer_` updated

When a new node becomes the entry point (its layer exceeds the current maximum), the code updated `entry_point_` and `max_layer_` as two separate atomic stores:

```cpp
I.entry_point_.store(id);       // step 1
I.max_layer_.store(new_layer);  // step 2
```

A thread performing search between step 1 and step 2 reads the new `entry_point_` but the old `max_layer_`. It starts beam search from a node that claims to be on layer L but is actually on layer L+1 — and tries to access `node.neighbors[L]` which does not exist yet. Out-of-bounds access → SIGSEGV.

Fix: protect both reads and writes of `entry_point_` + `max_layer_` with a dedicated `entry_mu_` mutex. The search thread acquires a shared lock to read both values atomically; the writer acquires an exclusive lock to update both.

### Race 4: Forward edge vs back-edge on the same adjacency count

When adding edges for a new node `id` at level 0:

- **Forward edge** (`add_directed(id, nbr, 0, M_max)`): modifies `adj0_count_[id]`. This was done without any lock.
- **Back-edge** (`add_directed(nbr, id, 0, M_max)`): modifies `adj0_count_[id]` too (because `id` is the destination). This was done under `stripe_lock(id)`.

Both paths modify the same array element `adj0_count_[id]` — one with the lock held, one without. This is a data race even on the same thread's "own" data, because another thread inserting a different node can trigger the back-edge path concurrently.

The symptom was subtle: not a crash, but silently lost edges, which caused lower recall. This was initially misdiagnosed as a search algorithm bug until the asymmetric locking pattern was spotted.

Fix: always hold `stripe_lock(id)` for forward edges at all levels, not just upper layers. There is no asymmetry — every modification to a node's adjacency list is protected by `stripe_lock(node_id)`.

### Bonus: `insert_for_recovery` idempotency

Discovered during stress testing after the four races were fixed. When a checkpoint partially writes (graph snapshot written, id_map rename fails), recovery loads the newer graph snapshot and replays the WAL. The WAL contains insert records for nodes already in the snapshot. `insert_for_recovery` called `insert_with_id` unconditionally — for an existing node, this re-runs `assign_layer()` (random), clears all neighbors, and rebuilds connections. The graph becomes inconsistent.

Fix: at the start of `insert_for_recovery`, skip nodes that already exist and are not tombstoned:

```cpp
if (I.node_exists(id) && !I.nodes_flat_[id].tombstone)
    return;
```

WAL replay is now idempotent: replaying a record for an already-present node is a no-op.

---

## 4. Memory vs Speed Tradeoff

VectorDB uses ~2.4× more memory than hnswlib and faiss for the same index:

| System | SIFT-1M RSS | B/vec |
|--------|------------:|------:|
| VectorDB | 2,017 MB | 2,115 |
| hnswlib | 820 MB | 860 |
| faiss | 822 MB | 862 |

This is an intentional design choice, not an oversight. The extra memory comes from two decisions:

### Decision 1: Collocated memory layout (`node_blocks_`)

Each node's layer-0 neighbor IDs and float vector are stored in a single contiguous block:

```
node_blocks_[id × stride .. (id+1)×stride):
  [adj0 neighbors: M0 × NodeId = 128 bytes]
  [vector data:    dim × float  = 512 bytes]
  ────────────────────────────────────────
  stride = 640 bytes/node
```

During search, after reading a neighbor's ID from `node_blocks_`, the neighbor's own vector is at a known offset in the same allocation — one multiply-add, no pointer chase. All M0=32 neighbor vectors can be prefetched before the distance loop begins:

```cpp
for (int k = 0; k < cnt; ++k)
    __builtin_prefetch(vec_ptr(nbrs[k]), 0, 0);
for (int k = 0; k < cnt; ++k)
    expand(nbrs[k]);
```

hnswlib and faiss store neighbor IDs and vectors in separate arrays. Reading a neighbor's vector requires following an additional pointer, which is a DRAM round-trip at large N where the graph exceeds L3 cache (~40–50MB on Apple M-series). This is why faiss's HNSW QPS collapses at high ef — it becomes DRAM-bound.

The collocated layout costs memory (vectors stored inside `node_blocks_`), but reduces DRAM round-trips per neighbor from 2 to 1 and allows effective prefetching.

### Decision 2: Separate `VectorFile` for durability

VectorDB writes every inserted vector to `vectors.vdb` on disk. On recovery after a crash, the WAL is replayed and `insert_for_recovery` reads vectors from this file to rebuild the graph. Without it, a crash mid-WAL-replay would have no way to reconstruct the distance function inputs.

The file is memory-mapped (`mmap`), which makes random access fast but means the OS maps the file into the process address space — it shows up in RSS. For SIFT-1M: 1M × 512 bytes = 488 MB mapped in addition to the 640 MB in `node_blocks_`.

hnswlib and faiss have no persistence layer. They are pure in-memory structures. If the process crashes, the index is gone.

### The tradeoff in one sentence

VectorDB trades ~2.4× more memory for: 1.6–1.9× lower per-query latency (vs hnswlib/faiss), crash recovery with WAL, and metadata filtering — none of which hnswlib or faiss provide.

For memory-constrained environments, the `node_blocks_` layout could be separated (vectors stored externally, neighbor IDs only inline). This would halve the `node_blocks_` cost at the price of one extra pointer dereference per neighbor — recovering the same DRAM bottleneck that the current layout avoids. It is a meaningful engineering choice, not a free lunch.
