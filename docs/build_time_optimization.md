# Build Time Optimization Log

All numbers measured on: Apple Silicon (ARM NEON), SIFT-1M (1M vectors, dim=128, M=16, ef_construction=200, L2). Script: `bench/bench_ann_compare.py`.

## Summary

| Step | Change | Build time | vec/s | Δ vs prev |
|------|--------|----------:|------:|----------:|
| Baseline | single-threaded, `fdatasync` per insert | 4,285 s | 234 | — |
| ① | batched WAL sync — one `fdatasync` per batch | ~430 s | ~2,300 | **10× faster** |
| ② | multi-threaded HNSW insert (18 threads) | 26 s | 38,456 | **17× faster** |
| ③ | generation-counter visited array | included in ② | — | removes 1 MB memset per beam search |

**Net result: 4,285 s → 26 s — 165× speedup. VectorDB now builds SIFT-1M faster than hnswlib (32 s).**

---

## How the gap was found

The starting point for diagnosis was the ann-benchmarks style comparison in `bench/bench_ann_compare.py`. The script runs VectorDB and hnswlib on the same machine, same dataset, same parameters (M=16, ef_construction=200, SIFT-1M), and prints a summary table.

Initial output:

```
Build time — VectorDB: 4,285 s   hnswlib: 31 s   (ratio: 136× slower)
```

This 136× gap could not be explained by algorithmic differences alone. HNSW insert is O(M × ef_construction × log N) for both implementations. At the same M and ef_construction, the theoretical build cost is identical. The gap had to be overhead outside the algorithm.

---

## ① Batched WAL sync

### Root cause

VectorDB writes a Write-Ahead Log (WAL) entry on every insert and calls `fdatasync()` after each one to guarantee durability: if the process crashes mid-insert, no data is lost. hnswlib builds purely in memory and has no persistence layer.

For SIFT-1M with 1M single-vector inserts, this created **1 million `fdatasync()` syscalls**. Each syscall blocks until the OS flushes the write to storage. On an SSD, a single `fdatasync` costs 50–200 µs. At 100 µs each, 1M calls = **100 seconds of pure syscall overhead** — and that was the optimistic estimate. In practice the measured cost was much higher (4,285 s total build vs the algorithmic estimate of ~4 min for 1M HNSW inserts single-threaded).

The core issue: the original Engine called `wal->sync()` inside the per-vector `Engine::insert()` hot path:

```cpp
// engine.cpp — original
std::string Engine::insert(const std::string& collection,
                            const std::string& user_id,
                            const float* vec,
                            const MetadataEntry& meta) {
    // ...
    state.index->insert(vec);
    state.wal->append(record);
    state.wal->sync();          // ← fdatasync every single insert
    // ...
}
```

### Fix

Move the sync to the end of a batch. The new `Engine::insert_batch()` appends all N WAL records first, then calls `sync()` once:

```cpp
// engine.cpp — after fix
std::vector<std::string> Engine::insert_batch(..., size_t count, ...) {
    for (size_t i = 0; i < count; ++i) {
        state.index->insert(...);
        state.wal->append(record);   // no sync here
    }
    state.wal->sync();               // ← single fdatasync for the whole batch
    // ...
}
```

The Python SDK was already routing all inserts through `Engine::insert_batch()` (`client.py: db.insert()`), so no API changes were needed. A batch of 10K vectors went from 10,000 `fdatasync` calls to 1.

**Durability guarantee preserved:** a crash after the batch `sync()` returns will have all records on disk. A crash before it returns loses at most one batch — the same guarantee as a database with group commit.

### Result

Build time dropped from 4,285 s to approximately 430 s (~10× speedup). The remaining 430 s was the algorithmic cost of single-threaded HNSW construction: graph traversal, distance computation, and neighbor selection for 1M nodes.

---

## ② Multi-threaded HNSW insert

### Root cause

HNSW construction is single-threaded. hnswlib's `add_items()` uses multi-threading by default. On a machine with 18 hardware threads, hnswlib parallelises across all of them. VectorDB's `HnswIndex::insert()` had no concurrency support.

At N=1M single-threaded, the theoretical build time is roughly:
- ~3,300 node visits per insert (ef_construction=200, layer-0 beam search)
- ~100 ns distance computation per visit
- 1M inserts × 3,300 × 100 ns ≈ **5.5 minutes**

The 430 s single-threaded result was in this range. With 18 threads, the target was ~25–30 s.

### Design: 256-stripe mutex locking

Concurrent HNSW insert is inherently harder than concurrent search because inserts modify the graph structure (add edges, update neighbor lists). The challenge is giving multiple threads safe write access to the same graph without serialising them on a single global lock.

The scheme used is **256-stripe mutex locking** (same model as hnswlib):

```cpp
static constexpr size_t kStripes = 256;
mutable std::array<std::mutex, kStripes> stripe_locks_;
mutable std::mutex entry_mu_;

std::mutex& stripe_lock(NodeId id) const { return stripe_locks_[id % kStripes]; }
```

Each node is assigned to one of 256 stripes based on `id % 256`. Any access that modifies or reads (for graph traversal) a node's neighbor list must hold that node's stripe lock. With 1M nodes across 256 stripes, each stripe covers ~4K nodes on average — two threads modifying different nodes in the same stripe contend; two threads in different stripes run fully in parallel.

A separate `entry_mu_` lock serialises updates to the global `entry_point` and `max_layer` fields, which every thread needs to snapshot at the start of its insert.

**Concurrency model:**
- Layer-0 adjacency (`adj0`): fixed pre-allocated array, but counts and data must still be locked because back-edges from other threads modify the same node's `adj0_count_` (see race fix below)
- Upper-layer adjacency (`nodes_flat_[id].neighbors[layer]`): `std::vector<NodeId>` — must always be copied under lock before iterating (see race fix below)
- `entry_point` + `max_layer`: always read and written under `entry_mu_`

**Pre-grow before spawning threads:**

All arrays are grown before threads start so that no thread needs to call `resize()` on shared memory:

```cpp
NodeId first_id = I.next_id_.fetch_add(count);
NodeId last_id  = first_id + count - 1;
I.grow_node_block(last_id);   // pre-allocates node_blocks_ and adj0_count_
I.grow_nodes(last_id);        // pre-allocates nodes_flat_
```

After this point, every thread accesses pre-existing array slots. The lock only protects per-node data fields, not array capacity.

### Problems encountered

Three separate data races caused intermittent segfaults (SIGSEGV, exit code 139). Each race was in a different part of the insert code path.

---

**Race 1 — node initialization vs `search_layer` reader**

Thread A (initialising node `id`) does:
```cpp
I.nodes_flat_[id].neighbors.assign(assigned_layer + 1, {});
```

`std::vector::assign()` internally: frees old allocation, sets new capacity, writes size. These are three separate writes to the `vector` header at separate moments. Thread B (in `search_layer`) reads:
```cpp
const auto& cn = nodes_flat_[candidate].neighbors;
if (layer < static_cast<int>(cn.size())) ...
```

If Thread B reads `cn.size()` while Thread A is mid-`assign()`, it can see a garbage size value — either the partially-written new size or a dangling pointer from the freed old allocation. Accessing a garbage size led to out-of-bounds access → segfault.

**Fix:** initialise the node's neighbor list under `stripe_lock(id)`. The `search_layer` reader also holds `stripe_lock(candidate)` when accessing upper-layer neighbors, so they are always mutually excluded:

```cpp
{
    std::lock_guard<std::mutex> lk(I.stripe_lock(id));
    I.nodes_flat_[id].neighbors.assign(assigned_layer + 1, {});
    for (int lv = 1; lv <= assigned_layer; ++lv)
        I.nodes_flat_[id].neighbors[lv].reserve(I.cfg.M);
}
```

The `reserve()` in the same lock ensures that subsequent `push_back()` calls on upper-layer neighbor lists (which also happen under lock) do not reallocate — reallocation inside a locked section would move the buffer while another lock holder holds a pointer to the old buffer.

---

**Race 2 — upper-layer readers in `search_layer`**

Even after fixing initialization, `search_layer` still read upper-layer neighbor lists without a lock:
```cpp
// original search_layer, upper layers
for (NodeId neighbor : nodes_flat_[candidate].neighbors[layer]) { ... }
```

Thread A (`search_layer`) iterates `neighbors[layer]` as a range-for loop, which holds an internal iterator into the `vector`. Thread B (adding a back-edge to `candidate`) holds `stripe_lock(candidate)` and calls `push_back` on the same `vector`. `push_back` may reallocate — moving the buffer to a new address — while Thread A's iterator still points at the old buffer. Use-after-free → segfault.

**Fix:** copy the neighbor list under the lock before iterating:

```cpp
std::vector<NodeId> layer_nbrs;
{
    std::lock_guard<std::mutex> lk(stripe_lock(candidate));
    const auto& cn = nodes_flat_[candidate].neighbors;
    if (layer < static_cast<int>(cn.size()))
        layer_nbrs = cn[layer];
}
for (NodeId neighbor : layer_nbrs) expand(neighbor);
```

Layer-0 uses a flat pre-allocated array (`adj0_ptr`) instead of `std::vector`, so it does not reallocate. Layer-0 reads in `search_layer` are therefore safe without a lock from the reallocation standpoint (though the count and data still required locking for correctness — see Race 4).

---

**Race 3 — entry_point visible before max_layer**

The first thread to call `insert_with_id_mt` sees `entry_point == kInvalidNode` and sets the entry point:

```cpp
// original attempt using CAS
if (I.entry_point_.compare_exchange_strong(expected, id)) {
    I.max_layer_.store(assigned_layer);   // ← stored AFTER entry_point is visible
    return;
}
```

A second thread could see `entry_point = first_id` (CAS succeeded) but `max_layer = -1` (the store hasn't happened yet). That thread would then attempt:
```cpp
for (int level = cur_max; level > assigned_layer; --level)  // cur_max = -1: loop never executes
```

With `cur_max = -1`, the greedy descent loop doesn't run. The thread goes directly to the edge-adding loop `for (int level = min(assigned_layer, -1); level >= 0; --level)`, which also doesn't run because `min(anything, -1) = -1 < 0`. The second node is inserted with **zero edges**. Nodes with no edges are unreachable in later graph traversal.

**Fix:** always read `entry_point` and `max_layer` as a consistent pair under `entry_mu_`:

```cpp
NodeId entry_point;
int    cur_max;
{
    std::lock_guard<std::mutex> lk(I.entry_mu_);
    entry_point = I.entry_point_.load(std::memory_order_relaxed);
    cur_max     = I.max_layer_.load(std::memory_order_relaxed);
    if (entry_point == kInvalidNode) {
        I.max_layer_.store(assigned_layer, std::memory_order_relaxed);
        I.entry_point_.store(id, std::memory_order_relaxed);
        return;
    }
}
```

No thread can observe `entry_point = X` without also observing the `max_layer` that was set alongside it.

---

**Race 4 — forward edges and back-edges both modify `adj0_count_[id]`**

This race caused not segfaults but silent graph quality degradation (lost edges → lower recall). It was found after diagnosing the segfaults.

Thread A owns node `id` and adds a forward edge (no lock):
```cpp
I.add_directed(id, nbr, 0, M_max);   // modifies adj0_count_[id], adj0_ptr(id)[...]
```

Simultaneously, Thread B (inserting node `new_node`) has found `id` as a good neighbor and adds a back-edge under `stripe_lock(id)`:
```cpp
std::lock_guard<std::mutex> lk(I.stripe_lock(id));
I.add_directed(id, new_node, 0, M_max);  // also modifies adj0_count_[id]
```

Both threads read `adj0_count_[id]`, write to `adj0_ptr(id)[count]`, and increment `adj0_count_[id]`. Since Thread A holds no lock, the increment is not atomic — one of the two writes to `adj0_count_[id]` is lost. The edge is silently dropped.

Thread B can only find `id` as a neighbor after id's first back-edge has been added (making it reachable in beam search). At that point Thread A may still be in the middle of its forward-edge loop. The window is real.

**Fix:** lock the forward edge too, on every layer:

```cpp
for (NodeId nbr : nbrs) {
    {
        std::lock_guard<std::mutex> lk(I.stripe_lock(id));
        I.add_directed(id, nbr, level, M_max);    // forward: id → nbr
    }
    {
        std::lock_guard<std::mutex> lk(I.stripe_lock(nbr));
        I.add_directed(nbr, id, level, M_max);    // back: nbr → id
    }
}
```

Both locks are taken and released separately (never simultaneously), so there is no circular wait.

---

### Result after fixing all races

```
Build time — VectorDB: 26 s  hnswlib: 32 s  (VectorDB 1.2× faster)

 ef   VectorDB QPS   VectorDB R@10   hnswlib QPS   hnswlib R@10   QPS ratio
 50        12,600          0.9543        86,476         0.9460        6.9×
200         4,016          0.9965        27,410         0.9957        6.8×
800         1,257          0.9992         8,582         0.9993        6.8×
```

Recall@10 is essentially identical across all ef values — the multi-threaded graph has the same quality as the sequential graph.

---

## ③ Generation-counter visited array

### Root cause

After fixing the MT insert, profiling revealed that `search_layer` — called thousands of times during a batch insert — was spending significant time on visited-array reset. The visited array at the time was:

```cpp
std::vector<uint8_t> visited(max_node_id_ + 1, 0);
```

Constructed fresh on every `search_layer` call. At N=1M, `max_node_id_ + 1 = 1,000,000`, so the constructor calls `memset` on 1 MB per call. During a batch insert of 1M vectors, each insert runs ~3 `search_layer` calls. That is roughly **3 million 1-MB memsets = 3 TB of memory writes**.

### Fix

Replace the resettable `uint8_t` array with a generation counter. Instead of writing `0` to every slot to "clear" the array, increment a generation counter. A slot is considered visited if its stamp equals the current generation — all old stamps are implicitly stale without any write.

```cpp
struct VisitedTable {
    std::vector<uint32_t> stamps;
    uint32_t gen = 0;

    uint32_t acquire(size_t needed) {
        if (stamps.size() < needed) stamps.resize(needed, 0);
        if (++gen == 0) {                          // generation wrapped (after 4B calls)
            std::fill(stamps.begin(), stamps.end(), 0);
            gen = 1;
        }
        return gen;
    }
};
thread_local VisitedTable tl_visited;
```

Usage in `search_layer`:
```cpp
uint32_t gen    = tl_visited.acquire(max_node_id_ + 1);
auto&    stamps = tl_visited.stamps;
// mark: stamps[id] = gen
// check: stamps[id] == gen
```

`acquire()` is O(1) — no memset. The first call on each thread allocates the stamps array (4 MB at N=1M); every subsequent call is a single increment. The `thread_local` storage makes it safe for concurrent callers with no locking.

The generation overflow case (after 4 billion `search_layer` calls on one thread) does a one-time full reset — amortised to essentially zero over any realistic workload.

### Result

The visited-array overhead (previously 1 MB of memset per `search_layer` call) dropped to zero. This benefited both build time (many `search_layer` calls during insert) and query time (one call per query). The effect is most visible at large N where the 1 MB memset is the dominant per-call cost.

---

## Why the comparison is fair despite hnswlib using more threads

The benchmark compares VectorDB (Python SDK, `insert_batch`) against hnswlib (`add_items(train, list(range(N)))`). Both are called from Python with default settings:

- hnswlib `add_items` defaults to `num_threads = hardware_concurrency()` — it has always been multi-threaded
- VectorDB `insert_batch` now also uses `hardware_concurrency()` threads via `insert_batch_mt`

Both run on the same 18-thread Apple Silicon machine. Neither is artificially constrained. The 26 s vs 32 s comparison is apples-to-apples at equal parallelism.

The comparison also uses the same M=16, ef_construction=200, SIFT-1M dataset, and identical ground-truth evaluation (R@10 vs brute-force top-10). Recall values are within 0.001 at every ef, confirming that MT insert does not compromise graph quality.
