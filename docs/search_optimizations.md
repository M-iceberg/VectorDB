# HNSW Search Optimization Log

All numbers measured on: Apple Silicon (ARM NEON), N=100K, dim=128, ef=200, M=16, Release build.

## Summary

| Step | Change | QPS | Δ vs prev | Δ vs baseline |
|------|--------|----:|----------:|--------------:|
| Baseline | original implementation | 1,669 | — | — |
| ① | visited: `unordered_set` → flat `uint8_t` | 2,738 | +64% | +64% |
| ② | vecs: `unordered_map` → flat `float[]` | 3,570 | +30% | +114% |
| ③ | nodes: `unordered_map` → flat `vector<HnswNode>` | 3,872 | +8% | +132% |
| ④ | adj0: `vector<NodeId>` → flat CSR array | 4,233 | +9% | +154% |
| ⑤ | look-ahead prefetch on adj0 neighbors | 4,459 | +5% | +167% |

---

## Background: how HNSW search works

HNSW (Hierarchical Navigable Small World) builds a multi-layer proximity graph over all inserted vectors. Layer 0 contains every node with up to M0=32 edges each; upper layers contain exponentially fewer nodes and serve as long-range shortcuts.

A `search_layer(query, entry_point, ef, layer)` call is a greedy beam search:

1. Start at `entry_point`. Maintain two heaps: `C` (candidates to expand, min-heap by distance) and `W` (result set, max-heap by distance, capped at `ef`).
2. Pop the closest candidate from `C`. If it is farther than the worst result in `W`, stop.
3. For each neighbor of that candidate: if not visited, compute distance to query, insert into both heaps.
4. Repeat until `C` is empty or the stopping criterion fires.

At N=100K, ef=200, each query visits roughly 3,300 nodes and touches each one's neighbor list and stored vector.

---

## Why pointer indirection causes cache misses

Optimizations ①–④ all follow the same principle: replace pointer-based data structures with flat arrays. Understanding why this matters requires understanding the memory hierarchy.

### Memory hierarchy

```
CPU registers    ~0.3 ns    a few dozen slots
L1 cache         ~1 ns      ~64 KB
L2 cache         ~4 ns      ~512 KB
L3 cache         ~10 ns     ~8–24 MB (Apple Silicon)
RAM (heap)       ~50–100 ns  GBs
Disk (SSD)       ~100,000 ns persistent storage (WAL, graph.bin, etc.)
```

Cache misses in HNSW search are RAM→cache loads, not disk reads. The disk only comes into play for WAL replay and checkpoint loading at startup.

The CPU loads memory in 64-byte **cache lines**. When it reads address A, the 64 bytes surrounding A are loaded into cache together. If the next access is to a nearby address, it may already be in cache (cache hit, ~1–4 ns). If it is to a completely unrelated address, it is a cache miss (~50–100 ns, stall while waiting for RAM).

### Why pointer indirection causes cache misses

The heap stores both pointers and actual data, but they live at separate addresses:

```
vector<float> header (24 bytes): [ptr | size | capacity]  ← somewhere on heap
                                    │
                                    └→ [1.0, 0.5, 0.3, ...]  ← actual data, elsewhere on heap
```

Reading a `vector<float>` requires two separate memory accesses at two unrelated addresses. Each one may be a cache miss. A chain of three data structures:

```
unordered_map → [bucket: ptr] → [vector header: ptr, size, cap] → [float, float, ...]
 cache miss #1      cache miss #2                                    cache miss #3
```

The CPU cannot predict where each pointer leads. The hardware prefetcher — which speculatively loads memory it thinks you will need next — cannot help because the next address is not known until the current pointer is dereferenced.

### Why flat arrays eliminate this

```
vecs_flat_.data() + id * dim → [float, float, float, ...]
 base ptr already in L1           one cache miss max, address computable in advance
```

`vecs_flat_.data()` is a field of the `Impl` struct that stays in L1 cache throughout a search. The offset `id * dim` is pure arithmetic — no memory access. The resulting address is known before the load, so `__builtin_prefetch` can issue it early (optimization ⑤). With pointer indirection, the address is unknown until each pointer is dereferenced, so prefetch cannot help.

The improvement is not that flat arrays make access sequential — HNSW still jumps to scattered node IDs. The improvement is that each access requires **zero pointer chases** instead of two or three, eliminating the chain of stalled loads.

---

## ① Visited-node tracking: `unordered_set` → flat `uint8_t` array

### Role in HNSW search
The visited set prevents revisiting the same node twice in one query. In step 3 above, the same neighbor can be found via multiple paths through the graph — without a visited check, it would be pushed onto `C` repeatedly, each time triggering another distance computation and another round of expansions. The visited set ensures each node is evaluated at most once per query.

### Original implementation
```cpp
std::unordered_set<NodeId> visited;
visited.insert(ep);
// ...
if (!visited.insert(neighbor).second) continue;
```
**Why it was written this way:** `unordered_set` is the natural C++ container for membership tracking. During early development, correctness was the priority; allocator behavior was not considered.

### Problem found by profiling
macOS `sample` during a 10K-query search run (8-second window, 8,422 total samples):

| Component | % of search time |
|-----------|----------------:|
| `unordered_set` alloc + hash insert | 50.7% |
| free visited set after each search | 9.8% |
| **visited total** | **60.6%** |
| NeonL2::compute (NEON SIMD) | 12.4% |

Every `search_layer()` call allocates a fresh hash table, inserts up to `ef` NodeIds (each potentially triggering `operator new` on rehash), and frees it on return. At ef=200, N=100K, that is ~3,300 hash-table insertions per query.

### New implementation
```cpp
std::vector<uint8_t> visited(max_node_id_ + 1, 0);
visited[ep] = 1;
// ...
if (visited[neighbor]) continue;
visited[neighbor] = 1;
```
One allocation of `max_node_id_ + 1` bytes (100 KB at N=100K). Check and mark are single byte reads/writes with no `operator new` during traversal. The array is contiguous so it fits in L2 cache.

**Why it works:** hash table overhead (load factor, collision probing, per-bucket allocation) dominated the actual membership test. A flat array has zero overhead beyond the index arithmetic.

---

## ② Vector storage: `unordered_map<NodeId, vector<float>>` → flat `float[]`

### Role in HNSW search
To decide whether a candidate node belongs in the result set, we compute the L2 distance between the query vector and the node's stored vector. `vecs` is the table that maps node ID → the float[] data the user inserted at build time. `dist_q()` is called once per node visited — roughly 3,300 times per query.

### Original implementation
```cpp
std::unordered_map<NodeId, std::vector<float>> vecs;

float dist_q(const float* query, NodeId b) const {
    return dc->compute(query, vecs.at(b).data(), cfg.dim);
}
```
**Why it was written this way:** NodeIds are user-provided and not guaranteed to be sequential. `unordered_map` handles sparse or arbitrary IDs without preallocating memory for gaps. Correct and flexible.

### Problem
`vecs.at(b)` = hash lookup → get `std::vector<float>` header → call `.data()` → pointer to heap float data. Two levels of indirection on every distance computation. Called ~3,300 times per query.

### New implementation
```cpp
std::vector<float> vecs_flat_;  // vecs_flat_[id * dim .. (id+1)*dim)

float* vec_ptr(NodeId id) { return vecs_flat_.data() + id * cfg.dim; }

float dist_q(const float* query, NodeId b) const {
    return dc->compute(query, vec_ptr(b), cfg.dim);
}
```
`vec_ptr(b)` = one multiply-add. Zero indirection.

**Trade-off:** requires NodeIds to be dense (gaps waste `dim * sizeof(float)` = 512 B per deleted slot). Acceptable for this workload where IDs are 0..N.

---

## ③ Node graph structure: `unordered_map<NodeId, HnswNode>` → flat `vector<HnswNode>`

### Role in HNSW search
`HnswNode` holds the neighbor list for each node at every layer — the actual edges of the HNSW graph. During step 3 of beam search, after popping candidate `c` from `C`, we look up `nodes[c]` to get its neighbor list and iterate over each neighbor. This lookup is the entry point into the graph traversal — every node expansion starts here.

### Original implementation
```cpp
std::unordered_map<NodeId, HnswNode> nodes;

const auto& candidate_nbrs = nodes.at(candidate).neighbors;
```
**Why it was written this way:** same reason as vecs — flexible for arbitrary NodeIds, and the natural container when building incrementally without knowing the final N.

### Problem
`nodes.at(candidate)` = hash lookup on every node expansion. The search_layer hot loop calls this for every neighbor of every candidate visited. At ef=200 and M=16, that is thousands of hash lookups per query.

### New implementation
```cpp
std::vector<HnswNode> nodes_flat_;  // indexed by NodeId

// HnswNode default-constructs with id = kInvalidNode (sentinel for "not present")
bool node_exists(NodeId id) const {
    return id < nodes_flat_.size() && nodes_flat_[id].id != kInvalidNode;
}

const auto& candidate_nbrs = nodes_flat_[candidate].neighbors;
```
Direct array access. `HnswNode::id == kInvalidNode` serves as the "slot not yet inserted" sentinel.

**Why the gain is smaller than ①②:** `HnswNode.neighbors` is still a `vector<vector<NodeId>>` whose inner data lives on the heap. Eliminating the hash lookup helps, but the neighbor ID data itself is still scattered in memory.

---

## ④ Layer-0 adjacency: `vector<vector<NodeId>>` → flat CSR array

### Role in HNSW search
Layer 0 is the bottom layer of the HNSW hierarchy — it contains all N nodes, each with up to M0=32 neighbors. Roughly 90% of search traversal happens on layer 0: upper layers are traversed quickly (few nodes, greedy descent), then the bulk of ef=200 beam search runs across the dense layer-0 graph. Reading a candidate's layer-0 neighbor list is the single hottest memory access in the entire search path — it runs once per candidate popped from `C`.

### Original implementation
```cpp
struct HnswNode {
    std::vector<std::vector<NodeId>> neighbors;  // neighbors[layer][i]
};

// In search_layer:
for (NodeId neighbor : nodes_flat_[candidate].neighbors[0]) { ... }
```
Reading one layer-0 neighbor requires:
1. `nodes_flat_[candidate]` — direct array access (after ③)
2. `.neighbors` — `vector` header inside HnswNode (direct)
3. `.neighbors[0]` — inner `vector<NodeId>` header on heap (one pointer chase to scattered allocation)
4. `[i]` — actual NodeId (data in that allocation)

**Why it was written this way:** `vector<vector<NodeId>>` naturally grows per-layer without knowing M upfront. Simple and correct.

### Problem
Each node's layer-0 neighbor list is a separate heap allocation scattered across memory. At N=100K, that is 100K separate allocations for layer-0 neighbor lists. Every cache line loaded to read node A's neighbors carries nothing useful for node B's neighbors. The memory access pattern is entirely random from the hardware prefetcher's perspective.

### New implementation
```cpp
std::vector<NodeId>  adj0_;        // adj0_[id * M0 + k] = k-th neighbor at layer 0
std::vector<uint8_t> adj0_count_;  // actual neighbor count per node

const NodeId* adj0_ptr(NodeId id) const { return adj0_.data() + (size_t)id * cfg.M0; }

// In search_layer:
int cnt = adj0_count_[candidate];
const NodeId* nbrs = adj0_ptr(candidate);
for (int k = 0; k < cnt; ++k) expand(nbrs[k]);
```
`adj0_ptr(id)` = one multiply-add, same pattern as `vec_ptr`. All M0 neighbor IDs for a given node occupy a contiguous 128-byte slot (M0=32 × 4 bytes = one or two cache lines). Upper-layer adjacency (layers 1+) stays as `HnswNode.neighbors[layer]` since upper-layer traversal is a negligible fraction of search time.

**Why this enables prefetch (see ⑤):** with a flat array, all M0 neighbor IDs are readable via pointer arithmetic alone — no heap dereference needed. We can compute all 32 prefetch target addresses before issuing a single distance computation.

---

## ⑤ Look-ahead prefetch

### Role in HNSW search
After reading the M0=32 neighbor IDs from `adj0_ptr(candidate)`, we call `dist_q(query, nb)` for each — which loads the 512-byte float vector at `vec_ptr(nb)`. Those 32 vectors live at scattered positions in `vecs_flat_` (one slot per node ID). If a vector is not in cache, loading it takes ~50–100 ns of DRAM latency. Without prefetch, the CPU issues one load, stalls until the data arrives, computes the distance, then issues the next load. With prefetch, all 32 load requests are issued upfront, and the memory controller fetches them in parallel while the CPU computes distance for the first one.

### Failed attempts before flat data structures

**Attempt 1 — `compute_batch()` prefetch (original code):**
```cpp
// In compute_batch():
__builtin_prefetch(candidates + (i + 2) * dim, 0, 1);
```
Never called during HNSW search. `search_layer()` calls `dist_q()` → `compute()` directly, one neighbor at a time. `compute_batch()` is used for bulk scans, not graph traversal. The prefetch was dead code on the search path.

**Attempt 2 — graph prefetch on unordered_map:**
```cpp
// After pushing neighbor to C:
auto v = vecs.find(nn);
if (v != vecs.end()) __builtin_prefetch(v->second.data(), 0, 0);
```
QPS dropped 1,669 → 1,386 (−17%). To compute the address to prefetch, this called `vecs.find()` — a hash lookup. The cost of finding what to prefetch exceeded the memory latency being avoided.

**Attempt 3 — graph prefetch after expand():**
```cpp
for (int k = 0; k < cnt; ++k) {
    expand(nbrs[k]);
    __builtin_prefetch(vec_ptr(nbrs[k]), 0, 0);  // too late
}
```
`expand()` calls `dist_q()` which loads `vec_ptr(nbrs[k])` synchronously. The data is already in cache by the time the prefetch hint is issued. No measurable effect.

### Working implementation
```cpp
// Issue all prefetches before computing any distances.
for (int k = 0; k < cnt; ++k)
    __builtin_prefetch(vec_ptr(nbrs[k]), 0, 0);
// Now compute distances — neighbor vectors are arriving from memory while dist_q(nbrs[0]) runs.
for (int k = 0; k < cnt; ++k) expand(nbrs[k]);
```

Without prefetch, the 32 neighbors are processed serially: load vec(nbrs[0]) → stall 50–100 ns → compute distance → load vec(nbrs[1]) → stall → compute → ... Each neighbor pays full DRAM latency.

With the two-loop structure, all 32 load requests are issued first. While the CPU computes `dist_q(nbrs[0])`, the memory controller fetches nbrs[1..31] in parallel. By the time distance 0 is done, distance 1's data is already in cache. Serial waiting becomes computation–memory overlap.

**Why this works: cost of computing the prefetch address must be less than the memory latency being hidden.**

```
vec_ptr(nbrs[k]) = base + nbrs[k] * dim   →  1 multiply-add ≈ 1 ns
load that vec from RAM                     →  50–100 ns
```

Spending 1 ns to issue a hint that saves 50–100 ns is worthwhile. With `unordered_map`, computing the address required a hash lookup (~30–50 ns) — nearly as expensive as the cache miss itself, so prefetch had no net benefit.

### Measured results (N=100K, dim=128, ef=200, 5K queries, Apple M-series)

| | All optimizations (①–⑤) | No prefetch (①–④ only) | Original baseline |
|--|--|--|--|
| QPS | **4,407** | 4,219 | 1,669 |
| Per-query latency | 0.226 ms | 0.237 ms | 0.599 ms |
| Distance fraction | 14% | 14% | ~12% |
| Prefetch gain | **+4.5%** | — | — |

**Flat arrays (①–④) account for nearly all the gain**: 1,669 → 4,219 QPS (+152%). Prefetch (⑤) adds 4.5% on top at N=100K. This is expected: `vecs_flat_` is 51 MB and Apple Silicon L3 is 8–24 MB, so frequently-accessed vectors are already in L3 — there is limited DRAM latency to hide.

**Prefetch impact grows with dataset size.** At N=1M, `vecs_flat_` is 512 MB — guaranteed DRAM miss on every graph hop (~50–100 ns each). Prefetching all M0=32 neighbors before computing any distance converts 32 serial DRAM stalls into one overlapped batch, yielding proportionally larger gains. This is why hnswlib and faiss invest heavily in graph prefetch for large-scale indexes.

### macOS call-graph profile of the search phase (prefetch ON)

Profiled with `sample` (macOS `sample <pid> 8`), 6797 total samples, captured during search:

| Component | Samples | % of search_layer |
|-----------|--------:|------------------:|
| `NeonL2::compute` (distance) | 3,041 | **51%** |
| visited check + prefetch loop overhead | ~2,262 | **38%** |
| `priority_queue::push` (heap update) | ~618 | **10%** |
| `bzero` (visited array reset per query) | 23 | <1% |
| introsort (final result drain) | ~8 | <1% |

The 51% in `NeonL2::compute` does not mean 51% is pure arithmetic — the CPU is counted as "in NeonL2" even during stalls waiting for vector data to arrive from memory. With prefetch active, stall time is partially overlapped with prior computation, which is why `NeonL2` still accounts for a large fraction: it's where the hardware is doing useful work *and* where it is stalling on remaining misses.

**Verify locally:**
```
cmake -B build_noprefetch -DVECTORDB_PROFILE=ON -DVORTEXDB_NO_PREFETCH=ON
cmake -B build_prefetch   -DVECTORDB_PROFILE=ON
cmake --build build_noprefetch -j8 --target vortex_core vortex_storage vortex_server bench_profile
cmake --build build_prefetch   -j8 --target vortex_core vortex_storage vortex_server bench_profile
./build_noprefetch/bench/bench_profile --n 100000 --queries 5000 --ef 200
./build_prefetch/bench/bench_profile   --n 100000 --queries 5000 --ef 200
```

For a visual flamegraph, use Linux `perf record -g` + `flamegraph.pl` (brendangregg/FlameGraph). On macOS with Xcode: `xctrace record --template "Time Profiler"`. Both are equivalent to the `sample` call graph above but render as an SVG.

---

## Why graph prefetch required flat data structures first

All three early prefetch attempts failed for the same reason: **you cannot issue a prefetch hint without first computing the target address, and that computation must be cheaper than the cache miss you are trying to hide.**

| Data structure | Cost to compute `vec_ptr(nb)` |
|---------------|-------------------------------|
| `unordered_map<NodeId, vector<float>>` | hash lookup + pointer deref ≈ 30–50 ns |
| flat `vecs_flat_` | `data() + nb * dim` = 1 multiply-add ≈ 1 ns |

At 30–50 ns per address lookup, prefetching M0=32 neighbors per candidate adds ~1 µs of overhead — more than the cache benefit being sought. After converting to flat arrays, computing all 32 prefetch addresses costs ~32 ns total, making the net benefit positive.

**The lesson:** prefetch is only effective when address computation is cheaper than the memory latency being hidden. Flat data structures are a prerequisite, not an optimization on their own.

---

## Design Refactor: NodeId auto-assignment and user ID separation

This change is not a performance optimization — it is a correctness and API design fix that the flat array optimizations made necessary to address.

### Problem

The flat array optimizations (②③④) require NodeIds to be **dense integers starting from 0**. If any caller passes a non-sequential NodeId (e.g., `id=99999` as the second insert), `vecs_flat_` would resize to `99999 * 512 B ≈ 48 MB` and `adj0_` to `99999 * 128 B ≈ 12 MB`, wasting both memory and cache capacity.

The original code had no guard against this. The C++ `Engine::insert(collection, uint32_t id, vec)` accepted any id from the caller. The guarantee that ids were sequential existed only in the Python SDK layer (`_next_id` counter in `client.py`) — invisible to the C++ layer and bypassable by any C++ caller.

### Original design

```
Python SDK                          C++ Engine
  _next_id: Dict[str, int]   →      insert(col, uint32_t id, vec)
  "doc-a" → 0                →      HnswIndex::insert(0, vec)
  "doc-b" → 1                →      HnswIndex::insert(1, vec)
  id_map.json (persisted)
```

`HnswIndex::insert(NodeId id, const float* vec)` trusted the caller to provide valid sequential ids. No enforcement existed at the C++ level.

### Why users still provide IDs

A fully anonymous insert (`db.insert(vectors=vecs)` with no ids) would require the user to store and manage the system-assigned ids themselves to later identify search results or delete specific vectors. Since vectors represent real entities (documents, products, embeddings), the user always has a meaningful id for each one. Having the user provide the id and getting it back in search results is more useful than auto-assigned opaque integers.

### New design

```
Python SDK                          C++ Engine
  insert(ids=["doc-a"], ...)  →     insert(col, "doc-a", vec)
  (no id mapping logic)             ↓
                                    user_to_node["doc-a"] = idx.insert(vec)
                                                                ↓
                                                     HnswIndex: next_id_++ (auto-assign)
```

**`HnswIndex::insert(const float* vec) -> NodeId`** — auto-assigns `next_id_++` internally. No caller can pass an arbitrary NodeId.

**`HnswIndex::insert_for_recovery(NodeId id, const float* vec)`** — used only during WAL replay and re-inserts of existing user_ids. Updates `next_id_ = max(next_id_, id+1)` so future auto-assigns never collide.

**`Engine`** manages `user_to_node` and `node_to_user` maps per collection. On `insert("doc-a", vec)`:
- If "doc-a" is new: `NodeId = idx.insert(vec)`, store `"doc-a" ↔ NodeId`
- If "doc-a" already exists (re-insert after delete): use existing NodeId, call `insert_for_recovery(NodeId, vec)` — reuses the same slot, no gap created

**`remove`** does not erase from the id maps — the user_id → NodeId mapping is kept forever. This ensures that re-inserting the same user_id always reuses the same NodeId (no wasted flat array slots from insert-delete-insert cycles).

### WAL format change

Insert payload gains a `user_id` field so that recovery can rebuild the id maps from the WAL alone:

```
Before: [node_id: 4B][vec: dim*4B][metadata...]
After:  [node_id: 4B][uid_len: 2B][uid: N bytes][vec: dim*4B][metadata...]
```

On `checkpoint()`, the id maps are written to `id_map.bin` in the collection directory. On startup: load `id_map.bin` (covers checkpoint-era entries), then WAL replay fills in any entries added after the last checkpoint.

### What this guarantees

- **C++ layer**: `HnswIndex` can never receive a non-sequential NodeId through the normal insert path. The flat array assumption is enforced by the API, not by convention.
- **Python SDK**: all id mapping logic removed (`_str_to_int`, `_next_id`, `id_map.json`). The SDK converts user ids to strings and passes them to Engine directly.
- **Search results**: `SearchResult.user_id` (string) — callers always get back the id they inserted, never an internal NodeId.
