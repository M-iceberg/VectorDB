# HNSW Index Design

## What is HNSW?

HNSW (Hierarchical Navigable Small World) is an approximate nearest neighbor (ANN) index. Given a query vector, it finds the k vectors in the dataset that are closest to the query — not guaranteed to be the exact k nearest, but close enough in practice (recall@10 > 90% with default settings).

The alternative to ANN is brute-force: compute the distance from the query to every stored vector and sort. That's O(n) per query and becomes too slow at large n. HNSW gets sub-linear query time by building a graph structure at insert time that allows fast navigation at search time.

## The Core Idea: Layered Graph

HNSW builds a multi-layer graph over the vectors. Each node is a vector, and edges connect nearby vectors. The layers form a hierarchy:

```
Layer 2:  o ————————————— o          (few nodes, long-range edges)
Layer 1:  o — o — o ————— o — o      (more nodes, medium-range edges)
Layer 0:  o-o-o-o-o-o-o-o-o-o-o-o   (all nodes, short-range edges)
```

- **Layer 0** contains every node. Edges are short — neighbors are close in vector space.
- **Higher layers** contain fewer and fewer nodes, selected randomly. Edges are longer — they skip over large portions of the space.

This structure lets search start at the top (fast, coarse navigation) and descend to the bottom (slow, fine-grained navigation), like a skip list but in high-dimensional vector space.

## Layer Assignment

When a node is inserted, it's randomly assigned a maximum layer:

```cpp
l = floor(-ln(uniform(0,1)) * ml)   // ml = 1/ln(M)
```

**Step 1: `-ln(uniform(0,1))` follows an exponential distribution.**

`U = uniform(0,1)` is a random number between 0 and 1. `-ln(U)` follows an exponential distribution — a standard result from probability theory (inverse transform sampling). Small values are common, large values are rare and decay quickly:

```
-ln(U) value:   0.1  — high probability
                1.0  — medium probability
                3.0  — low probability
                5.0  — very rare
```

**Step 2: multiplying by `ml = 1/ln(M)` controls the decay rate.**

`ml` is a scaling factor. With M=16: `ml = 1/ln(16) ≈ 0.36`. This compresses the distribution so most values land near 0.

**Step 3: `floor(...)` maps continuous values to integer layers.**

```
0.0 ~ 0.99  →  layer 0
1.0 ~ 1.99  →  layer 1
2.0 ~ 2.99  →  layer 2
...
```

**Result:** with M=16, the probability of being assigned each layer is roughly:

```
layer 0:  ~94%
layer 1:  ~6%
layer 2:  ~0.4%
layer 3:  ~0.02%
```

Each layer up has approximately 1/M the nodes of the layer below. This is intentional — `ml = 1/ln(M)` is chosen specifically to make this ratio exactly 1/M, keeping the layer density in sync with the graph's connectivity parameter M.

A node assigned layer `l` participates in layers 0 through `l` — it has a neighbor list at each of those layers. Most nodes exist only at layer 0; a few are randomly promoted to higher layers and act as long-range "highway" nodes for fast traversal.

## Insert

The goal of insert is to add `q` into the graph with good neighbors at every layer it participates in, so that future searches passing through this region can navigate quickly.

`l` is computed by the layer assignment formula at the start of insert. A node assigned layer `l` appears in layers 0 through `l` — it has a neighbor list at each of those layers. Layers above `l` have no record of this node; searches at those layers cannot see it.

**Step 1: Greedy descent from `max_layer` to `l+1`**

There is one global entry point for the entire graph, stored in `Impl::entry_point`. It is always the node with the highest `max_layer`. Every insert and search starts from this node — it is not searched for, it is maintained automatically: the first node inserted becomes the entry point, and any later node assigned a higher layer replaces it (Step 3).

Say `q` is assigned layer 2 and the current `max_layer` is 5. Starting from `entry_point` at layer 5, we walk down to layer 3 (`l+1`), finding a node close to `q` to use as the entry point into `q`'s layers — but we stop before layer 2, because `q` doesn't exist in layers above `l`.

These upper layers are just used for navigation, not for building edges, so ef=1 (keep only the single closest candidate at each step) is enough — fast and cheap.

```
layer 5: start from entry_point, find closest neighbor, move there
layer 4: continue, find closest neighbor, move there
layer 3: continue, find closest neighbor, move there
layer 2: stop — begin Step 2
```

**Step 2: Beam search from `l` down to layer 0**

From layer `l` down to layer 0, find neighbors for `q` and connect edges at each layer.

This uses beam search with `ef_construction` candidates — much more thorough than Step 1, because these are the layers `q` actually lives in. Edge quality here directly affects search recall later.

At each layer:
1. Find the `ef_construction` closest candidates to `q`
2. Pick the best `M` of them as neighbors (M0 at layer 0)
3. Add bidirectional edges: `q → neighbor` and `neighbor → q`

Bidirectional edges are critical — any future search passing through this region from any direction can find `q`.

After each layer's beam search, the closest candidate found becomes the entry point for the next (lower) layer. This works because nodes exist across multiple layers — a node assigned layer `l` has a separate neighbor list at each layer 0 through `l`. The same node id is valid as an entry point at any of those layers, just with a different set of neighbors. So the closest node found at layer 2 is still present at layer 1 and layer 0, and is a better starting point than the one from layer above.

**Step 3: Update entry point**

If `q`'s assigned layer is higher than the current `max_layer`, `q` becomes the new global entry point. The entry point must always be one of the highest-layer nodes, since all searches start there.

**The overall tradeoff:**

Step 1 is "navigate quickly to `q`'s neighborhood." Step 2 is "carefully build edges at every layer `q` lives in." Step 3 is "keep global state consistent." Insert is slow (large `ef_construction`) by design — paying that cost at build time is what makes search fast later.

## Search

**Step 1: Greedy descent from `max_layer` to layer 1**

Same as insert Phase 1 — ef=1, follow the single closest neighbor at each layer, moving down until layer 1. This quickly navigates to a node close to the query.

Unlike insert, greedy descent stops at layer 1 instead of `assigned_layer + 1`. The reason: insert stops early because Phase 2 takes over from there. Search has no equivalent — it needs to do a full beam search at layer 0 to get good recall, so greedy descent just needs to find a good entry point into layer 0.

Precision is not lost by using ef=1 in the upper layers. High layers are sparse with long edges — each step covers a large distance and quickly reaches the right neighborhood. The real precision work happens at layer 0.

**Step 2: Full beam search at layer 0**

Layer 0 contains all nodes with the shortest, densest edges. This is where the precise nearest neighbor search happens.

`beam_width = max(k, ef_search)` — beam_width must be at least k because some candidates may be tombstoned and filtered out. If `beam_width == k` and some are tombstoned, fewer than k results would be returned. `ef_search > k` is the main recall vs. speed tradeoff knob: larger values explore more candidates and find better results, at the cost of more distance computations.

**Step 3: Filter tombstones, return top-k**

`candidates` is already sorted ascending by distance. Iterate in order, skip tombstoned nodes, take the first k live ones. Return fewer than k if the index has fewer than k live nodes.

## Remove

Sets `tombstone = true` on the node and decrements `live_count`. No edges are removed or relinked — the node stays in the graph as a routing intermediary. Search may still traverse through tombstoned nodes to reach live neighbors, but will never include them in results.

The tradeoff: deletion is O(1) and requires no graph restructuring, but tombstoned nodes accumulate over time and slightly degrade search quality (wasted traversal through dead nodes). Full compaction — removing tombstoned nodes and relinking their neighbors — is left for a later milestone.

## Size

Returns `live_count` — the number of non-tombstoned nodes. Maintained incrementally: +1 on insert, -1 on remove. O(1).

## Beam Search (`search_layer`)

`search_layer` is the core subroutine used by both insert and search. Given a query, an entry point `ep`, a beam width `ef`, and a layer number, it returns the closest `ef` nodes in that layer.

**What is beam search?**

Beam search sits between two extremes:

- **Greedy search (ef=1):** follow only the single closest neighbor at each step. Fast, but easily gets stuck in a local optimum and misses the true nearest neighbors.
- **Exhaustive search (ef=∞):** explore every possible path. Always finds the optimum, but too slow at scale.
- **Beam search (ef=N):** keep the N closest candidates at each step and explore all of them. `ef` is the beam width — larger means more accurate but slower.

**Parameters:**

- **`ep`** — entry point, the starting node for this layer's search
- **`ef`** — beam width, how many candidates to keep at once

`ef` is called with different values depending on context:

| caller | ef value | why |
|--------|----------|-----|
| insert Step 1 (greedy descent) | `1` | just needs a quick entry point, no precision needed |
| insert Step 2 (build edges) | `ef_construction` | needs high-quality neighbors for good recall later |
| search | `max(k, ef_search)` | tradeoff knob between recall and query speed |

**How it works:**

Maintains two heaps:
- **C** (min-heap) — candidates to explore, closest first
- **W** (max-heap) — best results found so far, worst at top for easy eviction

Initialize both with `ep`. Then loop:
1. Pop the closest candidate `c` from C
2. If `c` is already farther than the worst result in W, stop — further exploration cannot improve W
3. For each unvisited neighbor `e` of `c` at this layer:
   - If `dist(e) < dist(worst in W)` or `|W| < ef`: add `e` to both C and W
   - If `|W| > ef`: evict the worst from W

Return W sorted ascending by distance.

**Why two heaps work:**

C and W each solve a different problem:

- **C (min-heap) answers "where to explore next":** always expands the closest candidate first, so search pushes in the most promising direction and doesn't waste time on obviously bad paths.
- **W (max-heap) answers "when to stop and what to keep":** maintains the best `ef` results seen so far, with the worst at the top for easy comparison and eviction. When the closest candidate in C is already farther than W's worst result, further exploration cannot improve W — safe to stop.

The key is that C controls exploration order and W controls termination and result quality. Together they guarantee: no close node is skipped, and the result set is always the best `ef` nodes seen so far.

## Delete (Tombstone)

Deleting a node just sets `tombstone = true` and decrements `live_count`. The node stays in the graph and can still be traversed as a routing intermediary — its edges are not removed. Search skips tombstoned nodes in the final result but may still pass through them to reach live nodes.

This avoids the complexity of relinking neighbors on deletion. The tradeoff is that tombstoned nodes accumulate over time and slightly degrade search quality. Full graph compaction is left for a later milestone.

## HnswNode

Each node in the graph stores:

```cpp
struct HnswNode {
    NodeId id;
    int    layer;                              // highest layer this node appears in
    vector<vector<NodeId>> neighbors;          // neighbors[level][i]
    bool   tombstone;
    // raw float vector is NOT stored here — lives separately in vecs (or VectorFile later)
};
```

The float vector is stored separately in `Impl::vecs` (a `NodeId → vector<float>` map). This keeps the graph structure compact — `HnswNode` only holds edges, not the large float arrays. Day 13 will replace `vecs` with `VectorFile` for on-disk storage.

---

## Code Structure

## HNSW Index Structure

Three layers, from outside in:

```
HnswIndex          ← public API, the only thing callers see
  └── Impl         ← all internal state and logic (pimpl pattern)
        └── dc     ← pluggable distance compute, chosen at runtime
```

### HnswIndex

The public-facing class defined in `hnsw_index.h`. Exposes only three operations: `insert`, `search`, `remove`. Internal details are hidden behind a `unique_ptr<Impl>` so changes to the implementation don't force recompilation of callers.

### Impl

The real data lives here:

- `nodes` — graph structure, maps `NodeId → HnswNode` (layer assignment + neighbor lists)
- `vecs` — raw float vectors, maps `NodeId → vector<float>` (Day 13: will be replaced by `VectorFile` for on-disk storage)
- `entry_point`, `max_layer` — global entry point into the graph and the highest layer reached so far
- `live_count` — count of non-tombstoned nodes
- `rng`, `ml` — random layer assignment state (`ml = 1/ln(M)`)
- `dc` — the distance compute instance (see below)

### dc — DistanceCompute

`DistanceCompute` is an abstract base class. Concrete subclasses implement different distance metrics and SIMD paths:

```
DistanceCompute  (abstract)
  ├── NaiveL2 / NaiveCosine / NaiveIP     ← scalar fallback, always compiled
  └── NeonL2  / NeonCosine / NeonIP       ← ARM NEON (compiled on ARM only)
      AvxL2   / AvxCosine  / AvxIP        ← x86 AVX2 (compiled on x86 only)
```

`DistanceCompute::create(metric)` is a factory that returns the right implementation for the current platform and requested metric. After construction, all distance calls go through `dc->compute(a, b, dim)` — `Impl` doesn't need to know which path is running.

### Why pimpl?

`HnswIndex`'s header exposes only `unique_ptr<Impl>`, so the internal layout of `Impl` is invisible to callers. Adding fields to `Impl` (e.g. during Week 3 storage integration) doesn't require recompiling anything that includes `hnsw_index.h`.

### Why `unique_ptr<DistanceCompute>`?

`DistanceCompute` is an abstract class — it has at least one pure virtual function (`= 0`), so it cannot be instantiated directly. You can only hold it through a pointer to a concrete subclass:

```cpp
// ❌ cannot instantiate an abstract class
DistanceCompute dc;

// ✅ hold a pointer to a concrete subclass
std::unique_ptr<DistanceCompute> dc = std::make_unique<NeonL2>();
dc->compute(a, b, dim);  // dispatches to NeonL2::compute at runtime
```

This is **polymorphism** — calling a virtual function through a base class pointer dispatches to the correct subclass implementation at runtime.

`unique_ptr` is used instead of a raw pointer for automatic memory management (RAII). When `unique_ptr` goes out of scope, its destructor automatically calls `delete` — no manual cleanup needed:

```cpp
// raw pointer — must remember to delete, leaks if an exception is thrown
DistanceCompute* dc = new NeonL2();
delete dc;  // manual

// unique_ptr — automatically deleted when Impl is destroyed
std::unique_ptr<DistanceCompute> dc = std::make_unique<NeonL2>();
```

### Lifetime chain

`dc`'s lifetime is bound to `Impl`, which is bound to `HnswIndex`. When `HnswIndex` is destroyed, everything cleans up automatically:

```
HnswIndex destroyed
  → impl_ (unique_ptr<Impl>) destroyed
    → Impl destroyed
      → dc (unique_ptr<DistanceCompute>) destroyed
        → NeonL2 deleted
```
