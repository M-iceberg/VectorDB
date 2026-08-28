// -----------------------------------------------------------------------------
// hnsw_index.cpp — HNSW graph implementation
//
// Algorithm: Malkov & Yashunin 2018 "Efficient and robust approximate nearest
// neighbor search using Hierarchical Navigable Small World graphs".
//
// Overview of the three main operations:
//
//   insert(id, vec)
//     1. Assign a random layer l ~ floor(-ln(U) * ml), where ml = 1/ln(M).
//        This gives an exponential layer distribution: most nodes at layer 0,
//        exponentially fewer at higher layers.
//     2. Greedy descent from the current max_layer down to l+1, keeping only
//        the single closest candidate at each level (ef=1). This quickly finds
//        a good entry point for the beam search phase.
//     3. Beam search (ef_construction candidates) from layer min(l, max_layer)
//        down to 0. At each level, select M (or M0 at layer 0) neighbors from
//        the candidates and add bidirectional edges. Prune to M_max if a
//        neighbor list exceeds capacity.
//     4. If l > max_layer, update the global entry point.
//
//   search(query, k, ef_search)
//     1. Greedy descent from max_layer to layer 1 (ef=1) to find the entry
//        point for the base layer.
//     2. Beam search at layer 0 with ef = max(k, ef_search).
//     3. Filter tombstones, return top-k sorted by ascending distance.
//
//   remove(id)
//     Soft delete: sets tombstone=true and decrements live_count. The node
//     remains in the graph as a routing intermediary so no relinking is needed.
//     Tombstoned nodes are skipped in search results but may still be traversed.
//
// Beam search (search_layer):
//   Maintains two heaps:
//     C — min-heap of candidates to explore (closest first)
//     W — max-heap of best results seen so far (furthest at top for pruning)
//   Expand the closest candidate in C; for each unvisited neighbor e:
//     if dist(e) < dist(furthest in W) or |W| < ef: add e to both C and W.
//     if |W| > ef: evict the furthest from W.
//   Stop when the closest unexplored candidate is farther than the worst
//   result in W (further exploration cannot improve the result set).
// -----------------------------------------------------------------------------
#include "hnsw_index.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <mutex>
#include <queue>
#include <random>
#include <stdexcept>
#include <thread>

namespace vectordb {

// ---------------------------------------------------------------------------
// Thread-local visited table — generation counter eliminates memset per call.
//
// Previously: std::vector<uint8_t> visited(max_node_id_+1, 0) allocated and
// zero-initialised on every search_layer call. At N=1M this was 1 MB of
// zeroing ~2 times per insert, totalling ~2 TB of memset across a 1M build.
//
// Now: each thread owns a stamps[] array and a uint32_t generation counter.
// Marking a node visited = stamps[id] = gen (one store). Checking = stamps[id]==gen
// (one load). "Resetting" = ++gen (one add). No memset, no heap allocation
// after the first call on a thread. On gen overflow (2^32 calls) we do one
// full reset — negligible amortised cost.
// ---------------------------------------------------------------------------
struct VisitedTable {
    std::vector<uint32_t> stamps;
    uint32_t gen = 0;

    uint32_t acquire(size_t needed) {
        if (stamps.size() < needed) stamps.resize(needed, 0);
        if (++gen == 0) {
            std::fill(stamps.begin(), stamps.end(), 0);
            gen = 1;
        }
        return gen;
    }
};
thread_local VisitedTable tl_visited;

// ---------------------------------------------------------------------------
// Impl — private state
// ---------------------------------------------------------------------------

struct HnswIndex::Impl {
    HnswConfig cfg;
    std::unique_ptr<DistanceCompute> dc;

    // Graph structure: flat array indexed by NodeId.
    // nodes_flat_[id] is valid when node.id != kInvalidNode.
    std::vector<HnswNode> nodes_flat_;

    void grow_nodes(NodeId id) {
        if (id >= nodes_flat_.size())
            nodes_flat_.resize(id + 1);
    }
    bool node_exists(NodeId id) const {
        return id < nodes_flat_.size() && nodes_flat_[id].id != kInvalidNode;
    }

    // Performance mode uses unified per-node storage: each node occupies one
    // contiguous block of (cfg.M0 + cfg.dim) uint32_t elements, laid out as:
    //   [adj0 neighbors: cfg.M0 × NodeId] [vector data: cfg.dim × float]
    //
    // Collocating adj0 and vec in one array means a prefetch of node n's block
    // warms both its neighbor list (for when n becomes a candidate) and its
    // vector (for dist_q). With separate arrays the two accesses were unrelated
    // random-access patterns the hardware prefetcher could not predict.
    //
    // Compact mode keeps only cfg.M0 adjacency entries here and obtains vectors
    // from external_vectors_, normally the mmap-backed VectorFile.
    // node_stride_ is measured in uint32_t units; NodeId and float are both 4B.
    size_t                node_stride_ = 0;
    std::vector<uint32_t> node_blocks_;
    std::vector<uint8_t>  adj0_count_;
    const float*          external_vectors_ = nullptr;

    void grow_node_block(NodeId id) {
        if (id > max_node_id_) max_node_id_ = id;
        size_t needed = (size_t)(id + 1) * node_stride_;
        if (needed > node_blocks_.size()) {
            node_blocks_.resize(needed, 0);
            adj0_count_.resize(id + 1, 0);
        }
    }
    void grow_adj0(NodeId id) { grow_node_block(id); }

    NodeId*       adj0_ptr(NodeId id)       { return reinterpret_cast<NodeId*>(node_blocks_.data() + (size_t)id * node_stride_); }
    const NodeId* adj0_ptr(NodeId id) const { return reinterpret_cast<const NodeId*>(node_blocks_.data() + (size_t)id * node_stride_); }
    float* internal_vec_ptr(NodeId id) {
        return reinterpret_cast<float*>(
            node_blocks_.data() + (size_t)id * node_stride_ + cfg.M0);
    }
    const float* vec_ptr(NodeId id) const {
        if (cfg.store_vectors)
            return reinterpret_cast<const float*>(
                node_blocks_.data() + (size_t)id * node_stride_ + cfg.M0);
        if (!external_vectors_)
            throw std::logic_error("HnswIndex: compact vector source is not set");
        return external_vectors_ + static_cast<size_t>(id) * cfg.dim;
    }

    void store_vec(NodeId id, const float* vec) {
        grow_node_block(id);
        if (cfg.store_vectors)
            std::memcpy(internal_vec_ptr(id), vec, cfg.dim * sizeof(float));
    }

    // Global graph state — atomic so multi-threaded insert can read/update safely.
    // On single-threaded paths load()/store() with relaxed ordering is sufficient.
    std::atomic<NodeId> entry_point_{kInvalidNode};
    std::atomic<int>    max_layer_{-1};
    std::atomic<size_t> live_count_{0};
    std::atomic<NodeId> next_id_{0};
    NodeId max_node_id_ = 0;

    // Per-node lock striping for neighbor list modification during parallel insert.
    // 256 stripes: false-sharing rate ~0.4% for random NodeIds, no resize issues
    // (std::mutex is not movable, so std::array avoids the vector-resize problem).
    static constexpr size_t kStripes = 256;
    mutable std::array<std::mutex, kStripes> stripe_locks_;
    mutable std::mutex entry_mu_;

    std::mutex& stripe_lock(NodeId id) const { return stripe_locks_[id % kStripes]; }

    std::mt19937 rng{42};
    double ml;

    explicit Impl(HnswConfig c)
        : cfg(c)
        , dc(DistanceCompute::create(c.metric))
        , node_stride_(static_cast<size_t>(c.M0) +
                       (c.store_vectors ? c.dim : 0))
        , ml(1.0 / std::log(static_cast<double>(c.M)))
    {}

    float dist(NodeId a, NodeId b) const {
        return dc->compute(vec_ptr(a), vec_ptr(b), cfg.dim);
    }
    float dist_q(const float* query, NodeId b) const {
        return dc->compute(query, vec_ptr(b), cfg.dim);
    }

    int assign_layer() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return static_cast<int>(-std::log(u(rng)) * ml);
    }

    // Thread-safe layer assignment — uses a thread_local RNG so no lock is needed.
    int assign_layer_mt() {
        thread_local std::mt19937 tl_rng{std::random_device{}()};
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return static_cast<int>(-std::log(u(tl_rng)) * ml);
    }

    // Beam search at a single layer.  Uses the thread_local VisitedTable so:
    //   (a) no heap allocation per call, (b) safe for concurrent callers (each
    //       thread has its own table).
    std::vector<std::pair<float, NodeId>> search_layer(
        const float* query, NodeId ep, int ef, int layer,
        bool concurrent_writes = false) const
    {
        using DistId = std::pair<float, NodeId>;
        auto cmp_max = [](const DistId& a, const DistId& b) { return a.first < b.first; };
        auto cmp_min = [](const DistId& a, const DistId& b) { return a.first > b.first; };
        std::priority_queue<DistId, std::vector<DistId>, decltype(cmp_max)> W(cmp_max);
        std::priority_queue<DistId, std::vector<DistId>, decltype(cmp_min)> C(cmp_min);

        // Generation-counter visited set — O(1) reset, no memset per call.
        uint32_t gen = tl_visited.acquire(max_node_id_ + 1);
        auto&    stamps = tl_visited.stamps;

        float ep_dist = dist_q(query, ep);
        W.push({ep_dist, ep});
        C.push({ep_dist, ep});
        stamps[ep] = gen;

        while (!C.empty()) {
            auto [candidate_dist, candidate] = C.top(); C.pop();
            if (candidate_dist > W.top().first) break;

            auto expand = [&](NodeId neighbor) {
                if (neighbor == kInvalidNode) return;
                if (neighbor >= stamps.size() || stamps[neighbor] == gen) return;
                stamps[neighbor] = gen;

                float neighbor_dist = dist_q(query, neighbor);
                float worst_dist    = W.top().first;
                if (neighbor_dist < worst_dist || static_cast<int>(W.size()) < ef) {
                    C.push({neighbor_dist, neighbor});
                    W.push({neighbor_dist, neighbor});
                    if (static_cast<int>(W.size()) > ef) W.pop();
                }
            };

            if (layer == 0) {
                // Parallel construction updates layer-0 edges in place. Fixed
                // capacity prevents reallocations, but unsynchronised reads of
                // adj0_count_/neighbor IDs would still be a C++ data race. Copy
                // the published prefix under the same stripe lock used by
                // writers. Normal query search keeps the direct, lock-free path
                // because Engine never overlaps it with index construction.
                if (concurrent_writes) {
                    std::vector<NodeId> layer_nbrs;
                    {
                        std::lock_guard<std::mutex> lk(stripe_lock(candidate));
                        int cnt = adj0_count_[candidate];
                        const NodeId* nbrs = adj0_ptr(candidate);
                        layer_nbrs.assign(nbrs, nbrs + cnt);
                    }
#ifndef VORTEXDB_NO_PREFETCH
                    for (NodeId neighbor : layer_nbrs)
                        __builtin_prefetch(vec_ptr(neighbor), 0, 0);
#endif
                    for (NodeId neighbor : layer_nbrs) expand(neighbor);
                    continue;
                }

                int cnt = adj0_count_[candidate];
                const NodeId* nbrs = adj0_ptr(candidate);
#ifndef VORTEXDB_NO_PREFETCH
                for (int k = 0; k < cnt; ++k)
                    __builtin_prefetch(vec_ptr(nbrs[k]), 0, 0);
#endif
                for (int k = 0; k < cnt; ++k) expand(nbrs[k]);
            } else {
                // Upper-layer neighbor lists are modified by concurrent inserts.
                // Copy under the stripe lock so we never read a vector mid-assign
                // or mid-push_back (which changes internal pointers → use-after-free).
                // Layer-0 (adj0) is a pre-allocated fixed array — no lock needed there.
                std::vector<NodeId> layer_nbrs;
                {
                    std::lock_guard<std::mutex> lk(stripe_lock(candidate));
                    const auto& cn = nodes_flat_[candidate].neighbors;
                    if (layer < static_cast<int>(cn.size()))
                        layer_nbrs = cn[layer];
                }
                for (NodeId neighbor : layer_nbrs) expand(neighbor);
            }
        }

        std::vector<DistId> result;
        result.reserve(W.size());
        while (!W.empty()) { result.push_back(W.top()); W.pop(); }
        std::sort(result.begin(), result.end());
        return result;
    }

    std::vector<NodeId> select_neighbors(
        const std::vector<std::pair<float, NodeId>>& candidates,
        int M_max) const
    {
        std::vector<NodeId> result;
        result.reserve(M_max);
        for (auto& [d_q_e, e] : candidates) {
            if (static_cast<int>(result.size()) >= M_max) break;
            if (!cfg.heuristic) { result.push_back(e); continue; }
            bool dominated = false;
            for (NodeId r : result) {
                if (dist(e, r) < d_q_e) { dominated = true; break; }
            }
            if (!dominated) result.push_back(e);
        }
        return result;
    }

    // Single-direction edge add from→to at layer (no locking, caller is responsible).
    // Used by: add_edge() (single-threaded), and insert_with_id_mt() where the
    // caller locks the destination node before calling for reverse edges.
    void add_directed(NodeId from, NodeId to, int layer, int M_max) {
        if (layer == 0) {
            uint8_t& cnt  = adj0_count_[from];
            NodeId*  nbrs = adj0_ptr(from);
            if (cnt < static_cast<uint8_t>(M_max)) {
                nbrs[cnt++] = to;
            } else {
                std::vector<std::pair<float, NodeId>> candidates;
                candidates.reserve(cnt + 1);
                for (int k = 0; k < cnt; ++k)
                    candidates.push_back({dist(from, nbrs[k]), nbrs[k]});
                candidates.push_back({dist(from, to), to});
                std::sort(candidates.begin(), candidates.end());
                auto selected = select_neighbors(candidates, M_max);
                cnt = static_cast<uint8_t>(selected.size());
                for (int k = 0; k < (int)selected.size(); ++k) nbrs[k] = selected[k];
                for (int k = cnt; k < M_max; ++k) nbrs[k] = kInvalidNode;
            }
        } else {
            auto& nbrs = nodes_flat_[from].neighbors[layer];
            if (static_cast<int>(nbrs.size()) < M_max) {
                nbrs.push_back(to);
            } else {
                std::vector<std::pair<float, NodeId>> candidates;
                candidates.reserve(nbrs.size() + 1);
                for (NodeId n : nbrs)
                    candidates.push_back({dist(from, n), n});
                candidates.push_back({dist(from, to), to});
                std::sort(candidates.begin(), candidates.end());
                nbrs = select_neighbors(candidates, M_max);
            }
        }
    }

    // Bidirectional edge (single-threaded path only).
    void add_edge(NodeId u, NodeId v, int layer, int M_max) {
        add_directed(u, v, layer, M_max);
        add_directed(v, u, layer, M_max);
    }
};

// ---------------------------------------------------------------------------
// Single-threaded insert helper (used by insert() and insert_for_recovery())
// ---------------------------------------------------------------------------

static void insert_with_id(HnswIndex::Impl& I, NodeId id, const float* vec) {
    I.store_vec(id, vec);

    bool is_new         = !I.node_exists(id);
    bool was_tombstoned = !is_new && I.nodes_flat_[id].tombstone;

    int assigned_layer = I.assign_layer();
    I.grow_nodes(id);
    I.grow_adj0(id);
    HnswNode& node  = I.nodes_flat_[id];
    node.id         = id;
    node.layer      = assigned_layer;
    node.tombstone  = false;
    node.neighbors.assign(assigned_layer + 1, {});
    if (is_new || was_tombstoned)
        I.live_count_.fetch_add(1, std::memory_order_relaxed);

    if (I.entry_point_.load(std::memory_order_relaxed) == kInvalidNode) {
        I.entry_point_.store(id, std::memory_order_relaxed);
        I.max_layer_.store(assigned_layer, std::memory_order_relaxed);
        return;
    }

    NodeId entry_point = I.entry_point_.load(std::memory_order_relaxed);
    int    cur_max     = I.max_layer_.load(std::memory_order_relaxed);

    for (int level = cur_max; level > assigned_layer; --level) {
        auto candidates = I.search_layer(vec, entry_point, 1, level);
        if (!candidates.empty()) entry_point = candidates[0].second;
    }

    for (int level = std::min(assigned_layer, cur_max); level >= 0; --level) {
        int max_neighbors = (level == 0) ? I.cfg.M0 : I.cfg.M;
        auto candidates   = I.search_layer(vec, entry_point, I.cfg.ef_construction, level);
        auto neighbors    = I.select_neighbors(candidates, max_neighbors);

        auto& node_nbrs = I.nodes_flat_[id].neighbors;
        if (static_cast<int>(node_nbrs.size()) <= level)
            node_nbrs.resize(level + 1);

        for (NodeId neighbor : neighbors)
            I.add_edge(id, neighbor, level, max_neighbors);

        if (!candidates.empty()) entry_point = candidates[0].second;
    }

    if (assigned_layer > cur_max) {
        I.entry_point_.store(id, std::memory_order_relaxed);
        I.max_layer_.store(assigned_layer, std::memory_order_relaxed);
    }
}

// ---------------------------------------------------------------------------
// Multi-threaded insert helper (used by insert_batch_mt())
//
// Concurrency model (matches hnswlib):
//   - search_layer: lock-free dirty reads (ANN tolerates stale edges)
//   - new node's own data: owned exclusively by this thread until edges are added
//   - back-edges (neighbor → new_node): protected by stripe_lock(neighbor)
//   - entry_point / max_layer updates: protected by entry_mu_
// ---------------------------------------------------------------------------

static void insert_with_id_mt(HnswIndex::Impl& I, NodeId id, const float* vec) {
    I.store_vec(id, vec);

    int assigned_layer = I.assign_layer_mt();

    // Initialize node data under stripe_lock(id) so that any concurrent
    // search_layer call that copies this node's upper-layer neighbor list
    // (also under stripe_lock(id)) sees a consistent, fully-assigned vector.
    {
        std::lock_guard<std::mutex> lk(I.stripe_lock(id));
        I.nodes_flat_[id].id        = id;
        I.nodes_flat_[id].layer     = assigned_layer;
        I.nodes_flat_[id].tombstone = false;
        I.nodes_flat_[id].neighbors.assign(assigned_layer + 1, {});
        // Pre-reserve upper-layer vectors so push_back never reallocates
        // while another thread holds the same stripe lock for a copy.
        for (int lv = 1; lv <= assigned_layer; ++lv)
            I.nodes_flat_[id].neighbors[lv].reserve(I.cfg.M);
    }
    I.live_count_.fetch_add(1, std::memory_order_relaxed);

    // Take a consistent snapshot of entry_point + max_layer.
    // The first-node case is also handled here so entry_point and max_layer
    // are always updated atomically (prevents a second thread from seeing
    // entry_point = X but max_layer = -1).
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

    // Phase 1: greedy descent to assigned_layer+1.
    for (int level = cur_max; level > assigned_layer; --level) {
        auto cands = I.search_layer(vec, entry_point, 1, level, true);
        if (!cands.empty()) entry_point = cands[0].second;
    }

    // Phase 2: beam search + bidirectional edges.
    for (int level = std::min(assigned_layer, cur_max); level >= 0; --level) {
        int M_max  = (level == 0) ? I.cfg.M0 : I.cfg.M;
        auto cands = I.search_layer(
            vec, entry_point, I.cfg.ef_construction, level, true);
        auto nbrs  = I.select_neighbors(cands, M_max);

        for (NodeId nbr : nbrs) {
            // Both forward (id→nbr) and back (nbr→id) edges must hold the
            // respective node's stripe lock. Without locking the forward edge,
            // a concurrent thread adding a back-edge to id (under stripe_lock(id))
            // races with this thread updating adj0_count_[id], losing edges and
            // degrading recall. Both locks are taken separately (never simultaneously)
            // so there is no deadlock risk.
            {
                std::lock_guard<std::mutex> lk(I.stripe_lock(id));
                I.add_directed(id, nbr, level, M_max);
            }
            {
                std::lock_guard<std::mutex> lk(I.stripe_lock(nbr));
                I.add_directed(nbr, id, level, M_max);
            }
        }

        if (!cands.empty()) entry_point = cands[0].second;
    }

    // Update global entry point if we reached a new max layer.
    if (assigned_layer > cur_max) {
        std::lock_guard<std::mutex> lk(I.entry_mu_);
        if (assigned_layer > I.max_layer_.load(std::memory_order_relaxed)) {
            I.max_layer_.store(assigned_layer, std::memory_order_relaxed);
            I.entry_point_.store(id, std::memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------------------
// HnswIndex public API
// ---------------------------------------------------------------------------

HnswIndex::HnswIndex(HnswConfig cfg) : impl_(std::make_unique<Impl>(cfg)) {}
HnswIndex::~HnswIndex() = default;

void HnswIndex::set_external_vector_base(const float* vectors) {
    impl_->external_vectors_ = vectors;
}

NodeId HnswIndex::insert(const float* vec) {
    auto& I  = *impl_;
    NodeId id = I.next_id_.fetch_add(1, std::memory_order_relaxed);
    insert_with_id(I, id, vec);
    return id;
}

void HnswIndex::insert_for_recovery(NodeId id, const float* vec) {
    auto& I = *impl_;
    NodeId cur = I.next_id_.load(std::memory_order_relaxed);
    while (id + 1 > cur) {
        if (I.next_id_.compare_exchange_weak(cur, id + 1,
                std::memory_order_relaxed, std::memory_order_relaxed))
            break;
    }
    // If the node was already loaded from a graph snapshot, skip re-insertion
    // so WAL replay is idempotent and doesn't corrupt existing edges.
    if (I.node_exists(id) && !I.nodes_flat_[id].tombstone)
        return;
    insert_with_id(I, id, vec);
}

// Parallel batch insert: pre-assigns all NodeIds, pre-grows all arrays, then
// spawns num_threads threads each inserting its slice. Returns the first NodeId
// assigned (IDs are first_id, first_id+1, ..., first_id+count-1).
NodeId HnswIndex::insert_batch_mt(const float* vecs, size_t count, int num_threads) {
    NodeId first_id = reserve_ids(count);
    insert_reserved_batch_mt(first_id, vecs, count, num_threads);
    return first_id;
}

NodeId HnswIndex::reserve_ids(size_t count) {
    auto& I = *impl_;
    if (count == 0) return I.next_id_.load(std::memory_order_relaxed);
    if (count > static_cast<size_t>(std::numeric_limits<NodeId>::max()))
        throw std::overflow_error("HnswIndex: batch is too large");
    NodeId amount = static_cast<NodeId>(count);
    NodeId first_id = I.next_id_.fetch_add(amount, std::memory_order_relaxed);
    if (first_id > std::numeric_limits<NodeId>::max() - (amount - 1)) {
        I.next_id_.fetch_sub(amount, std::memory_order_relaxed);
        throw std::overflow_error("HnswIndex: NodeId space exhausted");
    }
    return first_id;
}

void HnswIndex::insert_reserved_batch_mt(
    NodeId first_id, const float* vecs, size_t count, int num_threads) {
    if (count == 0) return;
    if (!vecs) throw std::invalid_argument("HnswIndex: vectors must not be null");
    auto& I = *impl_;
    NodeId last_id  = first_id + static_cast<NodeId>(count) - 1;

    // Pre-grow arrays single-threadedly to avoid resize races.
    I.grow_node_block(last_id);
    I.grow_nodes(last_id);

    if (num_threads <= 0)
        num_threads = static_cast<int>(std::thread::hardware_concurrency());
    num_threads = std::max(1, std::min(num_threads, static_cast<int>(count)));

    if (num_threads == 1) {
        for (size_t i = 0; i < count; ++i)
            insert_with_id_mt(I, first_id + i, vecs + i * I.cfg.dim);
    } else {
        std::vector<std::thread> threads;
        threads.reserve(num_threads);
        size_t chunk = (count + num_threads - 1) / num_threads;
        for (int t = 0; t < num_threads; ++t) {
            size_t start = (size_t)t * chunk;
            size_t end   = std::min(start + chunk, count);
            if (start >= count) break;
            threads.emplace_back([&, start, end] {
                for (size_t i = start; i < end; ++i)
                    insert_with_id_mt(I, first_id + i, vecs + i * I.cfg.dim);
            });
        }
        for (auto& t : threads) t.join();
    }
}

std::vector<std::pair<float, NodeId>> HnswIndex::search(
    const float* query, int k, int ef_search) const
{
    auto& I = *impl_;
    if (I.entry_point_.load(std::memory_order_relaxed) == kInvalidNode) return {};

    NodeId entry_point = I.entry_point_.load(std::memory_order_acquire);
    int    max_layer   = I.max_layer_.load(std::memory_order_acquire);

    for (int level = max_layer; level > 0; --level) {
        auto candidates = I.search_layer(query, entry_point, 1, level);
        if (!candidates.empty()) entry_point = candidates[0].second;
    }

    int beam_width = std::max(k, ef_search);
    auto candidates = I.search_layer(query, entry_point, beam_width, 0);

    std::vector<std::pair<float, NodeId>> result;
    result.reserve(k);
    for (auto& [dist, id] : candidates) {
        if (static_cast<int>(result.size()) >= k) break;
        if (I.node_exists(id) && !I.nodes_flat_[id].tombstone)
            result.push_back({dist, id});
    }
    return result;
}

std::vector<std::vector<std::pair<float, NodeId>>> HnswIndex::search_batch(
    const float* queries, int n_queries, int k, int ef_search,
    int num_threads) const
{
    std::vector<std::vector<std::pair<float, NodeId>>> results(n_queries);
    if (n_queries == 0) return results;

    const size_t dim = impl_->cfg.dim;
    if (num_threads <= 0) num_threads = static_cast<int>(std::thread::hardware_concurrency());
    num_threads = std::max(1, std::min(num_threads, n_queries));

    if (num_threads == 1) {
        for (int i = 0; i < n_queries; ++i)
            results[i] = search(queries + (size_t)i * dim, k, ef_search);
        return results;
    }

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    int chunk = (n_queries + num_threads - 1) / num_threads;
    for (int t = 0; t < num_threads; ++t) {
        int start = t * chunk;
        int end   = std::min(start + chunk, n_queries);
        if (start >= n_queries) break;
        threads.emplace_back([&, start, end] {
            for (int i = start; i < end; ++i)
                results[i] = search(queries + (size_t)i * dim, k, ef_search);
        });
    }
    for (auto& th : threads) th.join();
    return results;
}

void HnswIndex::remove(NodeId id) {
    auto& I = *impl_;
    if (!I.node_exists(id) || I.nodes_flat_[id].tombstone) return;
    I.nodes_flat_[id].tombstone = true;
    I.live_count_.fetch_sub(1, std::memory_order_relaxed);
}

size_t HnswIndex::size() const { return impl_->live_count_.load(); }

size_t HnswIndex::neighbor_count(NodeId id, int layer) const {
    auto& I = *impl_;
    if (!I.node_exists(id)) return 0;
    if (layer == 0) return I.adj0_count_[id];
    const auto& nbrs = I.nodes_flat_[id].neighbors;
    if (layer >= static_cast<int>(nbrs.size())) return 0;
    return nbrs[layer].size();
}

int HnswIndex::node_layer(NodeId id) const {
    auto& I = *impl_;
    if (!I.node_exists(id)) return -1;
    return I.nodes_flat_[id].layer;
}

std::vector<NodeId> HnswIndex::neighbors_of(NodeId id, int layer) const {
    auto& I = *impl_;
    if (!I.node_exists(id)) return {};
    if (layer == 0) {
        int cnt = I.adj0_count_[id];
        const NodeId* p = I.adj0_ptr(id);
        return std::vector<NodeId>(p, p + cnt);
    }
    const auto& nbrs = I.nodes_flat_[id].neighbors;
    if (layer >= static_cast<int>(nbrs.size())) return {};
    return nbrs[layer];
}

// ---------------------------------------------------------------------------
// Serialization support
// ---------------------------------------------------------------------------

std::vector<HnswIndex::NodeData> HnswIndex::snapshot() const {
    auto& I = *impl_;
    std::vector<NodeData> out;
    out.reserve(I.nodes_flat_.size());
    for (auto& node : I.nodes_flat_) {
        if (node.id == kInvalidNode) continue;
        NodeData nd;
        nd.id        = node.id;
        nd.layer     = node.layer;
        nd.tombstone = node.tombstone;
        nd.vec = std::vector<float>(I.vec_ptr(node.id), I.vec_ptr(node.id) + I.cfg.dim);
        int cnt = I.adj0_count_[node.id];
        const NodeId* p = I.adj0_ptr(node.id);
        nd.neighbors.resize(std::max(1, node.layer + 1));
        nd.neighbors[0].assign(p, p + cnt);
        for (int l = 1; l <= node.layer; ++l)
            nd.neighbors[l] = (l < static_cast<int>(node.neighbors.size()))
                               ? node.neighbors[l] : std::vector<NodeId>{};
        out.push_back(std::move(nd));
    }
    return out;
}

NodeId HnswIndex::entry_point_id() const {
    return impl_->entry_point_.load(std::memory_order_relaxed);
}
int HnswIndex::max_layer_val() const {
    return impl_->max_layer_.load(std::memory_order_relaxed);
}

void HnswIndex::restore(NodeId entry_point, int max_layer, size_t live_count,
                        std::vector<NodeData> nodes) {
    auto& I = *impl_;
    if (!I.nodes_flat_.empty())
        throw std::logic_error("HnswIndex::restore called on a non-empty index");
    I.entry_point_.store(entry_point, std::memory_order_relaxed);
    I.max_layer_.store(max_layer,     std::memory_order_relaxed);
    I.live_count_.store(live_count,   std::memory_order_relaxed);
    for (auto& nd : nodes) {
        I.store_vec(nd.id, nd.vec.data());
        I.grow_nodes(nd.id);
        I.grow_adj0(nd.id);
        HnswNode& node  = I.nodes_flat_[nd.id];
        node.id         = nd.id;
        node.layer      = nd.layer;
        node.tombstone  = nd.tombstone;
        if (!nd.neighbors.empty()) {
            const auto& l0 = nd.neighbors[0];
            uint8_t cnt = static_cast<uint8_t>(std::min((int)l0.size(), I.cfg.M0));
            I.adj0_count_[nd.id] = cnt;
            NodeId* p = I.adj0_ptr(nd.id);
            for (int k = 0; k < cnt; ++k) p[k] = l0[k];
        }
        node.neighbors = std::move(nd.neighbors);
        NodeId cur = I.next_id_.load(std::memory_order_relaxed);
        while (nd.id + 1 > cur) {
            if (I.next_id_.compare_exchange_weak(cur, nd.id + 1,
                    std::memory_order_relaxed, std::memory_order_relaxed))
                break;
        }
    }
}

}  // namespace vectordb
