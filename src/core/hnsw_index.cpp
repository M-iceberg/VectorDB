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
#include <cassert>
#include <cmath>
#include <cstring>
#include <queue>
#include <random>
#include <stdexcept>

namespace vectordb {

// ---------------------------------------------------------------------------
// Impl — private state
// ---------------------------------------------------------------------------

struct HnswIndex::Impl {
    HnswConfig cfg;
    std::unique_ptr<DistanceCompute> dc;

    // Graph structure: flat array indexed by NodeId.
    // nodes_flat_[id] is valid when node.id != kInvalidNode (default-constructed nodes are invalid).
    // Direct array access eliminates the hash lookup in the search_layer hot path.
    std::vector<HnswNode> nodes_flat_;

    void grow_nodes(NodeId id) {
        if (id >= nodes_flat_.size())
            nodes_flat_.resize(id + 1);  // default-constructs with id = kInvalidNode
    }
    bool node_exists(NodeId id) const {
        return id < nodes_flat_.size() && nodes_flat_[id].id != kInvalidNode;
    }

    // Unified per-node storage: each node occupies one contiguous block of
    // (cfg.M0 + cfg.dim) uint32_t elements, laid out as:
    //   [adj0 neighbors: cfg.M0 × NodeId] [vector data: cfg.dim × float]
    //
    // Collocating adj0 and vec in one array means a single prefetch of node n's
    // block warms up both its neighbor list (needed when n becomes a candidate)
    // and its vector (needed for dist_q). With separate adj0_ and vecs_flat_,
    // accessing neighbor n required two separate random-access trips into two
    // different memory regions; the hardware prefetcher could not predict either.
    //
    // node_stride_ = cfg.M0 + cfg.dim  (in uint32_t units; sizeof(NodeId)==sizeof(float)==4)
    // adj0_ptr(id) = &node_blocks_[id * node_stride_]              → NodeId[cfg.M0]
    // vec_ptr(id)  = (float*)&node_blocks_[id * node_stride_ + M0] → float[cfg.dim]
    //
    // adj0_count_[id] = number of valid layer-0 neighbors for node id (≤ cfg.M0).
    // Upper-layer neighbors (layer ≥ 1) remain in nodes_flat_[id].neighbors[layer].
    size_t                node_stride_ = 0;
    std::vector<uint32_t> node_blocks_;
    std::vector<uint8_t>  adj0_count_;

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

    // Vector data is stored at offset cfg.M0 within each node's block.
    // sizeof(NodeId) == sizeof(float) == sizeof(uint32_t) == 4, so the reinterpret_cast is safe.

    NodeId entry_point  = kInvalidNode;
    int    max_layer    = -1;
    size_t live_count   = 0;
    NodeId max_node_id_ = 0;  // largest NodeId ever inserted; sizes visited array in search_layer
    NodeId next_id_     = 0;  // next auto-assigned NodeId for insert()

    float*       vec_ptr(NodeId id)       { return reinterpret_cast<float*>(node_blocks_.data() + (size_t)id * node_stride_ + cfg.M0); }
    const float* vec_ptr(NodeId id) const { return reinterpret_cast<const float*>(node_blocks_.data() + (size_t)id * node_stride_ + cfg.M0); }

    void store_vec(NodeId id, const float* vec) {
        grow_node_block(id);
        std::memcpy(vec_ptr(id), vec, cfg.dim * sizeof(float));
    }

    std::mt19937 rng{42};  // Mersenne Twister random number generator for layer assignment; fixed seed for reproducibility
    double ml;  // 1 / ln(M) — controls layer assignment probability; larger M → smaller ml → fewer high-layer nodes

    // constructor ：Initializes config, selects the distance compute implementation for the given metric,
    // and precomputes ml. All other fields (nodes, vecs, entry_point, etc.) use their inline defaults.
    explicit Impl(HnswConfig c)
        : cfg(c)
        , dc(DistanceCompute::create(c.metric))
        , node_stride_(static_cast<size_t>(c.M0) + c.dim)
        , ml(1.0 / std::log(static_cast<double>(c.M)))
    {}

    // Distance between two stored nodes.
    float dist(NodeId a, NodeId b) const {
        return dc->compute(vec_ptr(a), vec_ptr(b), cfg.dim);
    }

    // Distance from an external query vector to a stored node.
    float dist_q(const float* query, NodeId b) const {
        return dc->compute(query, vec_ptr(b), cfg.dim);
    }

    // Sample a random layer using the exponential distribution from the paper.
    // ml = 1/ln(M) ensures the expected number of nodes at layer l is M^(-l).
    int assign_layer() {
        std::uniform_real_distribution<double> u(0.0, 1.0);
        return static_cast<int>(-std::log(u(rng)) * ml);
    }

    // Beam search at a single layer starting from entry point `ep`.
    // query: pointer to the first element of the query float array (length = cfg.dim).
    //        float* is used instead of vector<float> so callers can pass data from
    //        any source (array, vector, mmap) without copying.
    // returns: up to `ef` (distance, node_id) pairs sorted ascending by distance.
    //          callers use this differently: insert Step 1 takes only [0] as the next
    //          entry point; insert Step 2 picks M neighbors to connect; search filters
    //          tombstones and returns the top-k to the user.
    std::vector<std::pair<float, NodeId>> search_layer(
        const float* query, NodeId ep, int ef, int layer) const
    {
        using DistId = std::pair<float, NodeId>;  // alias for (distance, node_id) pair
        // W: result set — max-heap so we can evict the worst easily.
        auto cmp_max = [](const DistId& a, const DistId& b) { return a.first < b.first; };
        // C: candidate set — min-heap so we always explore the closest next.
        auto cmp_min = [](const DistId& a, const DistId& b) { return a.first > b.first; };

        // priority_queue<element type, underlying container, comparator>
        // vector<DistId> is the underlying storage; must be explicit here because a custom comparator is provided.
        std::priority_queue<DistId, std::vector<DistId>, decltype(cmp_max)> W(cmp_max);
        std::priority_queue<DistId, std::vector<DistId>, decltype(cmp_min)> C(cmp_min);
        // Flat visited array: O(1) lookup/insert, one allocation vs unordered_set's many.
        // max_node_id_ + 1 bytes; fits in L2 cache for N <= ~100K (12.5 KB).
        std::vector<uint8_t> visited(max_node_id_ + 1, 0);

        float ep_dist = dist_q(query, ep);
        W.push({ep_dist, ep});
        C.push({ep_dist, ep});
        visited[ep] = 1;

        while (!C.empty()) {
            // Always expand the closest unexplored candidate first (C is a min-heap).
            auto [candidate_dist, candidate] = C.top(); C.pop();

            // Termination: if the closest candidate in C is already farther than the
            // worst result in W, no future expansion can improve W — stop early.
            if (candidate_dist > W.top().first) break;

            // Expand: visit each neighbor at this layer.
            // Layer 0 reads from the flat CSR adj0_ array (one multiply-add, no pointer chase).
            // Upper layers fall back to the per-node neighbor vector.
            auto expand = [&](NodeId neighbor) {
                if (neighbor == kInvalidNode) return;
                if (visited[neighbor]) return;
                visited[neighbor] = 1;

                float neighbor_dist = dist_q(query, neighbor);
                float worst_dist    = W.top().first;
                if (neighbor_dist < worst_dist || static_cast<int>(W.size()) < ef) {
                    C.push({neighbor_dist, neighbor});
                    W.push({neighbor_dist, neighbor});
                    if (static_cast<int>(W.size()) > ef) W.pop();
                }
            };

            if (layer == 0) {
                int cnt = adj0_count_[candidate];
                const NodeId* nbrs = adj0_ptr(candidate);
                // Issue prefetches for all neighbor vectors before computing distances.
                // While dist_q(nbrs[0]) runs, nbrs[1..cnt-1] vectors arrive from memory.
#ifndef VORTEXDB_NO_PREFETCH
                for (int k = 0; k < cnt; ++k)
                    __builtin_prefetch(vec_ptr(nbrs[k]), 0, 0);
#endif
                for (int k = 0; k < cnt; ++k) expand(nbrs[k]);
            } else {
                const auto& candidate_nbrs = nodes_flat_[candidate].neighbors;
                if (layer >= static_cast<int>(candidate_nbrs.size())) continue;
                for (NodeId neighbor : candidate_nbrs[layer]) expand(neighbor);
            }
        }

        // Drain W into a vector sorted ascending by distance.
        std::vector<DistId> result;
        result.reserve(W.size());
        while (!W.empty()) { result.push_back(W.top()); W.pop(); }
        std::sort(result.begin(), result.end());
        return result;
    }

    // Algorithm 4 (Malkov & Yashunin 2018): heuristic neighbor selection.
    //
    // Simple greedy selection (take the M_max closest) tends to cluster all neighbors
    // in the same direction — if there are 50 vectors in one dense region and 5 spread
    // around, greedy fills the list with the dense cluster and misses the spread-out ones.
    // This hurts recall: searches coming from different directions can't navigate through.
    //
    // The heuristic adds a diversity check: a candidate e is admitted only if it is
    // closer to q than to any already-selected neighbor r. If dist(e, r) < dist(e, q)
    // for some r, then r already "covers" the direction of e — adding e would be
    // redundant and waste a slot. This forces selected neighbors to span different
    // directions in vector space, improving graph connectivity and recall.
    //
    // candidates must be sorted ascending by distance to q.
    std::vector<NodeId> select_neighbors(
        const std::vector<std::pair<float, NodeId>>& candidates,
        int M_max) const
    {
        std::vector<NodeId> result;
        result.reserve(M_max);

        for (auto& [d_q_e, e] : candidates) {
            if (static_cast<int>(result.size()) >= M_max) break;

            if (!cfg.heuristic) {
                // Greedy: take the M_max closest, no diversity check.
                result.push_back(e);
                continue;
            }

            // Diversity check: admit e only if no already-selected neighbor r is
            // closer to e than q is. If r is closer to e than q is, r already
            // covers e's direction — e would be a redundant neighbor.
            bool dominated = false;
            for (NodeId r : result) {
                if (dist(e, r) < d_q_e) {
                    dominated = true;
                    break;
                }
            }
            if (!dominated) result.push_back(e);
        }
        return result;
    }

    // Adds a bidirectional edge between u and v at `layer`: both u→v and v→u.
    // Bidirectional edges are required so the graph stays navigable from any direction.
    //
    // If a neighbor list is already full (>= M_max), heuristic pruning is applied:
    // build a candidate set from (existing neighbors ∪ new node), run select_neighbors
    // to pick the best M_max diverse subset, and replace the neighbor list.
    // This keeps each node's neighbor count within M_max while selecting diverse neighbors
    // rather than just the closest ones.
    void add_edge(NodeId u, NodeId v, int layer, int M_max) {
        if (layer == 0) {
            auto add_one_flat = [&](NodeId from, NodeId to) {
                grow_adj0(from);
                uint8_t& cnt = adj0_count_[from];
                NodeId*  nbrs = adj0_ptr(from);
                if (cnt < M_max) {
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
            };
            add_one_flat(u, v);
            add_one_flat(v, u);
        } else {
            auto add_one = [&](NodeId from, NodeId to) {
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
            };
            add_one(u, v);
            add_one(v, u);
        }
    }
};

// ---------------------------------------------------------------------------
// HnswIndex public API
// ---------------------------------------------------------------------------

// Constructs an empty index with the given config. All parameters (dim, metric, M, M0,
// ef_construction) are fixed at construction time and cannot be changed afterwards.
HnswIndex::HnswIndex(HnswConfig cfg) : impl_(std::make_unique<Impl>(cfg)) {}
HnswIndex::~HnswIndex() = default;

// Internal helper: inserts a vector with a specific NodeId.
// Used by both insert() (auto-id) and insert_for_recovery() (explicit id).
static void insert_with_id(HnswIndex::Impl& I, NodeId id, const float* vec) {

    // Store the vector in the flat array (grows if id > current max).
    I.store_vec(id, vec);

    // Determine whether this is a new node or a re-insert of an existing one.
    // live_count must only be incremented if the node is new or was tombstoned —
    // re-inserting a live node (e.g. same WAL record replayed twice) must not
    // double-count it.
    bool is_new         = !I.node_exists(id);
    bool was_tombstoned = !is_new && I.nodes_flat_[id].tombstone;

    // Create the node and assign a random layer.
    int assigned_layer = I.assign_layer();
    I.grow_nodes(id);
    I.grow_adj0(id);
    HnswNode& node  = I.nodes_flat_[id];
    node.id         = id;
    node.layer      = assigned_layer;
    node.tombstone  = false;
    node.neighbors.assign(assigned_layer + 1, {});
    if (is_new || was_tombstoned)
        ++I.live_count;

    // First node becomes the entry point.
    if (I.entry_point == kInvalidNode) {
        I.entry_point = id;
        I.max_layer   = assigned_layer;
        return;
    }

    NodeId entry_point = I.entry_point;

    // Phase 1: greedy descent from max_layer to assigned_layer+1 (ef=1, find entry point).
    for (int level = I.max_layer; level > assigned_layer; --level) {
        auto candidates = I.search_layer(vec, entry_point, 1, level);
        if (!candidates.empty()) entry_point = candidates[0].second;
    }

    // Phase 2: for each layer from assigned_layer down to 0, find the best neighbors
    // for the new node and connect bidirectional edges at that layer.
    // Layer 0 uses M0 (larger) as the neighbor cap; all other layers use M.
    // After each layer, update entry_point to the closest candidate found — this
    // gives a better starting point for the beam search on the next (lower) layer.
    for (int level = std::min(assigned_layer, I.max_layer); level >= 0; --level) {
        // layer 0 gets a larger neighbor cap (M0) since it's the main search layer.
        int max_neighbors = (level == 0) ? I.cfg.M0 : I.cfg.M;

        // beam search to find the ef_construction closest nodes to the new vector at this level.
        auto candidates = I.search_layer(vec, entry_point, I.cfg.ef_construction, level);

        // pick the best max_neighbors diverse candidates (heuristic pruning, Algorithm 4).
        auto neighbors = I.select_neighbors(candidates, max_neighbors);

        // ensure the node's neighbor list is sized for this level before connecting edges.
        auto& node_nbrs = I.nodes_flat_[id].neighbors;
        if (static_cast<int>(node_nbrs.size()) <= level)
            node_nbrs.resize(level + 1);

        // connect bidirectional edges between the new node and each selected neighbor.
        for (NodeId neighbor : neighbors) {
            I.add_edge(id, neighbor, level, max_neighbors);
        }

        // Update entry_point to the closest node found at this layer before descending.
        // A node assigned layer l has a separate neighbor list at each layer 0..l,
        // so the same node id is a valid entry point one layer down — just with a
        // different (layer-specific) set of neighbors. Starting closer to the new node
        // at each level reduces the beam search work at the next level.
        if (!candidates.empty()) entry_point = candidates[0].second;
    }

    // Update entry point if new node reaches a higher layer.
    if (assigned_layer > I.max_layer) {
        I.entry_point = id;
        I.max_layer   = assigned_layer;
    }
}

// Auto-assigns sequential NodeId using next_id_ counter.
// Returns the assigned NodeId.
NodeId HnswIndex::insert(const float* vec) {
    auto& I = *impl_;
    NodeId id = I.next_id_++;
    insert_with_id(I, id, vec);
    return id;
}

// Used for WAL replay and re-inserts with an existing NodeId.
// Updates next_id_ = max(next_id_, id+1) so future auto-assigns don't collide.
void HnswIndex::insert_for_recovery(NodeId id, const float* vec) {
    auto& I = *impl_;
    if (id + 1 > I.next_id_)
        I.next_id_ = id + 1;
    insert_with_id(I, id, vec);
}

// Returns the k approximate nearest neighbors of query, sorted ascending by distance.
//   query:     pointer to the float array of length cfg.dim.
//   k:         number of results to return.
//   ef_search: beam width at layer 0; higher = better recall, slower query.
//              should be >= k — beam search finds ef_search candidates before filtering
//              tombstones, so if ef_search == k and some are tombstoned, fewer than k
//              results will be returned. if ef_search < k, it is clamped to k internally.
// Tombstoned nodes are excluded from results. Returns fewer than k if the index
// has fewer than k live nodes.
std::vector<std::pair<float, NodeId>> HnswIndex::search(
    const float* query, int k, int ef_search) const
{
    auto& I = *impl_;
    if (I.entry_point == kInvalidNode) return {};

    NodeId entry_point = I.entry_point;

    // Step 1: greedy descent from max_layer to layer 1 (ef=1).
    // Quickly navigates to a node close to the query without exploring exhaustively.
    // Stops at layer 1 — layer 0 is handled separately with a full beam search.
    for (int level = I.max_layer; level > 0; --level) {
        auto candidates = I.search_layer(query, entry_point, 1, level);
        if (!candidates.empty()) entry_point = candidates[0].second;
    }

    // Step 2: full beam search at layer 0.
    // Layer 0 contains all nodes with the shortest, densest edges — this is where
    // the precise nearest neighbor search happens. beam_width = max(k, ef_search)
    // ensures enough candidates are found even after tombstones are filtered out.
    int beam_width = std::max(k, ef_search);
    auto candidates = I.search_layer(query, entry_point, beam_width, 0);

    // Step 3: filter tombstoned nodes and return the top-k results.
    // candidates is already sorted ascending by distance, so iterate in order.
    std::vector<std::pair<float, NodeId>> result;
    result.reserve(k);
    for (auto& [dist, id] : candidates) {
        if (static_cast<int>(result.size()) >= k) break;
        if (I.node_exists(id) && !I.nodes_flat_[id].tombstone)
            result.push_back({dist, id});
    }
    return result;
}

void HnswIndex::remove(NodeId id) {
    auto& I = *impl_;
    if (!I.node_exists(id) || I.nodes_flat_[id].tombstone) return;
    I.nodes_flat_[id].tombstone = true;
    --I.live_count;
}

size_t HnswIndex::size() const {
    return impl_->live_count;
}

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
        // Layer-0 neighbors live in adj0_, not in node.neighbors.
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

NodeId HnswIndex::entry_point_id() const { return impl_->entry_point; }
int    HnswIndex::max_layer_val()   const { return impl_->max_layer; }

void HnswIndex::restore(NodeId entry_point, int max_layer, size_t live_count,
                        std::vector<NodeData> nodes) {
    auto& I = *impl_;
    if (!I.nodes_flat_.empty())
        throw std::logic_error("HnswIndex::restore called on a non-empty index");
    I.entry_point = entry_point;
    I.max_layer   = max_layer;
    I.live_count  = live_count;
    for (auto& nd : nodes) {
        I.store_vec(nd.id, nd.vec.data());
        I.grow_nodes(nd.id);
        I.grow_adj0(nd.id);
        HnswNode& node  = I.nodes_flat_[nd.id];
        node.id         = nd.id;
        node.layer      = nd.layer;
        node.tombstone  = nd.tombstone;
        // Layer-0 neighbors go into adj0_; upper layers stay in node.neighbors.
        if (!nd.neighbors.empty()) {
            const auto& l0 = nd.neighbors[0];
            uint8_t cnt = static_cast<uint8_t>(std::min((int)l0.size(), I.cfg.M0));
            I.adj0_count_[nd.id] = cnt;
            NodeId* p = I.adj0_ptr(nd.id);
            for (int k = 0; k < cnt; ++k) p[k] = l0[k];
        }
        node.neighbors = std::move(nd.neighbors);  // keep upper layers; [0] ignored
        // Update next_id_ so auto-assigns after restore() don't collide.
        if (nd.id + 1 > I.next_id_)
            I.next_id_ = nd.id + 1;
    }
}

}  // namespace vectordb
