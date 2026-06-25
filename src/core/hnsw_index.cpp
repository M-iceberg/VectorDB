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
#include <queue>
#include <random>
#include <unordered_map>
#include <unordered_set>

namespace vectordb {

// ---------------------------------------------------------------------------
// Impl — private state
// ---------------------------------------------------------------------------

struct HnswIndex::Impl {
    HnswConfig cfg;
    std::unique_ptr<DistanceCompute> dc;

    // Graph structure: id → node (layer assignment + per-layer neighbor lists + tombstone flag).
    // Used during graph traversal — finding neighbors never needs the raw float data.
    std::unordered_map<NodeId, HnswNode> nodes;

    // Vector data: id → raw float array.
    // Kept separate from nodes because the two are accessed in different patterns:
    // graph traversal only touches nodes (compact, cache-friendly), while distance
    // computation only touches vecs. Merging them would bloat the graph traversal
    // working set and hurt cache efficiency.
    // (Day 13: replaced by VectorFile — vectors stored on disk in aligned slabs.)
    std::unordered_map<NodeId, std::vector<float>> vecs;

    NodeId entry_point = kInvalidNode;
    int    max_layer   = -1;
    size_t live_count  = 0;

    std::mt19937 rng{42};  // Mersenne Twister random number generator for layer assignment; fixed seed for reproducibility
    double ml;  // 1 / ln(M) — controls layer assignment probability; larger M → smaller ml → fewer high-layer nodes

    // constructor ：Initializes config, selects the distance compute implementation for the given metric,
    // and precomputes ml. All other fields (nodes, vecs, entry_point, etc.) use their inline defaults.
    explicit Impl(HnswConfig c)
        : cfg(c)
        , dc(DistanceCompute::create(c.metric))
        , ml(1.0 / std::log(static_cast<double>(c.M)))
    {}

    // Distance between two stored nodes (both looked up from vecs).
    float dist(NodeId a, NodeId b) const {
        return dc->compute(vecs.at(a).data(), vecs.at(b).data(), cfg.dim);
    }

    // Distance from an external query vector to a stored node.
    // query is a raw pointer — the query vector is not in vecs (not yet inserted or never will be).
    float dist_q(const float* query, NodeId b) const {
        return dc->compute(query, vecs.at(b).data(), cfg.dim);
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
        std::unordered_set<NodeId> visited;

        float ep_dist = dist_q(query, ep);
        W.push({ep_dist, ep});
        C.push({ep_dist, ep});
        visited.insert(ep);

        while (!C.empty()) {
            // Always expand the closest unexplored candidate first (C is a min-heap).
            auto [candidate_dist, candidate] = C.top(); C.pop();

            // Termination: if the closest candidate in C is already farther than the
            // worst result in W, no future expansion can improve W — stop early.
            if (candidate_dist > W.top().first) break;

            const auto& candidate_nbrs = nodes.at(candidate).neighbors;
            if (layer >= static_cast<int>(candidate_nbrs.size())) continue;

            // Expand: visit each neighbor of this candidate at the current layer.
            for (NodeId neighbor : candidate_nbrs[layer]) {
                if (neighbor == kInvalidNode) continue;
                if (!visited.insert(neighbor).second) continue;  // skip already-seen nodes

                float neighbor_dist = dist_q(query, neighbor);
                float worst_dist    = W.top().first;

                // Add neighbor to both heaps if it's better than W's worst, or W isn't full yet.
                if (neighbor_dist < worst_dist || static_cast<int>(W.size()) < ef) {
                    C.push({neighbor_dist, neighbor});
                    W.push({neighbor_dist, neighbor});
                    if (static_cast<int>(W.size()) > ef) W.pop();  // W is full — evict the worst
                }
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
        auto add_one = [&](NodeId from, NodeId to) {
            auto& nbrs = nodes[from].neighbors[layer];
            if (static_cast<int>(nbrs.size()) < M_max) {
                nbrs.push_back(to);
            } else {
                // List is full — rebuild with heuristic pruning.
                // Include the new candidate alongside existing neighbors so it gets
                // a fair chance to displace a redundant neighbor.
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
};

// ---------------------------------------------------------------------------
// HnswIndex public API
// ---------------------------------------------------------------------------

// Constructs an empty index with the given config. All parameters (dim, metric, M, M0,
// ef_construction) are fixed at construction time and cannot be changed afterwards.
HnswIndex::HnswIndex(HnswConfig cfg) : impl_(std::make_unique<Impl>(cfg)) {}
HnswIndex::~HnswIndex() = default;

// Inserts a vector into the index.
//   id:  caller-assigned identifier for this vector; must be unique across all inserts.
//   vec: the vector to insert — a float array of length cfg.dim (e.g. a text embedding).
// Assigns a random layer, runs greedy descent to find a good entry point, then
// beam-searches at each layer to find and connect neighbors. O(log n) on average.
void HnswIndex::insert(NodeId id, const float* vec) {
    auto& I = *impl_;

    // Store the vector.
    I.vecs[id] = std::vector<float>(vec, vec + I.cfg.dim);

    // Create the node and assign a random layer.
    int assigned_layer = I.assign_layer();
    HnswNode node;
    node.id       = id;
    node.layer    = assigned_layer;
    node.neighbors.resize(assigned_layer + 1);  // one neighbor list per layer 0..assigned_layer
    I.nodes[id]   = std::move(node);
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
        auto& node_nbrs = I.nodes[id].neighbors;
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
        auto it = I.nodes.find(id);
        if (it != I.nodes.end() && !it->second.tombstone) {
            result.push_back({dist, id});
        }
    }
    return result;
}

// Soft-deletes a node by setting its tombstone flag.
//   id: the node to remove; no-op if id doesn't exist or is already tombstoned.
// The node stays in the graph as a routing intermediary — edges are not removed.
// Tombstoned nodes are skipped in search results but may still be traversed.
void HnswIndex::remove(NodeId id) {
    auto& I = *impl_;
    auto it = I.nodes.find(id);
    if (it == I.nodes.end() || it->second.tombstone) return;
    it->second.tombstone = true;
    --I.live_count;
}

// Returns the number of live (non-tombstoned) nodes in the index.
size_t HnswIndex::size() const {
    return impl_->live_count;
}

size_t HnswIndex::neighbor_count(NodeId id, int layer) const {
    auto& I = *impl_;
    auto it = I.nodes.find(id);
    if (it == I.nodes.end()) return 0;
    const auto& nbrs = it->second.neighbors;
    if (layer >= static_cast<int>(nbrs.size())) return 0;
    return nbrs[layer].size();
}

int HnswIndex::node_layer(NodeId id) const {
    auto& I = *impl_;
    auto it = I.nodes.find(id);
    if (it == I.nodes.end()) return -1;
    return it->second.layer;
}

std::vector<NodeId> HnswIndex::neighbors_of(NodeId id, int layer) const {
    auto& I = *impl_;
    auto it = I.nodes.find(id);
    if (it == I.nodes.end()) return {};
    const auto& nbrs = it->second.neighbors;
    if (layer >= static_cast<int>(nbrs.size())) return {};
    return nbrs[layer];
}

// ---------------------------------------------------------------------------
// Serialization support
// ---------------------------------------------------------------------------

std::vector<HnswIndex::NodeData> HnswIndex::snapshot() const {
    auto& I = *impl_;
    std::vector<NodeData> out;
    out.reserve(I.nodes.size());
    for (auto& [id, node] : I.nodes) {
        NodeData nd;
        nd.id        = node.id;
        nd.layer     = node.layer;
        nd.tombstone = node.tombstone;
        nd.vec       = I.vecs.at(id);
        nd.neighbors = node.neighbors;
        out.push_back(std::move(nd));
    }
    return out;
}

NodeId HnswIndex::entry_point_id() const { return impl_->entry_point; }
int    HnswIndex::max_layer_val()   const { return impl_->max_layer; }

void HnswIndex::restore(NodeId entry_point, int max_layer, size_t live_count,
                        std::vector<NodeData> nodes) {
    auto& I = *impl_;
    if (!I.nodes.empty())
        throw std::logic_error("HnswIndex::restore called on a non-empty index");
    I.entry_point = entry_point;
    I.max_layer   = max_layer;
    I.live_count  = live_count;
    for (auto& nd : nodes) {
        I.vecs[nd.id] = std::move(nd.vec);
        HnswNode node;
        node.id        = nd.id;
        node.layer     = nd.layer;
        node.tombstone = nd.tombstone;
        node.neighbors = std::move(nd.neighbors);
        I.nodes[nd.id] = std::move(node);
    }
}

}  // namespace vectordb
