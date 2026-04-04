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

    std::unordered_map<NodeId, HnswNode>           nodes;  // graph structure: id → node (layer, neighbors)
    // Key-value store: id → raw float vector.
    // HnswNode only holds graph edges; the actual vector data lives here.
    // Given an id, vecs[id] returns the float array used for distance computation.
    // (Day 13: this map will be replaced by VectorFile — vectors stored on disk
    //  in aligned slabs, with id mapping to a file offset.)
    std::unordered_map<NodeId, std::vector<float>>  vecs;

    NodeId entry_point = kInvalidNode;
    int    max_layer   = -1;
    size_t live_count  = 0;

    std::mt19937 rng{42};
    double ml;  // 1 / ln(M) — controls layer assignment probability

    explicit Impl(HnswConfig c)
        : cfg(c)
        , dc(DistanceCompute::create(c.metric))
        , ml(1.0 / std::log(static_cast<double>(c.M)))
    {}

    // Distance between two stored nodes.
    float dist(NodeId a, NodeId b) const {
        return dc->compute(vecs.at(a).data(), vecs.at(b).data(), cfg.dim);
    }

    // Distance from a query pointer to a stored node.
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
    // Returns up to `ef` (distance, id) pairs sorted ascending by distance.
    std::vector<std::pair<float, NodeId>> search_layer(
        const float* query, NodeId ep, int ef, int layer) const
    {
        using DistId = std::pair<float, NodeId>;
        // W: result set — max-heap so we can evict the worst easily.
        auto cmp_max = [](const DistId& a, const DistId& b) { return a.first < b.first; };
        // C: candidate set — min-heap so we always explore the closest next.
        auto cmp_min = [](const DistId& a, const DistId& b) { return a.first > b.first; };

        std::priority_queue<DistId, std::vector<DistId>, decltype(cmp_max)> W(cmp_max);
        std::priority_queue<DistId, std::vector<DistId>, decltype(cmp_min)> C(cmp_min);
        std::unordered_set<NodeId> visited;

        float d_ep = dist_q(query, ep);
        W.push({d_ep, ep});
        C.push({d_ep, ep});
        visited.insert(ep);

        while (!C.empty()) {
            auto [d_c, c] = C.top(); C.pop();

            // If the closest unexplored candidate is already farther than the
            // worst result we have, further search cannot improve W.
            if (d_c > W.top().first) break;

            const auto& node_nbrs = nodes.at(c).neighbors;
            if (layer >= static_cast<int>(node_nbrs.size())) continue;

            for (NodeId e : node_nbrs[layer]) {
                if (e == kInvalidNode) continue;
                if (!visited.insert(e).second) continue;  // already seen

                float d_e = dist_q(query, e);
                float d_worst = W.top().first;

                if (d_e < d_worst || static_cast<int>(W.size()) < ef) {
                    C.push({d_e, e});
                    W.push({d_e, e});
                    if (static_cast<int>(W.size()) > ef) W.pop();  // evict worst
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

    // Select the M_max closest neighbors from a sorted candidate list.
    // Simple greedy selection — sufficient for correctness; the full diversity
    // heuristic from Algorithm 4 of the paper can be added later.
    std::vector<NodeId> select_neighbors(
        const std::vector<std::pair<float, NodeId>>& candidates,
        int M_max) const
    {
        std::vector<NodeId> result;
        result.reserve(std::min(M_max, static_cast<int>(candidates.size())));
        for (auto& [d, id] : candidates) {
            if (static_cast<int>(result.size()) >= M_max) break;
            result.push_back(id);
        }
        return result;
    }

    // Add a bidirectional edge between u and v at `layer`.
    // If either neighbor list exceeds M_max, evict the farthest neighbor.
    void add_edge(NodeId u, NodeId v, int layer, int M_max) {
        auto add_one = [&](NodeId from, NodeId to) {
            auto& nbrs = nodes[from].neighbors[layer];
            if (static_cast<int>(nbrs.size()) < M_max) {
                nbrs.push_back(to);
            } else {
                // Replace the farthest existing neighbor if `to` is closer.
                float d_to = dist(from, to);
                float worst_d = -1.0f;
                size_t worst_i = 0;
                for (size_t i = 0; i < nbrs.size(); ++i) {
                    float d = dist(from, nbrs[i]);
                    if (d > worst_d) { worst_d = d; worst_i = i; }
                }
                if (d_to < worst_d) nbrs[worst_i] = to;
            }
        };
        add_one(u, v);
        add_one(v, u);
    }
};

// ---------------------------------------------------------------------------
// HnswIndex public API
// ---------------------------------------------------------------------------

HnswIndex::HnswIndex(HnswConfig cfg) : impl_(std::make_unique<Impl>(cfg)) {}
HnswIndex::~HnswIndex() = default;

void HnswIndex::insert(NodeId id, const float* vec) {
    auto& I = *impl_;

    // Store the vector.
    I.vecs[id] = std::vector<float>(vec, vec + I.cfg.dim);

    // Create the node and assign a random layer.
    int l = I.assign_layer();
    HnswNode node;
    node.id       = id;
    node.layer    = l;
    node.neighbors.resize(l + 1);  // one neighbor list per layer 0..l
    I.nodes[id]   = std::move(node);
    ++I.live_count;

    // First node becomes the entry point.
    if (I.entry_point == kInvalidNode) {
        I.entry_point = id;
        I.max_layer   = l;
        return;
    }

    NodeId ep = I.entry_point;

    // Phase 1: greedy descent from max_layer to l+1 (ef=1, find entry point).
    for (int level = I.max_layer; level > l; --level) {
        auto W = I.search_layer(vec, ep, 1, level);
        if (!W.empty()) ep = W[0].second;
    }

    // Phase 2: beam search and edge insertion from min(l, max_layer) to 0.
    for (int level = std::min(l, I.max_layer); level >= 0; --level) {
        int M_max = (level == 0) ? I.cfg.M0 : I.cfg.M;
        auto W    = I.search_layer(vec, ep, I.cfg.ef_construction, level);
        auto nbrs = I.select_neighbors(W, M_max);

        // Ensure the node has a neighbor list at this level.
        auto& node_nbrs = I.nodes[id].neighbors;
        if (static_cast<int>(node_nbrs.size()) <= level)
            node_nbrs.resize(level + 1);

        for (NodeId n : nbrs) {
            I.add_edge(id, n, level, M_max);
        }

        if (!W.empty()) ep = W[0].second;
    }

    // Update entry point if new node reaches a higher layer.
    if (l > I.max_layer) {
        I.entry_point = id;
        I.max_layer   = l;
    }
}

std::vector<std::pair<float, NodeId>> HnswIndex::search(
    const float* query, int k, int ef_search) const
{
    auto& I = *impl_;
    if (I.entry_point == kInvalidNode) return {};

    NodeId ep = I.entry_point;

    // Greedy descent from max_layer to layer 1.
    for (int level = I.max_layer; level > 0; --level) {
        auto W = I.search_layer(query, ep, 1, level);
        if (!W.empty()) ep = W[0].second;
    }

    // Beam search at layer 0 with ef = max(k, ef_search).
    int ef = std::max(k, ef_search);
    auto W = I.search_layer(query, ep, ef, 0);

    // Filter tombstones and return at most k results.
    std::vector<std::pair<float, NodeId>> result;
    result.reserve(k);
    for (auto& [d, id] : W) {
        if (static_cast<int>(result.size()) >= k) break;
        auto it = I.nodes.find(id);
        if (it != I.nodes.end() && !it->second.tombstone) {
            result.push_back({d, id});
        }
    }
    return result;
}

void HnswIndex::remove(NodeId id) {
    auto& I = *impl_;
    auto it = I.nodes.find(id);
    if (it == I.nodes.end() || it->second.tombstone) return;
    it->second.tombstone = true;
    --I.live_count;
}

size_t HnswIndex::size() const {
    return impl_->live_count;
}

}  // namespace vectordb
