// -----------------------------------------------------------------------------
// hnsw_index.h — Hierarchical Navigable Small World (HNSW) index
//
// Approximate nearest-neighbor index based on the HNSW algorithm
// (Malkov & Yashunin 2018). Supports insert, k-NN search, and soft-delete.
//
// HnswConfig:
//   dim              — vector dimensionality (must match VectorFile)
//   metric           — distance function (L2 / Cosine / InnerProduct)
//   M                — max neighbors per node on layers 1+ (typ. 16)
//   M0               — max neighbors at layer 0 (typ. 2*M = 32)
//   ef_construction  — beam width during insert (higher → better recall,
//                      slower build; typ. 200)
//
// API:
//   HnswIndex(cfg)
//       Constructs an empty index with the given config.
//
//   insert(id, vec)
//       Inserts vector `vec` under key `id`. Assigns a random layer,
//       runs greedy descent from the top layer, then beam-searches
//       ef_construction neighbors at the assigned layer and below.
//       Creates bidirectional edges with heuristic pruning.
//
//   search(query, k, ef_search) → vector<(distance, id)>
//       Returns up to k nearest neighbors sorted by ascending distance.
//       ef_search controls the beam width (higher → better recall, slower).
//
//   remove(id)
//       Soft-delete: marks the node as tombstone. Does not relink neighbors.
//       Tombstoned nodes are excluded from search results but may still
//       be traversed as graph intermediaries.
//
//   size() → size_t
//       Count of live (non-tombstoned) nodes.
// -----------------------------------------------------------------------------
#pragma once
#include "hnsw_node.h"
#include "distance.h"
#include <memory>
#include <utility>
#include <vector>

namespace vectordb {

struct HnswConfig {
    size_t dim = 128;
    Metric metric = Metric::L2;
    int    M = 16;              // max neighbors per node (layers 1+)
    int    M0 = 32;             // max neighbors at layer 0 (= 2*M)
    int    ef_construction = 200;
};

// HNSW approximate nearest-neighbor index.
// insert / search / remove (tombstone) — see hnsw_index.cpp for implementation.
class HnswIndex {
public:
    explicit HnswIndex(HnswConfig cfg);
    ~HnswIndex();

    void insert(NodeId id, const float* vec);

    // Returns up to k (distance, id) pairs sorted by ascending distance.
    std::vector<std::pair<float, NodeId>> search(
        const float* query, int k, int ef_search) const;

    // Soft-delete: marks node as tombstone. Does not relink neighbors.
    void remove(NodeId id);

    size_t size() const;  // live (non-tombstoned) node count

    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
