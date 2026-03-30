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
