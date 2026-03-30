// HNSW insert / search / delete — implemented Week 2 (Days 6–10).
#include "hnsw_index.h"

namespace vectordb {

struct HnswIndex::Impl {
    HnswConfig cfg;
};

HnswIndex::HnswIndex(HnswConfig cfg)
    : impl_(std::make_unique<Impl>()) {
    impl_->cfg = cfg;
}

HnswIndex::~HnswIndex() = default;

void HnswIndex::insert(NodeId id, const float* vec) {
    // TODO Day 7: greedy descent + beam search + bidirectional edge creation
    (void)id; (void)vec;
}

std::vector<std::pair<float, NodeId>> HnswIndex::search(
    const float* query, int k, int ef_search) const {
    // TODO Day 9: two-phase search (top-layer greedy descent + layer-0 beam search)
    (void)query; (void)k; (void)ef_search;
    return {};
}

void HnswIndex::remove(NodeId id) {
    // TODO Day 10: tombstone; does not relink neighbors
    (void)id;
}

size_t HnswIndex::size() const {
    return 0;
}

}  // namespace vectordb
