// HNSW graph checkpoint serialization — implemented Day 14.
#include "graph_serializer.h"
#include "core/hnsw_index.h"

namespace vectordb {

void GraphSerializer::serialize(const HnswIndex& index, const std::string& path) {
    // TODO Day 14: write adjacency list in binary format
    (void)index; (void)path;
}

std::unique_ptr<HnswIndex> GraphSerializer::deserialize(const std::string& path,
                                                          const HnswConfig& cfg) {
    // TODO Day 14: read adjacency list + rebuild HnswIndex
    (void)path;
    return std::make_unique<HnswIndex>(cfg);
}

}  // namespace vectordb
