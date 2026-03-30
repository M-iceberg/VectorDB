#pragma once
#include <cstdint>
#include <vector>

namespace vectordb {

using NodeId = uint32_t;
static constexpr NodeId kInvalidNode = ~NodeId{0};

struct HnswNode {
    NodeId id = kInvalidNode;
    int    layer = 0;                            // highest layer this node appears in
    std::vector<std::vector<NodeId>> neighbors;  // neighbors[level][i]
    bool   tombstone = false;                    // soft-deleted; skipped in search results
};

}  // namespace vectordb
