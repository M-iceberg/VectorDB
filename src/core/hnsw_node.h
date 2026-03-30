// -----------------------------------------------------------------------------
// hnsw_node.h — Per-node data structure for the HNSW graph
//
// HnswNode holds a node's identity, its highest layer, its neighbor lists
// for each layer, and a tombstone flag for soft-deletion.
//
// NodeId is a uint32_t slot index that doubles as the key into VectorFile
// for retrieving the raw float vector.
//
// Types:
//   NodeId                  — uint32_t alias; kInvalidNode = sentinel (~0u)
//   HnswNode::id            — this node's id
//   HnswNode::layer         — highest layer this node participates in
//   HnswNode::neighbors[l]  — neighbor ids at layer l (len ≤ M or M0)
//   HnswNode::tombstone     — if true, skipped in search results but graph
//                             edges are left intact (no relinking on delete)
// -----------------------------------------------------------------------------
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
