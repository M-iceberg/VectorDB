#pragma once
#include "core/hnsw_node.h"  // NodeId
#include <memory>
#include <string>
#include <vector>

namespace vectordb {

// In-memory inverted index for metadata-based filtered search.
//
// String fields: exact match via hash_map<value, vector<NodeId>>
// Numeric fields: range query via sorted vector<pair<double, NodeId>> + binary search
// Boolean fields: implemented as string fields with "0"/"1" values
//
// Post-filter approach: HNSW searches with inflated ef_search, results filtered here.
class MetadataIndex {
public:
    MetadataIndex();
    ~MetadataIndex();

    void insert_string (const std::string& field, const std::string& value, NodeId id);
    void insert_numeric(const std::string& field, double value,             NodeId id);

    std::vector<NodeId> query_eq   (const std::string& field, const std::string& value) const;
    std::vector<NodeId> query_range(const std::string& field, double lo, double hi)      const;

    // Removes `id` from all field indices.
    void remove(NodeId id);

    MetadataIndex(const MetadataIndex&) = delete;
    MetadataIndex& operator=(const MetadataIndex&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
