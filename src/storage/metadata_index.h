// -----------------------------------------------------------------------------
// metadata_index.h — In-memory inverted index for metadata-based filtering
//
// Supports post-filter ANN search: HnswIndex searches with inflated ef_search,
// then results are filtered here against per-field predicates.
//
// String fields: hash_map<field, hash_map<value, vector<NodeId>>>
//   — supports exact-match (eq) and set membership (in) queries.
//
// Numeric fields: hash_map<field, sorted vector<(value, NodeId)>>
//   — supports range queries via lower_bound / upper_bound. O(log n + k).
//
// Boolean fields: stored as string fields with values "0" / "1".
//
// API:
//   insert_string(field, value, id)
//       Adds id to the inverted list for (field, value).
//
//   insert_numeric(field, value, id)
//       Inserts (value, id) into the sorted numeric list for field.
//
//   query_eq(field, value) → vector<NodeId>
//       Returns all ids where field == value (string equality).
//
//   query_range(field, lo, hi) → vector<NodeId>
//       Returns all ids where lo ≤ field ≤ hi (inclusive).
//
//   remove(id)
//       Removes id from every field index. Called on soft-delete.
//       O(total entries for id) — acceptable for delete-light workloads.
// -----------------------------------------------------------------------------
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
