// In-memory metadata inverted index — implemented Day 16.
#include "metadata_index.h"
#include <unordered_map>
#include <map>

namespace vectordb {

struct MetadataIndex::Impl {
    // string field → value → node list
    std::unordered_map<std::string,
        std::unordered_map<std::string, std::vector<NodeId>>> str_index;

    // numeric field → sorted (value, node) pairs
    std::unordered_map<std::string,
        std::vector<std::pair<double, NodeId>>> num_index;
};

MetadataIndex::MetadataIndex() : impl_(std::make_unique<Impl>()) {}
MetadataIndex::~MetadataIndex() = default;

void MetadataIndex::insert_string(const std::string& field,
                                   const std::string& value, NodeId id) {
    // TODO Day 16
    (void)field; (void)value; (void)id;
}

void MetadataIndex::insert_numeric(const std::string& field,
                                    double value, NodeId id) {
    // TODO Day 16
    (void)field; (void)value; (void)id;
}

std::vector<NodeId> MetadataIndex::query_eq(const std::string& field,
                                             const std::string& value) const {
    // TODO Day 16
    (void)field; (void)value;
    return {};
}

std::vector<NodeId> MetadataIndex::query_range(const std::string& field,
                                                double lo, double hi) const {
    // TODO Day 16: lower_bound / upper_bound on sorted num_index
    (void)field; (void)lo; (void)hi;
    return {};
}

void MetadataIndex::remove(NodeId id) {
    // TODO Day 16
    (void)id;
}

}  // namespace vectordb
