// -----------------------------------------------------------------------------
// metadata_index.cpp — In-memory metadata index  (Day 16)
// -----------------------------------------------------------------------------
#include "metadata_index.h"
#include <algorithm>
#include <unordered_map>
#include <vector>

namespace vectordb {

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct MetadataIndex::Impl {
    // string_idx[field][value] → unordered list of node IDs
    std::unordered_map<std::string,
        std::unordered_map<std::string, std::vector<NodeId>>> string_idx;

    // numeric_idx[field] → sorted vector of (value, NodeId), sorted by value
    std::unordered_map<std::string,
        std::vector<std::pair<double, NodeId>>> numeric_idx;

    // Reverse indices: used by remove() to avoid scanning all fields.
    // string_rev[id]  → list of (field, value) pairs this id appears in
    // numeric_rev[id] → list of (field, value) pairs this id appears in
    std::unordered_map<NodeId,
        std::vector<std::pair<std::string, std::string>>> string_rev;
    std::unordered_map<NodeId,
        std::vector<std::pair<std::string, double>>> numeric_rev;
};

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

MetadataIndex::MetadataIndex()  : impl_(std::make_unique<Impl>()) {}
MetadataIndex::~MetadataIndex() = default;

// ---------------------------------------------------------------------------
// insert_string
// ---------------------------------------------------------------------------

void MetadataIndex::insert_string(const std::string& field,
                                  const std::string& value,
                                  NodeId id) {
    impl_->string_idx[field][value].push_back(id);
    impl_->string_rev[id].emplace_back(field, value);
}

// ---------------------------------------------------------------------------
// insert_numeric
// ---------------------------------------------------------------------------

void MetadataIndex::insert_numeric(const std::string& field,
                                   double value,
                                   NodeId id) {
    auto& vec = impl_->numeric_idx[field];
    // Maintain sorted order (by value, then id).
    auto pos = std::lower_bound(vec.begin(), vec.end(), std::make_pair(value, id));
    vec.insert(pos, {value, id});
    impl_->numeric_rev[id].emplace_back(field, value);
}

// ---------------------------------------------------------------------------
// query_eq
// ---------------------------------------------------------------------------

std::vector<NodeId> MetadataIndex::query_eq(const std::string& field,
                                            const std::string& value) const {
    auto fit = impl_->string_idx.find(field);
    if (fit == impl_->string_idx.end()) return {};
    auto vit = fit->second.find(value);
    if (vit == fit->second.end()) return {};
    return vit->second;
}

// ---------------------------------------------------------------------------
// query_range
// ---------------------------------------------------------------------------

std::vector<NodeId> MetadataIndex::query_range(const std::string& field,
                                               double lo, double hi) const {
    if (lo > hi) return {};
    auto fit = impl_->numeric_idx.find(field);
    if (fit == impl_->numeric_idx.end()) return {};
    const auto& vec = fit->second;

    // lower_bound: first element with value >= lo
    auto lo_it = std::lower_bound(vec.begin(), vec.end(), lo,
        [](const std::pair<double, NodeId>& p, double v) { return p.first < v; });

    // upper_bound: first element with value > hi
    auto hi_it = std::upper_bound(vec.begin(), vec.end(), hi,
        [](double v, const std::pair<double, NodeId>& p) { return v < p.first; });

    std::vector<NodeId> result;
    result.reserve(static_cast<size_t>(hi_it - lo_it));
    for (auto it = lo_it; it != hi_it; ++it)
        result.push_back(it->second);
    return result;
}

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------

void MetadataIndex::remove(NodeId id) {
    // Remove from all string index entries this id appears in.
    auto sit = impl_->string_rev.find(id);
    if (sit != impl_->string_rev.end()) {
        for (auto& [field, value] : sit->second) {
            auto fit = impl_->string_idx.find(field);
            if (fit == impl_->string_idx.end()) continue;
            auto vit = fit->second.find(value);
            if (vit == fit->second.end()) continue;
            auto& ids = vit->second;
            ids.erase(std::remove(ids.begin(), ids.end(), id), ids.end());
        }
        impl_->string_rev.erase(sit);
    }

    // Remove from all numeric index entries this id appears in.
    auto nit = impl_->numeric_rev.find(id);
    if (nit != impl_->numeric_rev.end()) {
        for (auto& [field, value] : nit->second) {
            auto fit = impl_->numeric_idx.find(field);
            if (fit == impl_->numeric_idx.end()) continue;
            auto& vec = fit->second;
            // Binary search to the first entry with this value, then scan for id.
            auto lo = std::lower_bound(vec.begin(), vec.end(), value,
                [](const std::pair<double, NodeId>& p, double v) { return p.first < v; });
            for (auto it = lo; it != vec.end() && it->first == value; ++it) {
                if (it->second == id) {
                    vec.erase(it);
                    break;
                }
            }
        }
        impl_->numeric_rev.erase(nit);
    }
}

}  // namespace vectordb
