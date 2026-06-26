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
    // Idempotent: skip if (field, value, id) already present. Prevents
    // duplicate entries when WAL replay re-applies records that are already
    // captured in a metadata.bin snapshot (crash-during-checkpoint scenario).
    auto& rev = impl_->string_rev[id];
    for (auto& [f, v] : rev)
        if (f == field && v == value) return;

    impl_->string_idx[field][value].push_back(id);
    rev.emplace_back(field, value);
}

// ---------------------------------------------------------------------------
// insert_numeric
// ---------------------------------------------------------------------------

void MetadataIndex::insert_numeric(const std::string& field,
                                   double value,
                                   NodeId id) {
    auto& rev = impl_->numeric_rev[id];
    for (auto& [f, v] : rev)
        if (f == field && v == value) return;

    auto& vec = impl_->numeric_idx[field];
    // Maintain sorted order (by value, then id).
    auto pos = std::lower_bound(vec.begin(), vec.end(), std::make_pair(value, id));
    vec.insert(pos, {value, id});
    rev.emplace_back(field, value);
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

void MetadataIndex::for_each_string(
    std::function<void(const std::string&, const std::string&, NodeId)> cb) const {
    for (auto& [field, val_map] : impl_->string_idx)
        for (auto& [value, ids] : val_map)
            for (NodeId id : ids)
                cb(field, value, id);
}

void MetadataIndex::for_each_numeric(
    std::function<void(const std::string&, double, NodeId)> cb) const {
    for (auto& [field, pairs] : impl_->numeric_idx)
        for (auto& [value, id] : pairs)
            cb(field, value, id);
}

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
            if (ids.empty()) {
                fit->second.erase(vit);
                if (fit->second.empty())
                    impl_->string_idx.erase(fit);
            }
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
            if (vec.empty())
                impl_->numeric_idx.erase(fit);
        }
        impl_->numeric_rev.erase(nit);
    }
}

}  // namespace vectordb
