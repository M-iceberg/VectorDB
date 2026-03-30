#pragma once
#include "core/collection.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vectordb {

struct SearchRequest {
    std::string  collection;
    const float* query      = nullptr;
    int          top_k      = 10;
    int          ef_search  = 64;
};

struct SearchResult {
    float    distance;
    uint32_t id;
};

// Orchestrator: owns all collections and ties together HNSW index,
// VectorFile, WAL, and MetadataIndex per collection.
// All public methods are thread-safe (coarse-grained shared_mutex — Day 19).
class Engine {
public:
    explicit Engine(const std::string& data_dir);
    ~Engine();

    void create_collection(CollectionSchema schema);
    void drop_collection  (const std::string& name);

    void insert(const std::string& collection, uint32_t id, const float* vec);
    void remove(const std::string& collection, uint32_t id);

    std::vector<SearchResult> search(const SearchRequest& req) const;

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
