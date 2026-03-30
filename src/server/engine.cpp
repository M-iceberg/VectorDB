// DB engine orchestrator — implemented Day 18.
#include "engine.h"

namespace vectordb {

struct Engine::Impl {
    std::string data_dir;
    // TODO Day 18: map<name, CollectionState> where CollectionState holds
    //             HnswIndex + VectorFile + Wal + MetadataIndex
};

Engine::Engine(const std::string& data_dir) : impl_(std::make_unique<Impl>()) {
    impl_->data_dir = data_dir;
}

Engine::~Engine() = default;

void Engine::create_collection(CollectionSchema schema) {
    // TODO Day 18
    (void)schema;
}

void Engine::drop_collection(const std::string& name) {
    // TODO Day 18
    (void)name;
}

void Engine::insert(const std::string& collection, uint32_t id, const float* vec) {
    // TODO Day 18: WAL append → HNSW insert → VectorFile append
    (void)collection; (void)id; (void)vec;
}

void Engine::remove(const std::string& collection, uint32_t id) {
    // TODO Day 18: WAL append → HNSW tombstone → MetadataIndex remove
    (void)collection; (void)id;
}

std::vector<SearchResult> Engine::search(const SearchRequest& req) const {
    // TODO Day 18: HnswIndex search → metadata post-filter
    (void)req;
    return {};
}

}  // namespace vectordb
