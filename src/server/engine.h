// -----------------------------------------------------------------------------
// engine.h — DB engine orchestrator
//
// The Engine is the single entry point for all database operations. It owns
// one CollectionState per named collection, where each CollectionState
// bundles together:
//   HnswIndex      — the ANN graph
//   VectorFile     — mmap-backed raw vector storage
//   Wal            — write-ahead log for crash safety
//   MetadataIndex  — in-memory inverted/sorted index for filtered search
//
// Thread safety: all public methods are protected by a std::shared_mutex.
//   search() holds a shared (read) lock.
//   insert() / remove() / create_collection() / drop_collection() hold
//   an exclusive (write) lock.
//
// API:
//   Engine(data_dir)
//       Opens the engine. Runs crash recovery: loads the latest checkpoint
//       for each collection, then replays any WAL records beyond it.
//
//   create_collection(schema)
//       Creates a new collection with the given name, dim, and metric.
//       Throws if a collection with that name already exists.
//
//   drop_collection(name)
//       Destroys the collection and deletes its on-disk files.
//
//   insert(collection, id, vec)
//       Appends vec to VectorFile, writes an Insert WAL record, inserts
//       into HnswIndex, and updates MetadataIndex.
//
//   remove(collection, id)
//       Writes a Delete WAL record, tombstones the node in HnswIndex,
//       and removes it from MetadataIndex.
//
//   search(req) → vector<SearchResult>
//       Runs HNSW beam search with req.ef_search, applies any metadata
//       filter from req, and returns the top req.top_k results sorted by
//       ascending distance.
// -----------------------------------------------------------------------------
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

    // Writes a checkpoint: serializes the graph, then truncates the WAL up to
    // the current LSN. After this call, crash recovery restores state from the
    // checkpoint plus any WAL records appended after the checkpoint.
    void checkpoint(const std::string& collection);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
