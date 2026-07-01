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
//   list_collections() → vector<CollectionSchema>
//       Returns the schema of every open collection.
//
//   insert(collection, id, vec, meta)
//       Appends vec to VectorFile, writes an Insert WAL record (with metadata),
//       inserts into HnswIndex, and updates MetadataIndex.
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

// A single metadata predicate. Multiple filters in SearchRequest are ANDed.
struct FieldFilter {
    std::string field;
    enum class Op { Eq, Range } op = Op::Eq;
    std::string str_val;        // used when op == Eq
    double lo = 0, hi = 0;     // used when op == Range (inclusive bounds)
};

struct SearchRequest {
    std::string  collection;
    const float* query          = nullptr;
    int          top_k          = 10;
    int          ef_search      = 64;
    std::vector<FieldFilter> filters;  // empty = no filter
};

struct SearchResult {
    float       distance;
    std::string user_id;
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
    std::vector<CollectionSchema> list_collections() const;

    // meta is optional: pass {} or omit for vectors without metadata.
    // user_id: caller-provided string ID, or empty string to auto-assign.
    // Returns the effective user_id (either the provided one or the auto-assigned NodeId as string).
    std::string insert(const std::string& collection, const std::string& user_id,
                       const float* vec, const MetadataEntry& meta = {});

    // Batch insert: appends all WAL records then calls fdatasync() once for the whole
    // batch instead of once per vector. vecs is row-major: vecs[i*dim..(i+1)*dim-1]
    // is the i-th vector. user_ids and metas may be empty (auto-assign / no metadata).
    // Returns effective user_ids in the same order as input.
    std::vector<std::string> insert_batch(
        const std::string& collection,
        const std::vector<std::string>& user_ids,
        const float* vecs,
        size_t count,
        const std::vector<MetadataEntry>& metas = {});

    void remove(const std::string& collection, const std::string& user_id);

    std::vector<SearchResult> search(const SearchRequest& req) const;

    // Writes a checkpoint: serializes the graph and metadata index, then
    // truncates the WAL up to the current LSN.
    void checkpoint(const std::string& collection);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
