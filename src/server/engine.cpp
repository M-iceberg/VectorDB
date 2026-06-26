// -----------------------------------------------------------------------------
// engine.cpp — DB engine orchestrator  (Day 15)
//
// Per-collection on-disk layout under data_dir/{name}/:
//   schema.bin   — dim(8B) + metric(4B) + pad(4B) = 16 bytes
//   wal.log      — append-only WAL (+ wal.log.base sidecar)
//   vectors.vdb  — mmap-backed raw vector storage
//   graph.bin    — latest graph checkpoint (may not exist before first checkpoint)
// -----------------------------------------------------------------------------
#include "engine.h"
#include "core/hnsw_index.h"
#include "storage/graph_serializer.h"
#include "storage/vector_file.h"
#include "storage/wal.h"
#include "storage/wal_record.h"
#include <fcntl.h>
#include <filesystem>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unistd.h>

namespace vectordb {
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Per-collection runtime state
// ---------------------------------------------------------------------------

struct CollectionState {
    CollectionSchema            schema;
    std::unique_ptr<Wal>        wal;
    std::unique_ptr<VectorFile> vf;
    std::unique_ptr<HnswIndex>  index;
};

// ---------------------------------------------------------------------------
// Schema: tiny binary file so collections survive restart
// ---------------------------------------------------------------------------

struct SchemaRecord {
    uint64_t dim;
    uint32_t metric;
    uint32_t _pad;
};
static_assert(sizeof(SchemaRecord) == 16, "SchemaRecord must be 16 bytes");

// ---------------------------------------------------------------------------
// Impl
// ---------------------------------------------------------------------------

struct Engine::Impl {
    std::string                                      data_dir;
    mutable std::shared_mutex                        mu;
    std::unordered_map<std::string, CollectionState> cols;

    std::string col_dir(const std::string& name) const {
        return data_dir + "/" + name;
    }

    static void write_schema(const std::string& path, const CollectionSchema& s) {
        SchemaRecord rec{};
        rec.dim    = static_cast<uint64_t>(s.dim);
        rec.metric = static_cast<uint32_t>(s.metric);
        int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0)
            throw std::runtime_error("Engine: cannot write schema: " + path);
        if (::write(fd, &rec, sizeof(rec)) != sizeof(rec)) {
            ::close(fd);
            throw std::runtime_error("Engine: schema write failed");
        }
        ::close(fd);
    }

    static CollectionSchema read_schema(const std::string& path,
                                        const std::string& name) {
        SchemaRecord rec{};
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0)
            throw std::runtime_error("Engine: cannot read schema: " + path);
        if (::read(fd, &rec, sizeof(rec)) != sizeof(rec)) {
            ::close(fd);
            throw std::runtime_error("Engine: schema read failed");
        }
        ::close(fd);
        CollectionSchema s;
        s.name   = name;
        s.dim    = static_cast<size_t>(rec.dim);
        s.metric = static_cast<Metric>(rec.metric);
        return s;
    }

    // Open or recover a collection from disk. Called both by create_collection
    // (first open, no graph.bin yet) and by the constructor (recovery path).
    void open_collection(const CollectionSchema& schema) {
        const std::string& name = schema.name;
        std::string dir = col_dir(name);

        CollectionState state;
        state.schema = schema;
        state.wal   = std::make_unique<Wal>(dir + "/wal.log");
        state.vf    = std::make_unique<VectorFile>(dir + "/vectors.vdb", schema.dim);

        HnswConfig cfg;
        cfg.dim    = schema.dim;
        cfg.metric = schema.metric;

        std::string graph_path = dir + "/graph.bin";
        if (fs::exists(graph_path)) {
            state.index = GraphSerializer::deserialize(graph_path, cfg);
        } else {
            state.index = std::make_unique<HnswIndex>(cfg);
        }

        // Replay all WAL records in the file (= everything since last checkpoint).
        // No existence check needed: HnswIndex::insert() handles duplicate IDs
        // correctly (won't double-count live_count for re-inserts of live nodes,
        // and correctly un-tombstones nodes that were removed then re-inserted).
        auto& idx = *state.index;
        state.wal->replay(
            0,  // replay from beginning of current WAL file
            [&](uint32_t id, const float* vec, size_t /*dim*/) {
                idx.insert(id, vec);
            },
            [&](uint32_t id) {
                idx.remove(id);
            }
        );

        cols[name] = std::move(state);
    }
};

// ---------------------------------------------------------------------------
// Engine public API
// ---------------------------------------------------------------------------

Engine::Engine(const std::string& data_dir) : impl_(std::make_unique<Impl>()) {
    impl_->data_dir = data_dir;
    fs::create_directories(data_dir);

    // Discover and recover all persisted collections.
    for (auto& entry : fs::directory_iterator(data_dir)) {
        if (!entry.is_directory()) continue;
        std::string name        = entry.path().filename().string();
        std::string schema_path = entry.path().string() + "/schema.bin";
        if (!fs::exists(schema_path)) continue;
        CollectionSchema schema = Impl::read_schema(schema_path, name);
        impl_->open_collection(schema);
    }
}

Engine::~Engine() = default;

void Engine::create_collection(CollectionSchema schema) {
    if (schema.dim == 0)
        throw std::invalid_argument("Engine: dim must be > 0");

    std::unique_lock lock(impl_->mu);
    if (impl_->cols.count(schema.name))
        throw std::runtime_error("Engine: collection already exists: " + schema.name);

    std::string dir = impl_->col_dir(schema.name);
    fs::create_directories(dir);
    Impl::write_schema(dir + "/schema.bin", schema);
    impl_->open_collection(schema);
}

void Engine::drop_collection(const std::string& name) {
    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(name);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + name);
    impl_->cols.erase(it);
    fs::remove_all(impl_->col_dir(name));
}

void Engine::insert(const std::string& collection, uint32_t id, const float* vec) {
    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + collection);
    auto& state = it->second;

    auto payload = make_insert_payload(id, vec, state.schema.dim);
    state.wal->append(WalRecordType::Insert, payload.data(),
                      static_cast<uint32_t>(payload.size()));
    state.wal->sync();
    state.index->insert(id, vec);
    state.vf->append(vec);  // TODO: once HnswIndex::vecs is removed, VectorFile becomes
                            // the sole vector store and recovery must also replay into it.
}

void Engine::remove(const std::string& collection, uint32_t id) {
    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + collection);
    auto& state = it->second;

    auto payload = make_delete_payload(id);
    state.wal->append(WalRecordType::Delete, payload.data(),
                      static_cast<uint32_t>(payload.size()));
    state.wal->sync();
    state.index->remove(id);
}

std::vector<SearchResult> Engine::search(const SearchRequest& req) const {
    std::shared_lock lock(impl_->mu);
    auto it = impl_->cols.find(req.collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + req.collection);
    const auto& state = it->second;

    auto raw = state.index->search(req.query, req.top_k, req.ef_search);
    std::vector<SearchResult> results;
    results.reserve(raw.size());
    for (auto& [dist, id] : raw)
        results.push_back({dist, id});
    return results;
}

void Engine::checkpoint(const std::string& collection) {
    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + collection);
    auto& state = it->second;

    std::string graph_path = impl_->col_dir(collection) + "/graph.bin";
    GraphSerializer::serialize(*state.index, graph_path);

    // Truncate WAL up to the current end — records before this LSN are now
    // captured in graph.bin and no longer needed for recovery.
    Lsn ckpt_lsn = state.wal->current_lsn();
    state.wal->truncate_before(ckpt_lsn);
}

}  // namespace vectordb
