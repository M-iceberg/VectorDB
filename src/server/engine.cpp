// -----------------------------------------------------------------------------
// engine.cpp — DB engine orchestrator  (Day 15-17)
//
// Per-collection on-disk layout under data_dir/{name}/:
//   schema.bin    — dim(8B) + metric(4B) + pad(4B) = 16 bytes
//   wal.log       — append-only WAL (+ wal.log.base sidecar)
//   vectors.vdb   — mmap-backed raw vector storage
//   graph.bin     — latest graph checkpoint (may not exist before first checkpoint)
//   metadata.bin  — latest metadata index snapshot (may not exist before first checkpoint)
// -----------------------------------------------------------------------------
#include "engine.h"
#include "core/hnsw_index.h"
#include "storage/graph_serializer.h"
#include "storage/metadata_index.h"
#include "storage/metadata_serializer.h"
#include "storage/vector_file.h"
#include "storage/wal.h"
#include "storage/wal_record.h"
#include <algorithm>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <shared_mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <unistd.h>

namespace vectordb {
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Per-collection runtime state
// ---------------------------------------------------------------------------

struct CollectionState {
    CollectionSchema               schema;
    std::unique_ptr<Wal>           wal;
    std::unique_ptr<VectorFile>    vf;
    std::unique_ptr<HnswIndex>     index;
    std::unique_ptr<MetadataIndex> meta;
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

    // Open or recover a collection from disk.
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

        // Load graph snapshot if present.
        std::string graph_path = dir + "/graph.bin";
        if (fs::exists(graph_path)) {
            state.index = GraphSerializer::deserialize(graph_path, cfg);
        } else {
            state.index = std::make_unique<HnswIndex>(cfg);
        }

        // Load metadata snapshot if present.
        std::string meta_path = dir + "/metadata.bin";
        if (fs::exists(meta_path)) {
            state.meta = MetadataSerializer::deserialize(meta_path);
        } else {
            state.meta = std::make_unique<MetadataIndex>();
        }

        // Replay WAL records since the last checkpoint.
        // Passes vec_dim so replay() can locate the metadata section in each
        // Insert payload. Both HnswIndex and MetadataIndex are updated.
        auto& idx  = *state.index;
        auto& meta = *state.meta;
        state.wal->replay(
            0,
            schema.dim,
            [&](uint32_t id, const float* vec, size_t /*dim*/,
                const MetadataEntry& m) {
                idx.insert(id, vec);
                for (auto& [f, v] : m.strings)  meta.insert_string(f, v, id);
                for (auto& [f, v] : m.numerics) meta.insert_numeric(f, v, id);
            },
            [&](uint32_t id) {
                idx.remove(id);
                meta.remove(id);
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

void Engine::insert(const std::string& collection, uint32_t id,
                    const float* vec, const MetadataEntry& meta) {
    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + collection);
    auto& state = it->second;

    auto payload = make_insert_payload(id, vec, state.schema.dim, meta);
    state.wal->append(WalRecordType::Insert, payload.data(),
                      static_cast<uint32_t>(payload.size()));
    state.wal->sync();
    state.index->insert(id, vec);
    state.vf->append(vec);

    for (auto& [f, v] : meta.strings)  state.meta->insert_string(f, v, id);
    for (auto& [f, v] : meta.numerics) state.meta->insert_numeric(f, v, id);
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
    state.meta->remove(id);
}

std::vector<SearchResult> Engine::search(const SearchRequest& req) const {
    std::shared_lock lock(impl_->mu);
    auto it = impl_->cols.find(req.collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + req.collection);
    const auto& state = it->second;

    bool has_filter = !req.filters.empty();

    // Build allowlist from MetadataIndex when filters are present.
    std::unordered_set<NodeId> allowlist;
    if (has_filter) {
        bool first = true;
        for (auto& f : req.filters) {
            std::vector<NodeId> matches;
            if (f.op == FieldFilter::Op::Eq) {
                matches = state.meta->query_eq(f.field, f.str_val);
            } else {
                matches = state.meta->query_range(f.field, f.lo, f.hi);
            }

            if (first) {
                allowlist.insert(matches.begin(), matches.end());
                first = false;
            } else {
                // AND: keep only IDs present in both sets.
                std::unordered_set<NodeId> tmp;
                for (NodeId id : matches)
                    if (allowlist.count(id)) tmp.insert(id);
                allowlist = std::move(tmp);
            }
        }
    }

    // Inflate ef_search when filtering to compensate for candidates that
    // fail the filter. Use at least top_k * 5 or the requested ef_search.
    int ef = req.ef_search;
    if (has_filter)
        ef = std::max(ef, req.top_k * 5);

    int search_k = has_filter ? ef : req.top_k;
    auto raw = state.index->search(req.query, search_k, ef);

    std::vector<SearchResult> results;
    results.reserve(static_cast<size_t>(req.top_k));
    for (auto& [dist, id] : raw) {
        if (has_filter && !allowlist.count(id)) continue;
        results.push_back({dist, id});
        if (static_cast<int>(results.size()) == req.top_k) break;
    }
    return results;
}

void Engine::checkpoint(const std::string& collection) {
    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + collection);
    auto& state = it->second;

    std::string dir = impl_->col_dir(collection);
    // Serialize graph first, then metadata, then truncate WAL.
    // Graph + metadata must be on disk before WAL truncation so a crash
    // between the two steps always leaves a recoverable state.
    GraphSerializer::serialize(*state.index, dir + "/graph.bin");
    MetadataSerializer::serialize(*state.meta, dir + "/metadata.bin");

    Lsn ckpt_lsn = state.wal->current_lsn();
    state.wal->truncate_before(ckpt_lsn);
}

}  // namespace vectordb
