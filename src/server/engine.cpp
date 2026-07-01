// -----------------------------------------------------------------------------
// engine.cpp — DB engine orchestrator  (Day 15-17)
//
// Per-collection on-disk layout under data_dir/{name}/:
//   schema.bin    — dim(8B) + metric(4B) + pad(4B) = 16 bytes
//   wal.log       — append-only WAL (+ wal.log.base sidecar)
//   vectors.vdb   — mmap-backed raw vector storage
//   graph.bin     — latest graph checkpoint (may not exist before first checkpoint)
//   metadata.bin  — latest metadata index snapshot (may not exist before first checkpoint)
//   id_map.bin    — binary id map: [count:4B][node_id:4B][uid_len:2B][uid:N]...
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
#include <cstring>
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
    std::unordered_map<std::string, NodeId> user_to_node;
    std::unordered_map<NodeId, std::string> node_to_user;
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

    // Load id_map.bin if present: [count:4B][node_id:4B][uid_len:2B][uid:N]...
    static void load_id_map(const std::string& path, CollectionState& state) {
        int fd = ::open(path.c_str(), O_RDONLY);
        if (fd < 0) return;  // file doesn't exist yet

        uint32_t count = 0;
        if (::read(fd, &count, sizeof(count)) != sizeof(count)) {
            ::close(fd); return;
        }
        for (uint32_t i = 0; i < count; ++i) {
            uint32_t node_id = 0;
            uint16_t uid_len = 0;
            if (::read(fd, &node_id, sizeof(node_id)) != sizeof(node_id)) break;
            if (::read(fd, &uid_len, sizeof(uid_len)) != sizeof(uid_len)) break;
            std::string uid(uid_len, '\0');
            if (::read(fd, uid.data(), uid_len) != uid_len) break;
            state.user_to_node[uid]      = node_id;
            state.node_to_user[node_id]  = uid;
        }
        ::close(fd);
    }

    // Write id_map.bin: [count:4B][node_id:4B][uid_len:2B][uid:N]...
    static void save_id_map(const std::string& path, const CollectionState& state) {
        std::string tmp = path + ".tmp";
        int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
        if (fd < 0)
            throw std::runtime_error("Engine: cannot write id_map: " + tmp);

        uint32_t count = static_cast<uint32_t>(state.user_to_node.size());
        ::write(fd, &count, sizeof(count));
        for (auto& [uid, node_id] : state.user_to_node) {
            uint16_t uid_len = static_cast<uint16_t>(uid.size());
            ::write(fd, &node_id, sizeof(node_id));
            ::write(fd, &uid_len, sizeof(uid_len));
            ::write(fd, uid.data(), uid_len);
        }
        if (::fsync(fd) != 0) {
            ::close(fd);
            throw std::runtime_error("Engine: fsync id_map failed");
        }
        ::close(fd);
        if (::rename(tmp.c_str(), path.c_str()) != 0)
            throw std::runtime_error("Engine: rename id_map failed");
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

        // Load id_map checkpoint if present (covers checkpoint-era entries).
        std::string idmap_path = dir + "/id_map.bin";
        load_id_map(idmap_path, state);

        // Replay WAL records since the last checkpoint.
        // Passes vec_dim so replay() can locate the metadata section in each
        // Insert payload. Both HnswIndex and MetadataIndex are updated.
        // WAL replay also rebuilds/updates the id maps from the user_id field.
        auto& idx  = *state.index;
        auto& meta = *state.meta;
        state.wal->replay(
            0,
            schema.dim,
            [&](uint32_t id, const std::string& user_id, const float* vec,
                size_t /*dim*/, const MetadataEntry& m) {
                idx.insert_for_recovery(id, vec);
                state.user_to_node[user_id] = id;
                state.node_to_user[id]      = user_id;
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

std::vector<CollectionSchema> Engine::list_collections() const {
    std::shared_lock lock(impl_->mu);
    std::vector<CollectionSchema> result;
    result.reserve(impl_->cols.size());
    for (auto& [name, state] : impl_->cols)
        result.push_back(state.schema);
    return result;
}

void Engine::drop_collection(const std::string& name) {
    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(name);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + name);
    impl_->cols.erase(it);
    fs::remove_all(impl_->col_dir(name));
}

std::string Engine::insert(const std::string& collection, const std::string& user_id,
                           const float* vec, const MetadataEntry& meta) {
    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + collection);
    auto& state = it->second;

    NodeId node_id;
    std::string effective_id;
    auto uid_it = state.user_to_node.find(user_id);
    if (!user_id.empty() && uid_it != state.user_to_node.end()) {
        // Re-insert of existing user_id (e.g. after remove): reuse the same NodeId.
        effective_id = user_id;
        node_id = uid_it->second;
        auto payload = make_insert_payload(node_id, effective_id, vec, state.schema.dim, meta);
        state.wal->append(WalRecordType::Insert, payload.data(),
                          static_cast<uint32_t>(payload.size()));
        state.wal->sync();
        state.index->insert_for_recovery(node_id, vec);
    } else {
        // New user_id (or empty → auto-assign): let HnswIndex pick the NodeId.
        node_id = state.index->insert(vec);
        effective_id = user_id.empty() ? std::to_string(node_id) : user_id;
        state.user_to_node[effective_id] = node_id;
        state.node_to_user[node_id] = effective_id;
        auto payload = make_insert_payload(node_id, effective_id, vec, state.schema.dim, meta);
        state.wal->append(WalRecordType::Insert, payload.data(),
                          static_cast<uint32_t>(payload.size()));
        state.wal->sync();
    }

    state.vf->append(vec);

    for (auto& [f, v] : meta.strings)  state.meta->insert_string(f, v, node_id);
    for (auto& [f, v] : meta.numerics) state.meta->insert_numeric(f, v, node_id);
    return effective_id;
}

std::vector<std::string> Engine::insert_batch(
    const std::string& collection,
    const std::vector<std::string>& user_ids,
    const float* vecs,
    size_t count,
    const std::vector<MetadataEntry>& metas)
{
    if (count == 0) return {};

    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + collection);
    auto& state = it->second;
    const size_t dim = state.schema.dim;

    std::vector<std::string>  assigned(count);
    std::vector<NodeId>       node_ids(count);

    // Phase 1: assign NodeIds, insert into HNSW, append WAL records (no sync yet).
    // VectorFile and MetadataIndex updates come after sync to keep the commit order
    // consistent with the single-insert path.
    for (size_t i = 0; i < count; ++i) {
        const float*       vec      = vecs + i * dim;
        const std::string& user_id  = user_ids.empty() ? "" : user_ids[i];
        const MetadataEntry& meta   = metas.empty()    ? MetadataEntry{} : metas[i];

        NodeId      node_id;
        std::string effective_id;

        auto uid_it = (!user_id.empty())
                      ? state.user_to_node.find(user_id)
                      : state.user_to_node.end();

        if (!user_id.empty() && uid_it != state.user_to_node.end()) {
            effective_id = user_id;
            node_id      = uid_it->second;
            state.index->insert_for_recovery(node_id, vec);
        } else {
            node_id      = state.index->insert(vec);
            effective_id = user_id.empty() ? std::to_string(node_id) : user_id;
            state.user_to_node[effective_id] = node_id;
            state.node_to_user[node_id]      = effective_id;
        }

        auto payload = make_insert_payload(node_id, effective_id, vec, dim, meta);
        state.wal->append(WalRecordType::Insert, payload.data(),
                          static_cast<uint32_t>(payload.size()));

        node_ids[i]  = node_id;
        assigned[i]  = std::move(effective_id);
    }

    // Phase 2: single fdatasync for the whole batch.
    state.wal->sync();

    // Phase 3: VectorFile + MetadataIndex (in-memory, no fsync needed).
    for (size_t i = 0; i < count; ++i) {
        const float*       vec  = vecs + i * dim;
        const MetadataEntry& meta = metas.empty() ? MetadataEntry{} : metas[i];
        state.vf->append(vec);
        for (auto& [f, v] : meta.strings)  state.meta->insert_string(f, v, node_ids[i]);
        for (auto& [f, v] : meta.numerics) state.meta->insert_numeric(f, v, node_ids[i]);
    }

    return assigned;
}

void Engine::remove(const std::string& collection, const std::string& user_id) {
    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + collection);
    auto& state = it->second;

    auto uid_it = state.user_to_node.find(user_id);
    if (uid_it == state.user_to_node.end())
        return;  // user_id not found — nothing to remove
    NodeId node_id = uid_it->second;

    auto payload = make_delete_payload(node_id);
    state.wal->append(WalRecordType::Delete, payload.data(),
                      static_cast<uint32_t>(payload.size()));
    state.wal->sync();
    state.index->remove(node_id);
    state.meta->remove(node_id);
    // Do NOT erase from id maps — keep mapping for re-insert consistency.
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
    for (auto& [dist, node_id] : raw) {
        if (has_filter && !allowlist.count(node_id)) continue;
        auto uid_it = state.node_to_user.find(node_id);
        std::string uid = (uid_it != state.node_to_user.end())
                          ? uid_it->second
                          : std::to_string(node_id);
        results.push_back({dist, std::move(uid)});
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
    // Serialize graph first, then metadata, then id_map, then truncate WAL.
    // All snapshots must be on disk before WAL truncation so a crash
    // between the two steps always leaves a recoverable state.
    GraphSerializer::serialize(*state.index, dir + "/graph.bin");
    MetadataSerializer::serialize(*state.meta, dir + "/metadata.bin");
    Impl::save_id_map(dir + "/id_map.bin", state);

    Lsn ckpt_lsn = state.wal->current_lsn();
    state.wal->truncate_before(ckpt_lsn);
}

}  // namespace vectordb
