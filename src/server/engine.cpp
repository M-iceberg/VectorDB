// -----------------------------------------------------------------------------
// engine.cpp — DB engine orchestrator  (Day 15-17)
//
// Per-collection on-disk layout under data_dir/{name}/:
//   schema.bin    — dim(8B) + metric(4B) + pad(4B) = 16 bytes
//   wal.log       — append-only WAL (+ wal.log.base sidecar)
//   vectors.vdb   — mmap-backed raw vector storage
//   checkpoint.current       — atomically-published snapshot manifest
//   checkpoint-{generation}/ — immutable graph.bin, metadata.bin, id_map.bin
// Legacy root-level snapshot files are still read for backward compatibility.
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
#include <array>
#include <cctype>
#include <cstddef>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
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

static bool compact_storage_enabled() {
    const char* value = std::getenv("VECTORDB_STORAGE_MODE");
    if (!value || !*value || std::strcmp(value, "performance") == 0)
        return false;
    if (std::strcmp(value, "compact") == 0)
        return true;
    throw std::invalid_argument(
        "Engine: VECTORDB_STORAGE_MODE must be 'performance' or 'compact'");
}

static void validate_collection_name(const std::string& name) {
    if (name.empty() || name.size() > 128 || name == "." || name == "..")
        throw std::invalid_argument("Engine: invalid collection name");
    for (unsigned char ch : name) {
        if (!std::isalnum(ch) && ch != '_' && ch != '-' && ch != '.')
            throw std::invalid_argument(
                "Engine: collection name may contain only letters, digits, '.', '-', '_'");
    }
}

static void validate_record_fields(const std::string& user_id,
                                   const MetadataEntry& meta) {
    constexpr size_t max_u16 = std::numeric_limits<uint16_t>::max();
    if (user_id.size() > max_u16)
        throw std::invalid_argument("Engine: user_id exceeds 65535 bytes");
    if (meta.strings.size() > max_u16 || meta.numerics.size() > max_u16)
        throw std::invalid_argument("Engine: too many metadata fields");
    for (const auto& [field, value] : meta.strings) {
        if (field.size() > max_u16 || value.size() > max_u16)
            throw std::invalid_argument("Engine: metadata string exceeds 65535 bytes");
    }
    for (const auto& [field, value] : meta.numerics) {
        (void)value;
        if (field.size() > max_u16)
            throw std::invalid_argument("Engine: metadata field exceeds 65535 bytes");
    }
}

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
// Crash-atomic checkpoint manifest
// ---------------------------------------------------------------------------

static constexpr uint64_t kCheckpointMagic = 0x564442434B505431ULL;  // VDBCKPT1
static constexpr uint32_t kCheckpointVersion = 1;

struct SnapshotFileDigest {
    uint64_t size;
    uint32_t crc32;
    uint32_t _pad;
};

struct CheckpointManifest {
    uint64_t magic;
    uint32_t version;
    uint32_t _pad;
    uint64_t generation;
    SnapshotFileDigest graph;
    SnapshotFileDigest metadata;
    SnapshotFileDigest id_map;
    uint32_t manifest_crc32;
    uint32_t _pad2;
};
static_assert(sizeof(CheckpointManifest) == 80,
              "CheckpointManifest layout changed");

static uint32_t crc32_update(uint32_t crc, const void* data, size_t len) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < len; ++i) {
        crc ^= bytes[i];
        for (int bit = 0; bit < 8; ++bit)
            crc = (crc & 1u) ? (0xEDB88320u ^ (crc >> 1)) : (crc >> 1);
    }
    return crc;
}

static uint32_t crc32_bytes(const void* data, size_t len) {
    return crc32_update(0xFFFFFFFFu, data, len) ^ 0xFFFFFFFFu;
}

static void checked_write_all(int fd, const void* data, size_t len,
                              const std::string& what) {
    const auto* p = static_cast<const uint8_t*>(data);
    while (len > 0) {
        ssize_t n = ::write(fd, p, len);
        if (n <= 0) throw std::runtime_error("Engine: write failed: " + what);
        p += n;
        len -= static_cast<size_t>(n);
    }
}

static void checked_read_all(int fd, void* data, size_t len,
                             const std::string& what) {
    auto* p = static_cast<uint8_t*>(data);
    while (len > 0) {
        ssize_t n = ::read(fd, p, len);
        if (n <= 0) throw std::runtime_error("Engine: read failed: " + what);
        p += n;
        len -= static_cast<size_t>(n);
    }
}

static SnapshotFileDigest digest_file(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("Engine: cannot checksum snapshot file: " + path);

    std::array<uint8_t, 64 * 1024> buf{};
    uint32_t crc = 0xFFFFFFFFu;
    uint64_t size = 0;
    while (true) {
        ssize_t n = ::read(fd, buf.data(), buf.size());
        if (n == 0) break;
        if (n < 0) {
            ::close(fd);
            throw std::runtime_error("Engine: snapshot checksum read failed: " + path);
        }
        crc = crc32_update(crc, buf.data(), static_cast<size_t>(n));
        size += static_cast<uint64_t>(n);
    }
    ::close(fd);
    return {size, crc ^ 0xFFFFFFFFu, 0};
}

static void fsync_directory(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("Engine: cannot open directory for fsync: " + path);
    if (::fsync(fd) != 0) {
        ::close(fd);
        throw std::runtime_error("Engine: directory fsync failed: " + path);
    }
    ::close(fd);
}

static std::string snapshot_dir_name(uint64_t generation) {
    return "checkpoint-" + std::to_string(generation);
}

static CheckpointManifest read_checkpoint_manifest(const std::string& path) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0)
        throw std::runtime_error("Engine: cannot open checkpoint manifest: " + path);
    CheckpointManifest manifest{};
    try {
        checked_read_all(fd, &manifest, sizeof(manifest), path);
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);

    if (manifest.magic != kCheckpointMagic ||
        manifest.version != kCheckpointVersion)
        throw std::runtime_error("Engine: invalid checkpoint manifest: " + path);
    uint32_t expected = crc32_bytes(
        &manifest, offsetof(CheckpointManifest, manifest_crc32));
    if (manifest.manifest_crc32 != expected)
        throw std::runtime_error("Engine: corrupt checkpoint manifest: " + path);
    return manifest;
}

static void validate_snapshot_file(const std::string& path,
                                   const SnapshotFileDigest& expected) {
    SnapshotFileDigest actual = digest_file(path);
    if (actual.size != expected.size || actual.crc32 != expected.crc32)
        throw std::runtime_error("Engine: snapshot checksum mismatch: " + path);
}

static void validate_checkpoint(const std::string& collection_dir,
                                const CheckpointManifest& manifest) {
    std::string root = collection_dir + "/" +
                       snapshot_dir_name(manifest.generation);
    validate_snapshot_file(root + "/graph.bin", manifest.graph);
    validate_snapshot_file(root + "/metadata.bin", manifest.metadata);
    validate_snapshot_file(root + "/id_map.bin", manifest.id_map);
}

static void publish_checkpoint_manifest(const std::string& collection_dir,
                                        CheckpointManifest manifest) {
    manifest.manifest_crc32 = crc32_bytes(
        &manifest, offsetof(CheckpointManifest, manifest_crc32));
    std::string path = collection_dir + "/checkpoint.current";
    std::string tmp = path + ".tmp";
    int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0)
        throw std::runtime_error("Engine: cannot write checkpoint manifest: " + tmp);
    try {
        checked_write_all(fd, &manifest, sizeof(manifest), tmp);
        if (::fsync(fd) != 0)
            throw std::runtime_error("Engine: checkpoint manifest fsync failed");
    } catch (...) {
        ::close(fd);
        throw;
    }
    ::close(fd);
    if (::rename(tmp.c_str(), path.c_str()) != 0)
        throw std::runtime_error("Engine: checkpoint manifest rename failed");
    fsync_directory(collection_dir);
}

static void cleanup_old_checkpoint_generations(
    const std::string& collection_dir, const std::string& current_dir) {
    for (const auto& entry : fs::directory_iterator(collection_dir)) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        bool generation = name.rfind("checkpoint-", 0) == 0;
        bool orphan_temp = name.rfind(".checkpoint-", 0) == 0 &&
                           name.size() >= 4 &&
                           name.compare(name.size() - 4, 4, ".tmp") == 0;
        if ((generation || orphan_temp) && entry.path().string() != current_dir)
            fs::remove_all(entry.path());
    }
    fsync_directory(collection_dir);
}

// Test-only crash injection. Production behavior is unchanged unless the
// explicitly named environment variable is present.
static void checkpoint_failpoint(const char* phase) {
    const char* requested = std::getenv("VECTORDB_CHECKPOINT_FAILPOINT");
    if (requested && std::strcmp(requested, phase) == 0) {
        ::kill(::getpid(), SIGKILL);
        ::_exit(86);
    }
}

static void insert_failpoint(const char* phase) {
    const char* requested = std::getenv("VECTORDB_INSERT_FAILPOINT");
    if (requested && std::strcmp(requested, phase) == 0) {
        ::kill(::getpid(), SIGKILL);
        ::_exit(86);
    }
}

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

    static std::optional<CheckpointManifest> current_manifest(
        const std::string& collection_dir, bool validate_files = true) {
        std::string path = collection_dir + "/checkpoint.current";
        if (!fs::exists(path)) return std::nullopt;
        CheckpointManifest manifest = read_checkpoint_manifest(path);
        if (validate_files) validate_checkpoint(collection_dir, manifest);
        return manifest;
    }

    static std::string snapshot_root(const std::string& collection_dir) {
        auto manifest = current_manifest(collection_dir);
        if (!manifest) return collection_dir;  // legacy root-level snapshot
        return collection_dir + "/" +
               snapshot_dir_name(manifest->generation);
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
        if (::fsync(fd) != 0) {
            ::close(fd);
            throw std::runtime_error("Engine: schema fsync failed");
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
        try {
            checked_write_all(fd, &count, sizeof(count), tmp);
            for (auto& [uid, node_id] : state.user_to_node) {
                if (uid.size() > std::numeric_limits<uint16_t>::max())
                    throw std::runtime_error("Engine: user_id too long for id_map");
                uint16_t uid_len = static_cast<uint16_t>(uid.size());
                checked_write_all(fd, &node_id, sizeof(node_id), tmp);
                checked_write_all(fd, &uid_len, sizeof(uid_len), tmp);
                checked_write_all(fd, uid.data(), uid_len, tmp);
            }
        } catch (...) {
            ::close(fd);
            throw;
        }
        if (::fsync(fd) != 0) {
            ::close(fd);
            throw std::runtime_error("Engine: fsync id_map failed");
        }
        ::close(fd);
        if (::rename(tmp.c_str(), path.c_str()) != 0)
            throw std::runtime_error("Engine: rename id_map failed: " + std::string(strerror(errno)));
    }

    // Open or recover a collection from disk.
    void open_collection(const CollectionSchema& schema) {
        const std::string& name = schema.name;
        std::string dir = col_dir(name);
        std::string snapshot = snapshot_root(dir);

        CollectionState state;
        state.schema = schema;
        state.wal   = std::make_unique<Wal>(dir + "/wal.log");
        state.vf    = std::make_unique<VectorFile>(dir + "/vectors.vdb", schema.dim);

        HnswConfig cfg;
        cfg.dim    = schema.dim;
        cfg.metric = schema.metric;
        cfg.store_vectors = !compact_storage_enabled();

        // Load graph snapshot if present.
        std::string graph_path = snapshot + "/graph.bin";
        if (fs::exists(graph_path)) {
            state.index = GraphSerializer::deserialize(graph_path, cfg);
        } else {
            state.index = std::make_unique<HnswIndex>(cfg);
        }
        state.index->set_external_vector_base(state.vf->data());

        // Load metadata snapshot if present.
        std::string meta_path = snapshot + "/metadata.bin";
        if (fs::exists(meta_path)) {
            state.meta = MetadataSerializer::deserialize(meta_path);
        } else {
            state.meta = std::make_unique<MetadataIndex>();
        }

        // Load id_map checkpoint if present (covers checkpoint-era entries).
        std::string idmap_path = snapshot + "/id_map.bin";
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
                state.vf->write(id, vec);
                state.index->set_external_vector_base(state.vf->data());
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
    validate_collection_name(schema.name);
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

size_t Engine::collection_dimension(const std::string& name) const {
    std::shared_lock lock(impl_->mu);
    auto it = impl_->cols.find(name);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + name);
    return it->second.schema.dim;
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
    if (!vec) throw std::invalid_argument("Engine: vector must not be null");
    validate_record_fields(user_id, meta);
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
        insert_failpoint("after_wal_sync");
        state.vf->write(node_id, vec);
        state.index->set_external_vector_base(state.vf->data());
        state.index->insert_for_recovery(node_id, vec);
    } else {
        // Reserve the stable ID, durably publish the operation, then mutate
        // the graph. The exclusive Engine lock prevents readers from observing
        // the reserved-but-not-yet-applied state.
        node_id = state.index->reserve_ids(1);
        effective_id = user_id.empty() ? std::to_string(node_id) : user_id;
        auto payload = make_insert_payload(node_id, effective_id, vec, state.schema.dim, meta);
        state.wal->append(WalRecordType::Insert, payload.data(),
                          static_cast<uint32_t>(payload.size()));
        state.wal->sync();
        insert_failpoint("after_wal_sync");
        state.vf->write(node_id, vec);
        state.index->set_external_vector_base(state.vf->data());
        state.index->insert_for_recovery(node_id, vec);
    }

    state.user_to_node[effective_id] = node_id;
    state.node_to_user[node_id] = effective_id;

    for (auto& [f, v] : meta.strings)  state.meta->insert_string(f, v, node_id);
    for (auto& [f, v] : meta.numerics) state.meta->insert_numeric(f, v, node_id);
    return effective_id;
}

std::vector<std::string> Engine::insert_batch(
    const std::string& collection,
    const std::vector<std::string>& user_ids,
    const float* vecs,
    size_t count,
    const std::vector<MetadataEntry>& metas,
    int num_threads)
{
    if (count == 0) return {};
    if (!vecs) throw std::invalid_argument("Engine: vectors must not be null");
    if (!user_ids.empty() && user_ids.size() != count)
        throw std::invalid_argument("Engine: user_ids size must match vector count");
    if (!metas.empty() && metas.size() != count)
        throw std::invalid_argument("Engine: metadata size must match vector count");
    if (num_threads < 0)
        throw std::invalid_argument("Engine: num_threads must be non-negative");
    for (size_t i = 0; i < count; ++i) {
        const std::string& user_id = user_ids.empty() ? std::string{} : user_ids[i];
        const MetadataEntry& meta = metas.empty() ? MetadataEntry{} : metas[i];
        validate_record_fields(user_id, meta);
    }

    std::unique_lock lock(impl_->mu);
    auto it = impl_->cols.find(collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + collection);
    auto& state = it->second;
    const size_t dim = state.schema.dim;

    std::vector<std::string>  assigned(count);
    std::vector<NodeId>       node_ids(count);

    std::unordered_set<std::string> ids_in_batch;
    for (const auto& uid : user_ids) {
        if (!uid.empty() && !ids_in_batch.insert(uid).second)
            throw std::invalid_argument("Engine: duplicate user_id in batch: " + uid);
    }

    // Fast path: if all user_ids are new (no re-inserts), use multi-threaded
    // HNSW insert. insert_batch_mt assigns IDs atomically, pre-grows all arrays,
    // then spawns hardware_concurrency() threads each inserting its slice.
    bool all_new = true;
    for (size_t i = 0; i < count && all_new; ++i) {
        const std::string& uid = user_ids.empty() ? "" : user_ids[i];
        if (!uid.empty() && state.user_to_node.count(uid))
            all_new = false;
    }

    if (all_new) {
        // Reserve IDs first so WAL records can be committed before graph
        // construction begins.
        NodeId first_id = state.index->reserve_ids(count);

        // Append all records; the single sync below is the batch commit point.
        for (size_t i = 0; i < count; ++i) {
            NodeId      node_id      = first_id + static_cast<NodeId>(i);
            const std::string& uid   = user_ids.empty() ? "" : user_ids[i];
            std::string effective_id = uid.empty() ? std::to_string(node_id) : uid;
            const float* vec         = vecs + i * dim;
            const MetadataEntry& meta = metas.empty() ? MetadataEntry{} : metas[i];

            auto payload = make_insert_payload(node_id, effective_id, vec, dim, meta);
            state.wal->append(WalRecordType::Insert, payload.data(),
                              static_cast<uint32_t>(payload.size()));

            node_ids[i]  = node_id;
            assigned[i]  = std::move(effective_id);
        }
    } else {
        // Slow path: assign stable IDs for a mixture of new and reinserted IDs,
        // but defer every graph mutation until after WAL sync.
        for (size_t i = 0; i < count; ++i) {
            const float*       vec     = vecs + i * dim;
            const std::string& user_id = user_ids.empty() ? "" : user_ids[i];
            const MetadataEntry& meta  = metas.empty() ? MetadataEntry{} : metas[i];

            NodeId      node_id;
            std::string effective_id;

            auto uid_it = (!user_id.empty())
                          ? state.user_to_node.find(user_id)
                          : state.user_to_node.end();

            if (!user_id.empty() && uid_it != state.user_to_node.end()) {
                effective_id = user_id;
                node_id      = uid_it->second;
            } else {
                node_id      = state.index->reserve_ids(1);
                effective_id = user_id.empty() ? std::to_string(node_id) : user_id;
            }

            auto payload = make_insert_payload(node_id, effective_id, vec, dim, meta);
            state.wal->append(WalRecordType::Insert, payload.data(),
                              static_cast<uint32_t>(payload.size()));

            node_ids[i]  = node_id;
            assigned[i]  = std::move(effective_id);
        }
    }

    // Single fdatasync for the whole batch.
    state.wal->sync();
    insert_failpoint("after_wal_sync");

    // Publish vectors to their stable NodeId slots before graph construction.
    // A growing write may remap VectorFile, so refresh the compact-mode base
    // only after every vector has been written.
    for (size_t i = 0; i < count; ++i)
        state.vf->write(node_ids[i], vecs + i * dim);
    state.index->set_external_vector_base(state.vf->data());

    if (all_new) {
        state.index->insert_reserved_batch_mt(
            node_ids.front(), vecs, count, num_threads);
    } else {
        for (size_t i = 0; i < count; ++i)
            state.index->insert_for_recovery(node_ids[i], vecs + i * dim);
    }

    // Publish ID maps and metadata after graph construction.
    for (size_t i = 0; i < count; ++i) {
        const MetadataEntry& meta = metas.empty() ? MetadataEntry{} : metas[i];
        state.user_to_node[assigned[i]] = node_ids[i];
        state.node_to_user[node_ids[i]] = assigned[i];
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

std::vector<std::vector<SearchResult>> Engine::search_batch(
    const BatchSearchRequest& req) const
{
    if (req.n_queries < 0)
        throw std::invalid_argument("Engine: n_queries must be non-negative");
    if (req.n_queries > 0 && !req.queries)
        throw std::invalid_argument("Engine: queries must not be null");
    if (req.top_k <= 0)
        throw std::invalid_argument("Engine: top_k must be positive");
    if (req.ef_search <= 0)
        throw std::invalid_argument("Engine: ef_search must be positive");
    if (req.num_threads < 0)
        throw std::invalid_argument("Engine: num_threads must be non-negative");
    std::shared_lock lock(impl_->mu);
    auto it = impl_->cols.find(req.collection);
    if (it == impl_->cols.end())
        throw std::runtime_error("Engine: collection not found: " + req.collection);
    const auto& state = it->second;

    bool has_filter = !req.filters.empty();
    std::unordered_set<NodeId> allowlist;
    if (has_filter) {
        bool first = true;
        for (auto& f : req.filters) {
            std::vector<NodeId> matches;
            if (f.op == FieldFilter::Op::Eq)
                matches = state.meta->query_eq(f.field, f.str_val);
            else
                matches = state.meta->query_range(f.field, f.lo, f.hi);
            if (first) {
                allowlist.insert(matches.begin(), matches.end());
                first = false;
            } else {
                std::unordered_set<NodeId> tmp;
                for (NodeId id : matches)
                    if (allowlist.count(id)) tmp.insert(id);
                allowlist = std::move(tmp);
            }
        }
    }

    int ef       = req.ef_search;
    if (has_filter) ef = std::max(ef, req.top_k * 5);
    int search_k = has_filter ? ef : req.top_k;

    auto raw_batch = state.index->search_batch(
        req.queries, req.n_queries, search_k, ef, req.num_threads);

    std::vector<std::vector<SearchResult>> results(req.n_queries);
    for (int i = 0; i < req.n_queries; ++i) {
        results[i].reserve(req.top_k);
        for (auto& [dist, node_id] : raw_batch[i]) {
            if (has_filter && !allowlist.count(node_id)) continue;
            auto uid_it = state.node_to_user.find(node_id);
            std::string uid = (uid_it != state.node_to_user.end())
                              ? uid_it->second : std::to_string(node_id);
            results[i].push_back({dist, std::move(uid)});
            if (static_cast<int>(results[i].size()) == req.top_k) break;
        }
    }
    return results;
}

std::vector<SearchResult> Engine::search(const SearchRequest& req) const {
    if (!req.query)
        throw std::invalid_argument("Engine: query must not be null");
    if (req.top_k <= 0)
        throw std::invalid_argument("Engine: top_k must be positive");
    if (req.ef_search <= 0)
        throw std::invalid_argument("Engine: ef_search must be positive");
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

    // Build an immutable snapshot generation without touching the currently
    // published one. A manifest rename is the single commit point, so startup
    // sees either the old complete generation or the new complete generation.
    uint64_t generation = 1;
    if (auto current = Impl::current_manifest(dir, false))
        generation = current->generation + 1;

    std::string final_dir;
    std::string temp_dir;
    while (true) {
        final_dir = dir + "/" + snapshot_dir_name(generation);
        temp_dir = dir + "/." + snapshot_dir_name(generation) + ".tmp";
        if (!fs::exists(final_dir) && !fs::exists(temp_dir)) break;
        ++generation;  // skip orphaned generations left by a prior crash
    }
    if (!fs::create_directory(temp_dir))
        throw std::runtime_error("Engine: cannot create checkpoint directory");

    // Compact mode reads vectors directly from the mmap. Persist those slots
    // before publishing a checkpoint that can make the corresponding WAL
    // records eligible for truncation.
    state.vf->sync();
    GraphSerializer::serialize(*state.index, temp_dir + "/graph.bin");
    MetadataSerializer::serialize(*state.meta, temp_dir + "/metadata.bin");
    Impl::save_id_map(temp_dir + "/id_map.bin", state);
    fsync_directory(temp_dir);
    checkpoint_failpoint("after_snapshot_write");

    if (::rename(temp_dir.c_str(), final_dir.c_str()) != 0)
        throw std::runtime_error("Engine: checkpoint directory rename failed");
    fsync_directory(dir);
    checkpoint_failpoint("after_snapshot_publish");

    CheckpointManifest manifest{};
    manifest.magic = kCheckpointMagic;
    manifest.version = kCheckpointVersion;
    manifest.generation = generation;
    manifest.graph = digest_file(final_dir + "/graph.bin");
    manifest.metadata = digest_file(final_dir + "/metadata.bin");
    manifest.id_map = digest_file(final_dir + "/id_map.bin");
    publish_checkpoint_manifest(dir, manifest);
    checkpoint_failpoint("after_manifest_publish");

    // WAL truncation happens strictly after manifest publication. If a crash
    // occurs before this point, replaying the retained records is idempotent.
    Lsn ckpt_lsn = state.wal->current_lsn();
    state.wal->truncate_before(ckpt_lsn);
    checkpoint_failpoint("after_wal_truncate");
    cleanup_old_checkpoint_generations(dir, final_dir);
}

}  // namespace vectordb
