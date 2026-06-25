// -----------------------------------------------------------------------------
// graph_serializer.cpp — HNSW graph checkpoint serialization  (Day 14)
//
// File format (host byte order):
//   [GraphFileHeader: 32B]
//   per node (node_count times):
//     [id: 4B][layer: 4B][tombstone: 1B]
//     [vec: dim × 4B]
//     per layer (layer+1 times):
//       [nbr_count: 4B][nbr_ids: nbr_count × 4B]
// -----------------------------------------------------------------------------
#include "graph_serializer.h"
#include "../core/hnsw_index.h"
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace vectordb {

static constexpr uint64_t kGraphMagic   = 0x47524150480A0000ULL;
static constexpr uint32_t kGraphVersion = 1;

struct GraphFileHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t node_count;
    uint32_t entry_point;
    int32_t  max_layer;
    uint32_t dim;        // vector dimension — validated on deserialize
    uint32_t live_count; // non-tombstoned node count
} __attribute__((packed));
static_assert(sizeof(GraphFileHeader) == 32, "GraphFileHeader must be 32 bytes");

// RAII wrapper: closes fd on scope exit, whether normal return or exception.
struct FdGuard {
    int fd;
    explicit FdGuard(int f) : fd(f) {}
    ~FdGuard() { if (fd >= 0) ::close(fd); }
    FdGuard(const FdGuard&) = delete;
    FdGuard& operator=(const FdGuard&) = delete;
};

static void checked_write(int fd, const void* buf, size_t n) {
    if (::write(fd, buf, n) != static_cast<ssize_t>(n))
        throw std::runtime_error("GraphSerializer: write failed");
}

static void checked_read(int fd, void* buf, size_t n) {
    if (::read(fd, buf, n) != static_cast<ssize_t>(n))
        throw std::runtime_error("GraphSerializer: read failed or truncated file");
}

void GraphSerializer::serialize(const HnswIndex& index, const std::string& path) {
    auto nodes = index.snapshot();

    int raw_fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (raw_fd < 0)
        throw std::runtime_error("GraphSerializer: cannot open for write: " + path);
    FdGuard guard(raw_fd);

    GraphFileHeader hdr{};
    hdr.magic       = kGraphMagic;
    hdr.version     = kGraphVersion;
    hdr.node_count  = static_cast<uint32_t>(nodes.size());
    hdr.entry_point = index.entry_point_id();
    hdr.max_layer   = static_cast<int32_t>(index.max_layer_val());
    hdr.dim         = static_cast<uint32_t>(nodes.empty() ? 0 : nodes[0].vec.size());
    hdr.live_count  = static_cast<uint32_t>(index.size());
    checked_write(raw_fd, &hdr, sizeof(hdr));

    for (auto& nd : nodes) {
        uint32_t nid  = nd.id;
        int32_t  nlyr = static_cast<int32_t>(nd.layer);
        uint8_t  tomb = nd.tombstone ? 1u : 0u;
        checked_write(raw_fd, &nid,  sizeof(nid));
        checked_write(raw_fd, &nlyr, sizeof(nlyr));
        checked_write(raw_fd, &tomb, sizeof(tomb));

        checked_write(raw_fd, nd.vec.data(), nd.vec.size() * sizeof(float));

        for (int l = 0; l <= nd.layer; ++l) {
            const auto& nbrs = nd.neighbors[l];
            uint32_t cnt = static_cast<uint32_t>(nbrs.size());
            checked_write(raw_fd, &cnt, sizeof(cnt));
            if (cnt > 0)
                checked_write(raw_fd, nbrs.data(), cnt * sizeof(NodeId));
        }
    }
    // FdGuard closes raw_fd here
}

std::unique_ptr<HnswIndex> GraphSerializer::deserialize(
    const std::string& path, const HnswConfig& cfg)
{
    int raw_fd = ::open(path.c_str(), O_RDONLY);
    if (raw_fd < 0)
        throw std::runtime_error("GraphSerializer: cannot open: " + path);
    FdGuard guard(raw_fd);

    GraphFileHeader hdr;
    checked_read(raw_fd, &hdr, sizeof(hdr));
    if (hdr.magic != kGraphMagic)
        throw std::runtime_error("GraphSerializer: bad magic");
    if (hdr.version != kGraphVersion)
        throw std::runtime_error("GraphSerializer: unsupported version");
    if (hdr.node_count > 0 && hdr.dim != static_cast<uint32_t>(cfg.dim))
        throw std::runtime_error(
            "GraphSerializer: dim mismatch — file has " + std::to_string(hdr.dim) +
            ", cfg has " + std::to_string(cfg.dim));

    std::vector<HnswIndex::NodeData> nodes;
    nodes.reserve(hdr.node_count);

    for (uint32_t n = 0; n < hdr.node_count; ++n) {
        HnswIndex::NodeData nd;
        uint32_t nid;
        int32_t  nlyr;
        uint8_t  tomb;
        checked_read(raw_fd, &nid,  sizeof(nid));
        checked_read(raw_fd, &nlyr, sizeof(nlyr));
        checked_read(raw_fd, &tomb, sizeof(tomb));

        nd.id        = nid;
        nd.layer     = static_cast<int>(nlyr);
        nd.tombstone = (tomb != 0);

        nd.vec.resize(cfg.dim);
        checked_read(raw_fd, nd.vec.data(), cfg.dim * sizeof(float));

        nd.neighbors.resize(static_cast<size_t>(nlyr) + 1);
        for (int l = 0; l <= nlyr; ++l) {
            uint32_t cnt;
            checked_read(raw_fd, &cnt, sizeof(cnt));
            nd.neighbors[l].resize(cnt);
            if (cnt > 0)
                checked_read(raw_fd, nd.neighbors[l].data(), cnt * sizeof(NodeId));
        }

        nodes.push_back(std::move(nd));
    }
    // FdGuard closes raw_fd here

    auto index = std::make_unique<HnswIndex>(cfg);
    index->restore(hdr.entry_point, static_cast<int>(hdr.max_layer),
                   static_cast<size_t>(hdr.live_count), std::move(nodes));
    return index;
}

}  // namespace vectordb
