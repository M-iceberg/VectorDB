// -----------------------------------------------------------------------------
// metadata_serializer.cpp — Metadata index snapshot (Day 17)
// -----------------------------------------------------------------------------
#include "metadata_serializer.h"
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <unistd.h>

namespace vectordb {

static constexpr uint64_t kMetaMagic   = 0x4D455441440A0000ULL;
static constexpr uint32_t kMetaVersion = 1;

// ---------------------------------------------------------------------------
// RAII fd guard
// ---------------------------------------------------------------------------

struct FdGuard {
    int fd;
    explicit FdGuard(int f) : fd(f) {}
    ~FdGuard() { if (fd >= 0) ::close(fd); }
    FdGuard(const FdGuard&) = delete;
};

static void checked_write(int fd, const void* buf, size_t n) {
    if (::write(fd, buf, n) != static_cast<ssize_t>(n))
        throw std::runtime_error("MetadataSerializer: write failed");
}

static void checked_read(int fd, void* buf, size_t n) {
    if (::read(fd, buf, n) != static_cast<ssize_t>(n))
        throw std::runtime_error("MetadataSerializer: read failed or truncated file");
}

static void write_str(int fd, const std::string& s) {
    uint32_t len = static_cast<uint32_t>(s.size());
    checked_write(fd, &len, sizeof(len));
    if (len > 0) checked_write(fd, s.data(), len);
}

static std::string read_str(int fd) {
    uint32_t len = 0;
    checked_read(fd, &len, sizeof(len));
    if (len == 0) return {};
    std::string s(len, '\0');
    checked_read(fd, s.data(), len);
    return s;
}

// ---------------------------------------------------------------------------
// serialize
// ---------------------------------------------------------------------------

void MetadataSerializer::serialize(const MetadataIndex& idx,
                                   const std::string& path) {
    // Collect all entries first so we can write counts upfront.
    struct SEntry { std::string field, value; NodeId id; };
    struct NEntry { std::string field; double value; NodeId id; };
    std::vector<SEntry> strings;
    std::vector<NEntry> numerics;

    idx.for_each_string([&](const std::string& f, const std::string& v, NodeId id) {
        strings.push_back({f, v, id});
    });
    idx.for_each_numeric([&](const std::string& f, double v, NodeId id) {
        numerics.push_back({f, v, id});
    });

    FdGuard g(::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644));
    if (g.fd < 0)
        throw std::runtime_error("MetadataSerializer: cannot open for write: " + path);

    // Header
    checked_write(g.fd, &kMetaMagic,   sizeof(kMetaMagic));
    checked_write(g.fd, &kMetaVersion, sizeof(kMetaVersion));
    uint32_t ns = static_cast<uint32_t>(strings.size());
    uint32_t nn = static_cast<uint32_t>(numerics.size());
    checked_write(g.fd, &ns, sizeof(ns));
    checked_write(g.fd, &nn, sizeof(nn));

    for (auto& e : strings) {
        write_str(g.fd, e.field);
        write_str(g.fd, e.value);
        checked_write(g.fd, &e.id, sizeof(e.id));
    }
    for (auto& e : numerics) {
        write_str(g.fd, e.field);
        checked_write(g.fd, &e.value, sizeof(e.value));
        checked_write(g.fd, &e.id, sizeof(e.id));
    }
    if (::fsync(g.fd) != 0)
        throw std::runtime_error("MetadataSerializer: fsync failed: " + path);
}

// ---------------------------------------------------------------------------
// deserialize
// ---------------------------------------------------------------------------

std::unique_ptr<MetadataIndex> MetadataSerializer::deserialize(
    const std::string& path) {
    FdGuard g(::open(path.c_str(), O_RDONLY));
    if (g.fd < 0)
        throw std::runtime_error("MetadataSerializer: cannot open: " + path);

    uint64_t magic = 0;
    checked_read(g.fd, &magic, sizeof(magic));
    if (magic != kMetaMagic)
        throw std::runtime_error("MetadataSerializer: bad magic in " + path);

    uint32_t version = 0;
    checked_read(g.fd, &version, sizeof(version));
    if (version != kMetaVersion)
        throw std::runtime_error("MetadataSerializer: unsupported version");

    uint32_t ns = 0, nn = 0;
    checked_read(g.fd, &ns, sizeof(ns));
    checked_read(g.fd, &nn, sizeof(nn));

    auto idx = std::make_unique<MetadataIndex>();

    for (uint32_t i = 0; i < ns; ++i) {
        std::string field = read_str(g.fd);
        std::string value = read_str(g.fd);
        NodeId id = 0;
        checked_read(g.fd, &id, sizeof(id));
        idx->insert_string(field, value, id);
    }
    for (uint32_t i = 0; i < nn; ++i) {
        std::string field = read_str(g.fd);
        double value = 0;
        checked_read(g.fd, &value, sizeof(value));
        NodeId id = 0;
        checked_read(g.fd, &id, sizeof(id));
        idx->insert_numeric(field, value, id);
    }

    return idx;
}

}  // namespace vectordb
