// -----------------------------------------------------------------------------
// vector_file.cpp — mmap-backed flat vector storage  (Day 13)
//
// File layout:
//   [FileHeader: 64B][slot 0: dim*4B][slot 1: dim*4B]...[slot capacity-1]
//
// The entire file is mapped MAP_SHARED. append() writes directly into the
// mmap region and bumps slot_count in the header. When capacity is exhausted,
// ftruncate() extends the file and the mapping is replaced (munmap + mmap).
// Callers must not cache raw pointers returned by read() across an append()
// that triggers a remap.
// -----------------------------------------------------------------------------
#include "vector_file.h"
#include <algorithm>
#include <cassert>
#include <cstring>
#include <fcntl.h>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace vectordb {

static constexpr uint64_t kMagic           = 0x564442564543540AULL;  // "VDBVECTn"
static constexpr uint32_t kVersion         = 1;
static constexpr uint64_t kInitialCapacity = 1024;
static constexpr size_t   kHeaderSize      = 64;

// On-disk header, always kHeaderSize (64) bytes.
struct FileHeader {
    uint64_t magic;
    uint32_t version;
    uint32_t dim;
    uint64_t slot_count;   // number of slots written
    uint64_t capacity;     // total allocated slots in file
    uint8_t  _pad[32];
} __attribute__((packed));
static_assert(sizeof(FileHeader) == kHeaderSize, "FileHeader must be 64 bytes");

struct VectorFile::Impl {
    int      fd       = -1;
    void*    map_     = MAP_FAILED;
    size_t   map_size = 0;
    size_t   dim_;

    Impl(const std::string& path, size_t dim) : dim_(dim) {
        bool is_new = (::access(path.c_str(), F_OK) != 0);

        fd = ::open(path.c_str(), O_RDWR | O_CREAT, 0644);
        if (fd < 0)
            throw std::runtime_error("VectorFile: cannot open: " + path);

        if (is_new) {
            // Write a blank header and allocate space for initial capacity.
            FileHeader hdr{};
            hdr.magic      = kMagic;
            hdr.version    = kVersion;
            hdr.dim        = static_cast<uint32_t>(dim);
            hdr.slot_count = 0;
            hdr.capacity   = kInitialCapacity;
            if (::write(fd, &hdr, sizeof(hdr)) != sizeof(hdr))
                throw std::runtime_error("VectorFile: header write failed");

            size_t data_size = kInitialCapacity * dim * sizeof(float);
            if (::ftruncate(fd, static_cast<off_t>(kHeaderSize + data_size)) != 0)
                throw std::runtime_error("VectorFile: ftruncate failed");
        }

        // Map the whole file.
        struct stat st;
        if (::fstat(fd, &st) != 0)
            throw std::runtime_error("VectorFile: fstat failed");
        map_size = static_cast<size_t>(st.st_size);

        map_ = ::mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map_ == MAP_FAILED)
            throw std::runtime_error("VectorFile: mmap failed");
        ::madvise(map_, map_size, MADV_RANDOM);

        // Validate existing file.
        auto* hdr = header();
        if (hdr->magic != kMagic)
            throw std::runtime_error("VectorFile: bad magic");
        if (hdr->version != kVersion)
            throw std::runtime_error("VectorFile: unsupported version");
        if (hdr->dim != static_cast<uint32_t>(dim))
            throw std::runtime_error("VectorFile: dim mismatch");
    }

    ~Impl() {
        if (map_ != MAP_FAILED) ::munmap(map_, map_size);
        if (fd >= 0) ::close(fd);
    }

    FileHeader* header() { return reinterpret_cast<FileHeader*>(map_); }
    const FileHeader* header() const { return reinterpret_cast<const FileHeader*>(map_); }

    float* data_ptr() {
        return reinterpret_cast<float*>(static_cast<uint8_t*>(map_) + kHeaderSize);
    }
    const float* data_ptr() const {
        return reinterpret_cast<const float*>(
            static_cast<const uint8_t*>(map_) + kHeaderSize);
    }

    void grow() {
        auto* hdr = header();
        uint64_t new_cap  = hdr->capacity * 2;
        size_t   new_size = kHeaderSize + new_cap * dim_ * sizeof(float);

        if (::ftruncate(fd, static_cast<off_t>(new_size)) != 0)
            throw std::runtime_error("VectorFile: grow ftruncate failed");

        ::munmap(map_, map_size);
        map_size = new_size;
        map_ = ::mmap(nullptr, map_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (map_ == MAP_FAILED)
            throw std::runtime_error("VectorFile: grow mmap failed");
        ::madvise(map_, map_size, MADV_RANDOM);

        header()->capacity = new_cap;
    }
};

VectorFile::VectorFile(const std::string& path, size_t dim)
    : impl_(std::make_unique<Impl>(path, dim)) {}

VectorFile::~VectorFile() = default;

uint32_t VectorFile::append(const float* vec) {
    auto& I = *impl_;
    uint32_t slot = static_cast<uint32_t>(I.header()->slot_count);
    write(slot, vec);
    return slot;
}

void VectorFile::write(uint32_t slot, const float* vec) {
    if (!vec) throw std::invalid_argument("VectorFile: vector must not be null");
    auto& I = *impl_;
    while (slot >= I.header()->capacity)
        I.grow();

    // Re-fetch the header and data base after grow(); the mmap may have moved.
    auto* hdr = I.header();
    std::memcpy(I.data_ptr() + static_cast<size_t>(slot) * I.dim_,
                vec, I.dim_ * sizeof(float));
    hdr->slot_count = std::max<uint64_t>(hdr->slot_count,
                                        static_cast<uint64_t>(slot) + 1);
}

const float* VectorFile::read(uint32_t slot) const {
    assert(slot < impl_->header()->slot_count);
    return impl_->data_ptr() + slot * impl_->dim_;
}

const float* VectorFile::data() const {
    return impl_->data_ptr();
}

size_t VectorFile::slot_count() const {
    return static_cast<size_t>(impl_->header()->slot_count);
}

void VectorFile::sync() {
    auto& I = *impl_;
    const size_t populated = kHeaderSize +
        static_cast<size_t>(I.header()->slot_count) * I.dim_ * sizeof(float);
    if (::msync(I.map_, populated, MS_SYNC) != 0)
        throw std::runtime_error("VectorFile: msync failed");
    if (::fsync(I.fd) != 0)
        throw std::runtime_error("VectorFile: fsync failed");
}

}  // namespace vectordb
