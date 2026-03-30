#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace vectordb {

// mmap-backed flat vector storage (MAP_SHARED).
// Vectors are stored in fixed-size slots; the file grows via ftruncate + remap.
// madvise(MADV_RANDOM) is set because HNSW access patterns are non-sequential.
class VectorFile {
public:
    VectorFile(const std::string& path, size_t dim);
    ~VectorFile();

    uint32_t      append(const float* vec);       // returns the assigned slot index
    const float*  read(uint32_t slot) const;      // zero-copy pointer into mmap region
    size_t        slot_count() const;

    VectorFile(const VectorFile&) = delete;
    VectorFile& operator=(const VectorFile&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
