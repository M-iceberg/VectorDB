// -----------------------------------------------------------------------------
// vector_file.h — mmap-backed flat vector storage
//
// Stores float32 vectors in fixed-size slots on disk using MAP_SHARED mmap.
// Writes go directly to the file via the mmap region; msync() controls
// explicit flush. madvise(MADV_RANDOM) is set because HNSW access patterns
// are non-sequential.
//
// Growth: when capacity is exhausted, the file is extended via ftruncate(),
// the old mapping is unmapped, and a new mapping is created over the larger
// file. Callers must not cache raw pointers across an append() call.
//
// API:
//   VectorFile(path, dim)
//       Opens or creates the file. dim is fixed for the lifetime of the file
//       and must match the collection's dimensionality.
//
//   append(vec) → uint32_t
//       Writes vec to the next free slot and returns its slot index.
//       The slot index is stable and used as the NodeId in HnswIndex.
//
//   read(slot) → const float*
//       Returns a zero-copy pointer into the mmap region for the given slot.
//       Valid until the next append() that triggers a remap.
//
//   slot_count() → size_t
//       Total number of slots written (including tombstoned vectors).
// -----------------------------------------------------------------------------
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
    void          write(uint32_t slot, const float* vec); // insert or overwrite a stable NodeId slot
    const float*  read(uint32_t slot) const;      // zero-copy pointer into mmap region
    const float*  data() const;                   // contiguous slot-0 base; invalidated by a growing write
    size_t        slot_count() const;
    void          sync();                         // persist header and populated slots

    VectorFile(const VectorFile&) = delete;
    VectorFile& operator=(const VectorFile&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
