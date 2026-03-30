#pragma once
#include <cstddef>
#include <memory>

namespace vectordb {

// Slab-based arena allocator for fixed-size, 64-byte aligned float vectors.
// Amortizes malloc overhead for the hot insert path.
// Not thread-safe; external locking required.
class ArenaAllocator {
public:
    // `slot_bytes` must be a multiple of 64.
    explicit ArenaAllocator(size_t slot_bytes, size_t initial_slots = 1024);
    ~ArenaAllocator();

    float* alloc();          // returns a 64-byte aligned slot; never returns null
    void   free(float* ptr); // returns slot to the free list

    size_t capacity() const; // total allocated slots (grows on demand)
    size_t used() const;     // slots currently checked out

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
