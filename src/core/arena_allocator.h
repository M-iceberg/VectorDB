// -----------------------------------------------------------------------------
// arena_allocator.h — Slab-based arena allocator for fixed-size float vectors
//
// Amortizes per-insertion malloc overhead on the hot insert path. All slots
// are 64-byte aligned to match cache line boundaries and SIMD load requirements.
// Not thread-safe; the Engine's write lock covers all allocations.
//
// API:
//   ArenaAllocator(slot_bytes, initial_slots)
//       Constructs the arena. slot_bytes must be a multiple of 64.
//       Pre-allocates initial_slots slabs; grows automatically on demand.
//
//   alloc() → float*
//       Returns a 64-byte aligned slot. Never returns null; throws on OOM.
//       O(1) amortized (free-list pop, or new slab allocation).
//
//   free(ptr)
//       Returns the slot to the internal free list. O(1).
//       ptr must have been returned by alloc() on this allocator.
//
//   capacity() → size_t   total allocated slots (grows on demand)
//   used()     → size_t   slots currently checked out
// -----------------------------------------------------------------------------
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
