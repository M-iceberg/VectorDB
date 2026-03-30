// Slab arena allocator — implemented Day 13 alongside VectorFile.
#include "arena_allocator.h"
#include <stdexcept>
#include <vector>

namespace vectordb {

struct ArenaAllocator::Impl {
    size_t slot_bytes;
    size_t used_count = 0;
    // TODO Day 13: slab list, free list, aligned allocation via posix_memalign
};

ArenaAllocator::ArenaAllocator(size_t slot_bytes, size_t /*initial_slots*/)
    : impl_(std::make_unique<Impl>()) {
    impl_->slot_bytes = slot_bytes;
}

ArenaAllocator::~ArenaAllocator() = default;

float* ArenaAllocator::alloc() {
    // TODO Day 13
    throw std::runtime_error("ArenaAllocator::alloc not implemented");
}

void ArenaAllocator::free(float* ptr) {
    // TODO Day 13
    (void)ptr;
}

size_t ArenaAllocator::capacity() const { return 0; }
size_t ArenaAllocator::used()     const { return impl_->used_count; }

}  // namespace vectordb
