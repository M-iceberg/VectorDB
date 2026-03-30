// mmap-backed flat vector storage — implemented Day 13.
#include "vector_file.h"

namespace vectordb {

struct VectorFile::Impl {
    std::string path;
    size_t dim;
    size_t slot_count = 0;
    // TODO Day 13: fd, mmap ptr, file size tracking
};

VectorFile::VectorFile(const std::string& path, size_t dim)
    : impl_(std::make_unique<Impl>()) {
    impl_->path = path;
    impl_->dim  = dim;
}

VectorFile::~VectorFile() = default;

uint32_t VectorFile::append(const float* vec) {
    // TODO Day 13: ftruncate + mmap extend + write slot
    (void)vec;
    return static_cast<uint32_t>(impl_->slot_count++);
}

const float* VectorFile::read(uint32_t slot) const {
    // TODO Day 13: return pointer into mmap region
    (void)slot;
    return nullptr;
}

size_t VectorFile::slot_count() const {
    return impl_->slot_count;
}

}  // namespace vectordb
