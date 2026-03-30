// WAL append / iterate / truncate — implemented Day 11–12.
#include "wal.h"

namespace vectordb {

struct Wal::Impl {
    std::string path;
    Lsn current_lsn = 0;
    // TODO Day 11: file descriptor, write buffer
};

Wal::Wal(const std::string& path) : impl_(std::make_unique<Impl>()) {
    impl_->path = path;
}

Wal::~Wal() = default;

Lsn Wal::append(WalRecordType type, const void* payload, uint32_t length) {
    // TODO Day 11: write header + payload + fdatasync
    (void)type; (void)payload; (void)length;
    return ++impl_->current_lsn;
}

void Wal::sync() {
    // TODO Day 11: fdatasync(fd_)
}

void Wal::iterate(Lsn start_lsn,
                  std::function<void(Lsn, WalRecordType,
                                     const void*, uint32_t)> cb) const {
    // TODO Day 12: sequential reader + CRC check + corrupt-tail handling
    (void)start_lsn; (void)cb;
}

void Wal::truncate_before(Lsn lsn) {
    // TODO Day 15: called after checkpoint
    (void)lsn;
}

Lsn Wal::current_lsn() const {
    return impl_->current_lsn;
}

}  // namespace vectordb
