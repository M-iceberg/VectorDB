#pragma once
#include "wal_record.h"
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace vectordb {

using Lsn = uint64_t;

// Append-only write-ahead log.
// Records are sequentially written; fdatasync() is called per client batch.
// On crash, the reader truncates any incomplete tail record.
class Wal {
public:
    explicit Wal(const std::string& path);
    ~Wal();

    // Appends a record and returns its assigned LSN.
    Lsn append(WalRecordType type, const void* payload, uint32_t length);

    // Calls fdatasync(). Should be called once per client batch request.
    void sync();

    // Iterates all records with LSN >= start_lsn.
    // Stops silently on a corrupt tail record.
    void iterate(Lsn start_lsn,
                 std::function<void(Lsn, WalRecordType,
                                    const void* payload, uint32_t length)> cb) const;

    // Truncates everything before `lsn` (called after a successful checkpoint).
    void truncate_before(Lsn lsn);

    Lsn current_lsn() const;

    Wal(const Wal&) = delete;
    Wal& operator=(const Wal&) = delete;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
