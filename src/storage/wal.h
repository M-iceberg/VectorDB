// -----------------------------------------------------------------------------
// wal.h — Append-only Write-Ahead Log
//
// Provides crash-safe durability for insert and delete operations.
// Records are appended sequentially; fdatasync() is called once per client
// batch to bound the durability window. On crash, the reader detects and
// silently discards any incomplete tail record (partial write).
//
// Recovery flow (Engine startup):
//   1. Load the most recent checkpoint (graph snapshot + LSN).
//   2. Call Wal::iterate(checkpoint_lsn, cb) to replay records since then.
//   3. Call Wal::truncate_before(new_lsn) after the next checkpoint.
//
// API:
//   Wal(path)
//       Opens (or creates) the WAL file at path.
//
//   append(type, payload, length) → Lsn
//       Appends one record and returns its assigned LSN (monotonically
//       increasing uint64). Writes are buffered until sync() is called.
//
//   sync()
//       Calls fdatasync(). Must be called at the end of each client batch
//       to guarantee durability before returning a success response.
//
//   iterate(start_lsn, cb)
//       Reads all records with LSN ≥ start_lsn and calls cb for each.
//       Stops silently on a corrupt tail (CRC mismatch or truncated header).
//
//   truncate_before(lsn)
//       Removes all records with LSN < lsn from the file. Called after a
//       successful checkpoint to bound WAL growth.
//
//   current_lsn() → Lsn
//       Returns the LSN that will be assigned to the next append.
// -----------------------------------------------------------------------------
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
