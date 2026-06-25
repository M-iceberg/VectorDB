// -----------------------------------------------------------------------------
// wal_record.h — On-disk WAL record layout
//
// Defines the binary format of a single WAL record. All fields are
// little-endian. The CRC32 covers everything after the crc32 field itself
// (i.e. payload_length + type + timestamp_us + payload).
//
// On-disk layout:
//   [crc32: 4B][payload_length: 4B][type: 1B][timestamp_us: 8B][payload: N B]
//
// Types:
//   WalRecordType::Insert     — payload is (node_id: 4B, vector: dim*4B,
//                               optional metadata bytes)
//   WalRecordType::Delete     — payload is (node_id: 4B)
//   WalRecordType::Checkpoint — payload is (lsn: 8B); marks a safe
//                               truncation point in the WAL
//
//   WalRecordHeader           — packed struct matching the on-disk prefix
// -----------------------------------------------------------------------------
#pragma once
#include <cstdint>
#include <cstring>
#include <vector>

namespace vectordb {

enum class WalRecordType : uint8_t {
    Insert     = 1,
    Delete     = 2,
    Checkpoint = 3,
};

// On-disk record layout (little-endian):
//   [crc32: 4B][payload_length: 4B][type: 1B][timestamp_us: 8B][payload: N B]
// CRC covers everything after the crc32 field.
struct WalRecordHeader {
    uint32_t       crc32;
    uint32_t       payload_length;
    WalRecordType  type;
    uint64_t       timestamp_us;
} __attribute__((packed));

// ---------------------------------------------------------------------------
// Payload helpers
// ---------------------------------------------------------------------------

// Insert payload: [node_id: 4B][vec: dim * 4B]
inline std::vector<uint8_t> make_insert_payload(
    uint32_t id, const float* vec, size_t dim)
{
    std::vector<uint8_t> buf(sizeof(uint32_t) + dim * sizeof(float));
    std::memcpy(buf.data(), &id, sizeof(uint32_t));
    std::memcpy(buf.data() + sizeof(uint32_t), vec, dim * sizeof(float));
    return buf;
}

// Delete payload: [node_id: 4B]
inline std::vector<uint8_t> make_delete_payload(uint32_t id)
{
    std::vector<uint8_t> buf(sizeof(uint32_t));
    std::memcpy(buf.data(), &id, sizeof(uint32_t));
    return buf;
}

}  // namespace vectordb
