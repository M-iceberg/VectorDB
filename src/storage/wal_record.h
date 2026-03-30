#pragma once
#include <cstdint>

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

}  // namespace vectordb
