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
//   WalRecordType::Insert     — payload is:
//       [node_id: 4B][vec: dim*4B]
//       optionally followed by metadata:
//       [num_strings: 2B]
//         for each: [field_len: 2B][field bytes][val_len: 2B][val bytes]
//       [num_numerics: 2B]
//         for each: [field_len: 2B][field bytes][val: 8B double]
//       replay() requires vec_dim to locate the metadata section.
//
//   WalRecordType::Delete     — payload is (node_id: 4B)
//   WalRecordType::Checkpoint — payload is (lsn: 8B); marks a safe
//                               truncation point in the WAL
//
//   WalRecordHeader           — packed struct matching the on-disk prefix
// -----------------------------------------------------------------------------
#pragma once
#include "core/collection.h"  // MetadataEntry
#include <cstdint>
#include <cstring>
#include <string>
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

// Insert payload: [node_id: 4B][vec: dim*4B]
// optionally followed by metadata section if meta is non-empty.
inline std::vector<uint8_t> make_insert_payload(
    uint32_t id, const float* vec, size_t dim,
    const MetadataEntry& meta = {})
{
    // Base: id + vector
    std::vector<uint8_t> buf;
    buf.resize(sizeof(uint32_t) + dim * sizeof(float));
    std::memcpy(buf.data(), &id, sizeof(uint32_t));
    std::memcpy(buf.data() + sizeof(uint32_t), vec, dim * sizeof(float));

    if (meta.strings.empty() && meta.numerics.empty())
        return buf;

    // Append string metadata: [num_strings: 2B] then entries
    uint16_t ns = static_cast<uint16_t>(meta.strings.size());
    buf.resize(buf.size() + sizeof(uint16_t));
    std::memcpy(buf.data() + buf.size() - sizeof(uint16_t), &ns, sizeof(uint16_t));
    for (auto& [field, value] : meta.strings) {
        uint16_t fl = static_cast<uint16_t>(field.size());
        uint16_t vl = static_cast<uint16_t>(value.size());
        size_t off = buf.size();
        buf.resize(off + sizeof(uint16_t) + fl + sizeof(uint16_t) + vl);
        std::memcpy(buf.data() + off, &fl, sizeof(uint16_t)); off += sizeof(uint16_t);
        std::memcpy(buf.data() + off, field.data(), fl);      off += fl;
        std::memcpy(buf.data() + off, &vl, sizeof(uint16_t)); off += sizeof(uint16_t);
        std::memcpy(buf.data() + off, value.data(), vl);
    }

    // Append numeric metadata: [num_numerics: 2B] then entries
    uint16_t nn = static_cast<uint16_t>(meta.numerics.size());
    buf.resize(buf.size() + sizeof(uint16_t));
    std::memcpy(buf.data() + buf.size() - sizeof(uint16_t), &nn, sizeof(uint16_t));
    for (auto& [field, value] : meta.numerics) {
        uint16_t fl = static_cast<uint16_t>(field.size());
        size_t off = buf.size();
        buf.resize(off + sizeof(uint16_t) + fl + sizeof(double));
        std::memcpy(buf.data() + off, &fl, sizeof(uint16_t)); off += sizeof(uint16_t);
        std::memcpy(buf.data() + off, field.data(), fl);      off += fl;
        std::memcpy(buf.data() + off, &value, sizeof(double));
    }

    return buf;
}

// Parse metadata section from an Insert payload, given that the vector
// occupies bytes [4, 4 + vec_dim*4). Returns empty MetadataEntry if no
// metadata section is present.
inline MetadataEntry parse_insert_metadata(
    const void* payload, uint32_t payload_len, size_t vec_dim)
{
    MetadataEntry meta;
    size_t base = sizeof(uint32_t) + vec_dim * sizeof(float);
    if (payload_len <= base) return meta;

    const uint8_t* p   = static_cast<const uint8_t*>(payload) + base;
    const uint8_t* end = static_cast<const uint8_t*>(payload) + payload_len;

    auto read_u16 = [&]() -> uint16_t {
        if (p + sizeof(uint16_t) > end) return 0;
        uint16_t v; std::memcpy(&v, p, sizeof(v)); p += sizeof(v); return v;
    };
    auto read_str = [&](uint16_t len) -> std::string {
        if (p + len > end) return {};
        std::string s(reinterpret_cast<const char*>(p), len); p += len; return s;
    };

    uint16_t ns = read_u16();
    for (uint16_t i = 0; i < ns; ++i) {
        uint16_t fl = read_u16();
        std::string field = read_str(fl);
        uint16_t vl = read_u16();
        std::string value = read_str(vl);
        meta.strings[field] = value;
    }

    uint16_t nn = read_u16();
    for (uint16_t i = 0; i < nn; ++i) {
        uint16_t fl = read_u16();
        std::string field = read_str(fl);
        if (p + sizeof(double) > end) break;
        double value; std::memcpy(&value, p, sizeof(double)); p += sizeof(double);
        meta.numerics[field] = value;
    }

    return meta;
}

// Delete payload: [node_id: 4B]
inline std::vector<uint8_t> make_delete_payload(uint32_t id)
{
    std::vector<uint8_t> buf(sizeof(uint32_t));
    std::memcpy(buf.data(), &id, sizeof(uint32_t));
    return buf;
}

}  // namespace vectordb
