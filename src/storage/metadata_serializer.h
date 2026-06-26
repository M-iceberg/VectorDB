// -----------------------------------------------------------------------------
// metadata_serializer.h — Snapshot MetadataIndex to disk and restore it
//
// Writes all (field, value, id) entries from a MetadataIndex to a binary file
// so the index can be restored after a crash without replaying the full WAL.
// Called once per checkpoint alongside GraphSerializer.
//
// File format (little-endian):
//   [magic:          8B = 0x4D455441440A0000]
//   [version:        4B = 1]
//   [num_strings:    4B]
//   [num_numerics:   4B]
//   string entries (num_strings ×):
//     [field_len: 4B][field bytes][val_len: 4B][val bytes][id: 4B]
//   numeric entries (num_numerics ×):
//     [field_len: 4B][field bytes][val: 8B double][id: 4B]
// -----------------------------------------------------------------------------
#pragma once
#include "metadata_index.h"
#include <memory>
#include <string>

namespace vectordb {

class MetadataSerializer {
public:
    static void serialize(const MetadataIndex& idx, const std::string& path);
    static std::unique_ptr<MetadataIndex> deserialize(const std::string& path);
};

}  // namespace vectordb
