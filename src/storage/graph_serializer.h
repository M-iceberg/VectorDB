// -----------------------------------------------------------------------------
// graph_serializer.h — HNSW graph checkpoint serialization
//
// Serializes the full HNSW adjacency list to a binary file and deserializes
// it back into a live HnswIndex. Used by the checkpoint flow (Day 15) to
// snapshot the graph so WAL can be truncated.
//
// Binary format (host byte order):
//   [node_count: 4B]
//   per node: [id: 4B][layer: 4B][tombstone: 1B]
//             [neighbor_count_per_layer: 4B × (layer+1)]
//             [neighbor_ids: 4B × total_neighbors]
//
// API:
//   GraphSerializer::serialize(index, path)
//       Writes the complete adjacency list of `index` to `path`.
//       Overwrites the file if it already exists.
//
//   GraphSerializer::deserialize(path, cfg) → unique_ptr<HnswIndex>
//       Reads the file and returns a fully-reconstructed HnswIndex.
//       `cfg` must match the original build parameters (dim, M, M0, metric).
//       Throws std::runtime_error on format mismatch or I/O error.
// -----------------------------------------------------------------------------
#pragma once
#include <memory>
#include <string>

namespace vectordb {
class  HnswIndex;
struct HnswConfig;
}  // namespace vectordb

namespace vectordb {

// Serializes and deserializes the HNSW adjacency list to/from disk.
// Format: binary adjacency list — node count, per-node layer + neighbor lists.
class GraphSerializer {
public:
    // Writes the complete adjacency list of `index` to `path`.
    static void serialize(const HnswIndex& index, const std::string& path);

    // Reads from `path` and returns a fully-reconstructed HnswIndex.
    // `cfg` must match the original build parameters (dim, M, M0, metric).
    static std::unique_ptr<HnswIndex> deserialize(const std::string& path,
                                                   const HnswConfig& cfg);
};

}  // namespace vectordb
