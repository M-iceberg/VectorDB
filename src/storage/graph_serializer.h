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
