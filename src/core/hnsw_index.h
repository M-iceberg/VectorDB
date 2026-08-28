// -----------------------------------------------------------------------------
// hnsw_index.h — Hierarchical Navigable Small World (HNSW) index
//
// Approximate nearest-neighbor index based on the HNSW algorithm
// (Malkov & Yashunin 2018). Supports insert, k-NN search, and soft-delete.
//
// HnswConfig:
//   dim              — vector dimensionality (must match VectorFile)
//   metric           — distance function (L2 / Cosine / InnerProduct)
//   M                — max neighbors per node on layers 1+ (typ. 16)
//   M0               — max neighbors at layer 0 (typ. 2*M = 32)
//   ef_construction  — candidate pool size during insert's beam search.
//                      beam search keeps up to ef_construction candidates,
//                      then picks the closest M (or M0) from that pool to
//                      connect as neighbors. higher → better neighbor quality
//                      → higher recall, but slower inserts. only affects
//                      build time; has no effect during search. (typ. 200)
//
// API:
//   HnswIndex(cfg)
//       Constructs an empty index with the given config.
//
//   insert(id, vec)
//       Inserts vector `vec` under key `id`. Assigns a random layer,
//       runs greedy descent from the top layer, then beam-searches
//       ef_construction neighbors at the assigned layer and below.
//       Creates bidirectional edges with heuristic pruning.
//
//   search(query, k, ef_search) → vector<(distance, id)>
//       Returns up to k nearest neighbors sorted by ascending distance.
//       ef_search controls the beam width (higher → better recall, slower).
//
//   remove(id)
//       Soft-delete: marks the node as tombstone. Does not relink neighbors.
//       Tombstoned nodes are excluded from search results but may still
//       be traversed as graph intermediaries.
//
//   size() → size_t
//       Count of live (non-tombstoned) nodes.
// -----------------------------------------------------------------------------
#pragma once
#include "hnsw_node.h"
#include "distance.h"
#include <memory>
#include <utility>
#include <vector>

namespace vectordb {

struct HnswConfig {
    size_t dim = 128;
    Metric metric = Metric::L2;
    int    M = 16;              // max neighbors per node (layers 1+)
    int    M0 = 32;             // max neighbors at layer 0 (= 2*M)
    int    ef_construction = 200;
    bool   heuristic = true;    // true = Algorithm 4 diversity heuristic; false = greedy (for comparison only)
    // Performance mode keeps a private vector copy beside layer-0 adjacency.
    // Compact mode stores only adjacency and reads vectors from an external
    // contiguous source (normally VectorFile's mmap).
    bool   store_vectors = true;
};

// HNSW approximate nearest-neighbor index.
// insert / search / remove (tombstone) — see hnsw_index.cpp for implementation.
class HnswIndex {
public:
    explicit HnswIndex(HnswConfig cfg);  // explicit prevents implicit conversion:
                                         // compiler cannot silently turn a HnswConfig
                                         // into a HnswIndex; must call it directly.
    ~HnswIndex();

    // Auto-assigns sequential NodeId internally using next_id_ counter.
    // Returns the assigned NodeId.
    NodeId insert(const float* vec);

    // Used only by WAL replay and re-inserts of existing user_ids.
    // Updates next_id_ = max(next_id_, id+1).
    void insert_for_recovery(NodeId id, const float* vec);

    // Parallel batch insert: assigns count sequential NodeIds, pre-grows all
    // arrays, then spawns num_threads threads each inserting its slice.
    // num_threads <= 0 means hardware_concurrency(). Returns first assigned id.
    // IDs returned are first_id, first_id+1, ..., first_id+count-1.
    NodeId insert_batch_mt(const float* vecs, size_t count, int num_threads = 0);

    // Persistence path: reserve stable IDs, commit them to the WAL, then build
    // the graph using those exact IDs. This keeps WAL publication ahead of
    // in-memory graph mutation without sacrificing parallel construction.
    NodeId reserve_ids(size_t count);
    void insert_reserved_batch_mt(NodeId first_id, const float* vecs,
                                  size_t count, int num_threads = 0);

    // Supplies the contiguous [node_id * dim] vector array used when
    // HnswConfig::store_vectors is false. The caller must refresh the pointer
    // after remapping the backing store and keep it valid during index calls.
    void set_external_vector_base(const float* vectors);

    // Returns up to k (distance, id) pairs sorted by ascending distance.
    std::vector<std::pair<float, NodeId>> search(
        const float* query, int k, int ef_search) const;

    // Parallel batch search: runs n_queries independent searches across
    // num_threads threads. Thread-safe: each thread owns its tl_visited.
    // queries is row-major: queries[i*dim .. (i+1)*dim-1] is query i.
    // num_threads <= 0 means hardware_concurrency().
    std::vector<std::vector<std::pair<float, NodeId>>> search_batch(
        const float* queries, int n_queries, int k, int ef_search,
        int num_threads = 0) const;

    // Soft-delete: marks node as tombstone. Does not relink neighbors.
    void remove(NodeId id);

    size_t size() const;  // live (non-tombstoned) node count

    // Graph inspection — for testing and debugging only.
    // Returns the number of neighbors node `id` has at `layer`.
    // Returns 0 if the node doesn't exist or doesn't participate in `layer`.
    size_t neighbor_count(NodeId id, int layer) const;

    // Returns the highest layer this node participates in.
    // Returns -1 if the node doesn't exist.
    int node_layer(NodeId id) const;

    // Returns the neighbor list of node `id` at `layer`.
    // Returns empty vector if the node doesn't exist or doesn't participate in `layer`.
    std::vector<NodeId> neighbors_of(NodeId id, int layer) const;

    HnswIndex(const HnswIndex&) = delete;
    HnswIndex& operator=(const HnswIndex&) = delete;

    // -----------------------------------------------------------------------
    // Serialization support — used by GraphSerializer only.
    // -----------------------------------------------------------------------

    // Per-node snapshot: everything needed to reconstruct the graph.
    struct NodeData {
        NodeId id;
        int    layer;
        bool   tombstone;
        std::vector<float>              vec;        // raw float vector
        std::vector<std::vector<NodeId>> neighbors; // per-layer neighbor lists
    };

    // Returns all node data and global graph state for serialization.
    std::vector<NodeData> snapshot() const;
    NodeId entry_point_id() const;
    int    max_layer_val()   const;

    // Restores graph from serialized data, bypassing the normal insert algorithm.
    // Called by GraphSerializer::deserialize() after constructing an empty index.
    void restore(NodeId entry_point, int max_layer, size_t live_count,
                 std::vector<NodeData> nodes);
    // Pimpl (pointer to implementation): all internal state (nodes, vectors,
    // entry point, rng, etc.) is defined in hnsw_index.cpp, not here.
    // This header only forward-declares Impl — callers cannot access or even
    // see the internal fields. The unique_ptr owns the Impl and frees it on
    // destruction. To change internals, only hnsw_index.cpp needs recompiling.
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace vectordb
