#include <gtest/gtest.h>
#include "core/hnsw_index.h"
#include "storage/graph_serializer.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <vector>

namespace vectordb {
namespace {

class GraphSerializerTest : public ::testing::Test {
protected:
    std::string path_;
    void SetUp() override {
        path_ = std::filesystem::temp_directory_path() / "test_graph_XXXXXX";
        path_ += std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    void TearDown() override {
        std::filesystem::remove(path_);
    }

    static HnswConfig default_cfg(size_t dim = 16) {
        HnswConfig cfg;
        cfg.dim            = dim;
        cfg.metric         = Metric::L2;
        cfg.M              = 8;
        cfg.M0             = 16;
        cfg.ef_construction = 50;
        return cfg;
    }
};

// ---------------------------------------------------------------------------
// Roundtrip: serialize then deserialize, verify graph structure is identical
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, RoundtripGraphStructure) {
    const size_t dim = 16;
    const size_t N   = 200;

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto cfg = default_cfg(dim);
    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (auto& v : vecs)
        for (auto& x : v) x = dist(rng);

    HnswIndex original(cfg);
    for (size_t i = 0; i < N; ++i)
        original.insert_for_recovery(static_cast<NodeId>(i), vecs[i].data());

    GraphSerializer::serialize(original, path_);
    auto restored = GraphSerializer::deserialize(path_, cfg);

    EXPECT_EQ(restored->size(), original.size());

    // Verify neighbor lists are identical for every node at every layer.
    for (size_t i = 0; i < N; ++i) {
        NodeId id = static_cast<NodeId>(i);
        int orig_layer = original.node_layer(id);
        EXPECT_EQ(restored->node_layer(id), orig_layer);
        for (int l = 0; l <= orig_layer; ++l) {
            auto orig_nbrs    = original.neighbors_of(id, l);
            auto restored_nbrs = restored->neighbors_of(id, l);
            std::sort(orig_nbrs.begin(),     orig_nbrs.end());
            std::sort(restored_nbrs.begin(), restored_nbrs.end());
            EXPECT_EQ(restored_nbrs, orig_nbrs)
                << "node=" << id << " layer=" << l;
        }
    }
}

// ---------------------------------------------------------------------------
// Recall: search results after deserialize match results before serialize
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, SearchRecallUnchangedAfterRoundtrip) {
    const size_t dim    = 16;
    const size_t N      = 500;
    const size_t Q      = 20;
    const int    k      = 10;
    const int    ef     = 50;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    auto cfg = default_cfg(dim);
    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (auto& v : vecs)
        for (auto& x : v) x = dist(rng);

    HnswIndex original(cfg);
    for (size_t i = 0; i < N; ++i)
        original.insert_for_recovery(static_cast<NodeId>(i), vecs[i].data());

    // Run queries before serialization.
    std::vector<std::vector<float>> queries(Q, std::vector<float>(dim));
    for (auto& q : queries)
        for (auto& x : q) x = dist(rng);

    std::vector<std::vector<std::pair<float, NodeId>>> before(Q);
    for (size_t q = 0; q < Q; ++q)
        before[q] = original.search(queries[q].data(), k, ef);

    GraphSerializer::serialize(original, path_);
    auto restored = GraphSerializer::deserialize(path_, cfg);

    // Same queries after deserialization must return identical results.
    for (size_t q = 0; q < Q; ++q) {
        auto after = restored->search(queries[q].data(), k, ef);
        ASSERT_EQ(after.size(), before[q].size()) << "query " << q;
        for (size_t r = 0; r < after.size(); ++r) {
            EXPECT_EQ(after[r].second, before[q][r].second)
                << "query=" << q << " rank=" << r;
            EXPECT_FLOAT_EQ(after[r].first, before[q][r].first)
                << "query=" << q << " rank=" << r;
        }
    }
}

// ---------------------------------------------------------------------------
// Large-scale recall: 10K vectors, verify search results identical after
// serialize → deserialize (graph structure fully preserved at scale)
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, LargeScaleRecallUnchangedAfterRoundtrip) {
    const size_t dim = 16;
    const size_t N   = 10000;
    const size_t Q   = 50;
    const int    k   = 10;
    const int    ef  = 64;

    HnswConfig cfg;
    cfg.dim             = dim;
    cfg.metric          = Metric::L2;
    cfg.M               = 8;
    cfg.M0              = 16;
    cfg.ef_construction = 32;  // smaller ef for speed at 10K scale

    std::mt19937 rng(2024);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (auto& v : vecs)
        for (auto& x : v) x = dist(rng);

    HnswIndex original(cfg);
    for (size_t i = 0; i < N; ++i)
        original.insert_for_recovery(static_cast<NodeId>(i), vecs[i].data());

    // Run Q queries before serialization.
    std::vector<std::vector<float>> queries(Q, std::vector<float>(dim));
    for (auto& q : queries)
        for (auto& x : q) x = dist(rng);

    std::vector<std::vector<std::pair<float, NodeId>>> before(Q);
    for (size_t q = 0; q < Q; ++q)
        before[q] = original.search(queries[q].data(), k, ef);

    GraphSerializer::serialize(original, path_);
    auto restored = GraphSerializer::deserialize(path_, cfg);

    EXPECT_EQ(restored->size(), original.size());

    // Search results after deserialize must be identical to before serialize.
    size_t total_hits = 0;
    for (size_t q = 0; q < Q; ++q) {
        auto after = restored->search(queries[q].data(), k, ef);
        ASSERT_EQ(after.size(), before[q].size()) << "query " << q;
        for (size_t r = 0; r < after.size(); ++r) {
            if (after[r].second == before[q][r].second) ++total_hits;
        }
    }

    // Every result must match exactly — graph structure is deterministic.
    EXPECT_EQ(total_hits, Q * static_cast<size_t>(k));
}

// ---------------------------------------------------------------------------
// Tombstones survive roundtrip
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, TombstonesPreservedAfterRoundtrip) {
    const size_t dim = 8;
    const size_t N   = 100;

    std::mt19937 rng(7);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto cfg = default_cfg(dim);

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (auto& v : vecs) for (auto& x : v) x = d(rng);

    HnswIndex original(cfg);
    for (size_t i = 0; i < N; ++i)
        original.insert_for_recovery(static_cast<NodeId>(i), vecs[i].data());

    // Delete every even node.
    for (size_t i = 0; i < N; i += 2)
        original.remove(static_cast<NodeId>(i));

    size_t live_before = original.size();

    GraphSerializer::serialize(original, path_);
    auto restored = GraphSerializer::deserialize(path_, cfg);

    EXPECT_EQ(restored->size(), live_before);

    // Deleted nodes must not appear in search results.
    std::vector<float> query(dim, 0.0f);
    auto results = restored->search(query.data(), 10, 50);
    for (auto& [dist, id] : results)
        EXPECT_NE(id % 2, 0u) << "tombstoned node " << id << " appeared in results";
}

// ---------------------------------------------------------------------------
// Bad magic throws on deserialize
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, BadMagicThrows) {
    {
        std::ofstream f(path_, std::ios::binary);
        uint64_t junk = 0xDEADBEEFDEADBEEFULL;
        f.write(reinterpret_cast<const char*>(&junk), sizeof(junk));
    }
    EXPECT_THROW(GraphSerializer::deserialize(path_, default_cfg()),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// Dim mismatch throws on deserialize
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, DimMismatchThrowsOnDeserialize) {
    const size_t dim = 16;
    auto cfg = default_cfg(dim);

    std::mt19937 rng(99);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    HnswIndex index(cfg);
    for (uint32_t i = 0; i < 10; ++i) {
        std::vector<float> v(dim);
        for (auto& x : v) x = d(rng);
        index.insert_for_recovery(i, v.data());
    }
    GraphSerializer::serialize(index, path_);

    // Attempt to deserialize with a different dimension.
    auto wrong_cfg = default_cfg(32);
    EXPECT_THROW(GraphSerializer::deserialize(path_, wrong_cfg),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// Truncated file throws on deserialize (not silently corrupt)
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, TruncatedFileThrows) {
    const size_t dim = 8;
    auto cfg = default_cfg(dim);

    std::mt19937 rng(55);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    HnswIndex index(cfg);
    for (uint32_t i = 0; i < 20; ++i) {
        std::vector<float> v(dim);
        for (auto& x : v) x = d(rng);
        index.insert_for_recovery(i, v.data());
    }
    GraphSerializer::serialize(index, path_);

    // Truncate the file to half its size.
    auto sz = std::filesystem::file_size(path_);
    std::filesystem::resize_file(path_, sz / 2);

    EXPECT_THROW(GraphSerializer::deserialize(path_, cfg), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Insert after deserialize: restored index accepts new inserts correctly
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, InsertAfterDeserialize) {
    const size_t dim = 16;
    const size_t N   = 100;
    const size_t M   = 10;  // new nodes added after restore

    std::mt19937 rng(11);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    auto cfg = default_cfg(dim);

    std::vector<std::vector<float>> vecs(N + M, std::vector<float>(dim));
    for (auto& v : vecs) for (auto& x : v) x = d(rng);

    HnswIndex original(cfg);
    for (size_t i = 0; i < N; ++i)
        original.insert_for_recovery(static_cast<NodeId>(i), vecs[i].data());

    GraphSerializer::serialize(original, path_);
    auto restored = GraphSerializer::deserialize(path_, cfg);

    // Insert M more nodes into the restored index.
    for (size_t i = N; i < N + M; ++i)
        restored->insert_for_recovery(static_cast<NodeId>(i), vecs[i].data());

    EXPECT_EQ(restored->size(), N + M);

    // All N + M nodes must be searchable.
    std::vector<float> query(dim);
    for (auto& x : query) x = d(rng);
    auto results = restored->search(query.data(), 10, 50);
    EXPECT_EQ(static_cast<int>(results.size()), 10);

    // Results must be sorted ascending by distance.
    for (size_t r = 1; r < results.size(); ++r)
        EXPECT_LE(results[r - 1].first, results[r].first);
}

// ---------------------------------------------------------------------------
// Serialize twice to same path: second call correctly overwrites first
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, OverwriteExistingSnapshot) {
    const size_t dim = 8;
    auto cfg = default_cfg(dim);
    std::mt19937 rng(22);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    // First snapshot: 50 nodes.
    HnswIndex first(cfg);
    for (uint32_t i = 0; i < 50; ++i) {
        std::vector<float> v(dim);
        for (auto& x : v) x = d(rng);
        first.insert_for_recovery(i, v.data());
    }
    GraphSerializer::serialize(first, path_);

    // Second snapshot: 20 nodes (completely different index).
    HnswIndex second(cfg);
    for (uint32_t i = 0; i < 20; ++i) {
        std::vector<float> v(dim);
        for (auto& x : v) x = d(rng);
        second.insert_for_recovery(i, v.data());
    }
    GraphSerializer::serialize(second, path_);

    // Deserialize should see the second snapshot (20 nodes), not the first (50).
    auto restored = GraphSerializer::deserialize(path_, cfg);
    EXPECT_EQ(restored->size(), 20u);
}

// ---------------------------------------------------------------------------
// restore() on non-empty index throws
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, RestoreOnNonEmptyIndexThrows) {
    const size_t dim = 8;
    auto cfg = default_cfg(dim);
    std::mt19937 rng(33);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    HnswIndex index(cfg);
    std::vector<float> v(dim);
    for (auto& x : v) x = d(rng);
    index.insert_for_recovery(0, v.data());

    std::vector<HnswIndex::NodeData> empty_nodes;
    EXPECT_THROW(index.restore(kInvalidNode, -1, 0, std::move(empty_nodes)),
                 std::logic_error);
}

// ---------------------------------------------------------------------------
// Empty index roundtrip
// ---------------------------------------------------------------------------

TEST_F(GraphSerializerTest, EmptyIndexRoundtrip) {
    auto cfg = default_cfg();
    HnswIndex empty(cfg);
    GraphSerializer::serialize(empty, path_);
    auto restored = GraphSerializer::deserialize(path_, cfg);
    EXPECT_EQ(restored->size(), 0u);

    // Search on empty restored index must return nothing, not crash.
    std::vector<float> query(cfg.dim, 0.0f);
    auto results = restored->search(query.data(), 5, 20);
    EXPECT_TRUE(results.empty());
}

}  // namespace
}  // namespace vectordb
