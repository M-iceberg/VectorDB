// -----------------------------------------------------------------------------
// test_hnsw.cpp — Unit tests for HnswIndex
//
// Tests:
//   Basic:        empty search, single insert, size tracking
//   Correctness:  nearest neighbor is found, search caps at k
//   Tombstone:    remove() excludes node from results
//   Recall:       brute-force vs HNSW on 200 random vectors, recall >= 0.9
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "core/hnsw_index.h"
#include <algorithm>
#include <random>
#include <unordered_set>
#include <vector>

namespace vectordb {
namespace {

// Build a default config with small ef_construction for fast tests.
HnswConfig test_cfg(size_t dim = 4, Metric m = Metric::L2) {
    HnswConfig cfg;
    cfg.dim            = dim;
    cfg.metric         = m;
    cfg.M              = 8;
    cfg.M0             = 16;
    cfg.ef_construction = 50;
    return cfg;
}

std::vector<float> make_vec(size_t dim, float fill) {
    return std::vector<float>(dim, fill);
}

// Random float vector with a fixed seed offset.
std::vector<float> random_vec(size_t dim, uint64_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> v(dim);
    for (auto& x : v) x = dist(rng);
    return v;
}

// Brute-force k-NN (L2) over a set of stored vectors.
std::vector<NodeId> brute_force_knn(
    const float* query,
    const std::vector<std::vector<float>>& corpus,
    int k)
{
    size_t dim = corpus[0].size();
    std::vector<std::pair<float, NodeId>> dists;
    dists.reserve(corpus.size());
    for (size_t i = 0; i < corpus.size(); ++i) {
        float d = 0.0f;
        for (size_t j = 0; j < dim; ++j) {
            float diff = query[j] - corpus[i][j];
            d += diff * diff;
        }
        dists.push_back({d, static_cast<NodeId>(i)});
    }
    std::sort(dists.begin(), dists.end());
    std::vector<NodeId> ids;
    for (int i = 0; i < k && i < static_cast<int>(dists.size()); ++i) {
        ids.push_back(dists[i].second);
    }
    return ids;
}

// ---------------------------------------------------------------------------
// Basic tests
// ---------------------------------------------------------------------------

TEST(HnswBasic, EmptySearchReturnsEmpty) {
    HnswIndex idx(test_cfg());
    auto q = make_vec(4, 0.0f);
    auto result = idx.search(q.data(), 5, 10);
    EXPECT_TRUE(result.empty());
}

TEST(HnswBasic, SizeZeroInitially) {
    HnswIndex idx(test_cfg());
    EXPECT_EQ(idx.size(), 0u);
}

TEST(HnswBasic, InsertIncreasesSize) {
    HnswIndex idx(test_cfg());
    auto v = make_vec(4, 1.0f);
    idx.insert(0, v.data());
    EXPECT_EQ(idx.size(), 1u);
    idx.insert(1, v.data());
    EXPECT_EQ(idx.size(), 2u);
}

TEST(HnswBasic, SearchSingleNodeReturnsIt) {
    HnswIndex idx(test_cfg());
    auto v = make_vec(4, 1.0f);
    idx.insert(42, v.data());

    auto q = make_vec(4, 0.9f);
    auto result = idx.search(q.data(), 1, 10);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].second, 42u);
}

TEST(HnswBasic, SearchReturnsAtMostK) {
    HnswIndex idx(test_cfg());
    for (NodeId i = 0; i < 20; ++i) {
        auto v = random_vec(4, i);
        idx.insert(i, v.data());
    }
    auto q = random_vec(4, 999);
    auto result = idx.search(q.data(), 3, 20);
    EXPECT_LE(result.size(), 3u);
}

TEST(HnswBasic, SearchResultsSortedAscending) {
    HnswIndex idx(test_cfg());
    for (NodeId i = 0; i < 10; ++i) {
        auto v = random_vec(4, i);
        idx.insert(i, v.data());
    }
    auto q = random_vec(4, 999);
    auto result = idx.search(q.data(), 5, 20);
    for (size_t i = 1; i < result.size(); ++i) {
        EXPECT_LE(result[i-1].first, result[i].first);
    }
}

// ---------------------------------------------------------------------------
// Nearest neighbor correctness
// ---------------------------------------------------------------------------

TEST(HnswCorrectness, FindsExactNearestNeighbor) {
    // Insert 50 random vectors, then insert the query itself as node 99
    // (distance 0). All vectors share the same neighborhood so node 99
    // gets connected into the graph normally and must be returned as top-1.
    const size_t dim = 8;
    HnswIndex idx(test_cfg(dim));

    auto query = random_vec(dim, 777);

    for (NodeId i = 0; i < 50; ++i) {
        auto v = random_vec(dim, 1000 + i);
        idx.insert(i, v.data());
    }

    // Insert the query itself — distance from query = 0.
    idx.insert(99, query.data());

    auto result = idx.search(query.data(), 1, 50);
    ASSERT_FALSE(result.empty());
    EXPECT_EQ(result[0].second, 99u);
    EXPECT_NEAR(result[0].first, 0.0f, 1e-5f);
}

TEST(HnswCorrectness, DistancesAreNonNegative) {
    HnswIndex idx(test_cfg());
    for (NodeId i = 0; i < 10; ++i) {
        auto v = random_vec(4, i * 7);
        idx.insert(i, v.data());
    }
    auto q = random_vec(4, 555);
    auto result = idx.search(q.data(), 5, 20);
    for (auto& [d, id] : result) {
        EXPECT_GE(d, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// Tombstone (soft delete)
// ---------------------------------------------------------------------------

TEST(HnswTombstone, RemoveDecreasesSize) {
    HnswIndex idx(test_cfg());
    auto v = make_vec(4, 1.0f);
    idx.insert(0, v.data());
    idx.insert(1, v.data());
    EXPECT_EQ(idx.size(), 2u);
    idx.remove(0);
    EXPECT_EQ(idx.size(), 1u);
}

TEST(HnswTombstone, RemovedNodeNotReturnedInSearch) {
    const size_t dim = 4;
    HnswIndex idx(test_cfg(dim));

    // Insert a node very close to the query.
    std::vector<float> query = {1.0f, 0.0f, 0.0f, 0.0f};
    std::vector<float> close = {1.1f, 0.0f, 0.0f, 0.0f};
    std::vector<float> far   = {9.0f, 0.0f, 0.0f, 0.0f};

    idx.insert(1, close.data());
    idx.insert(2, far.data());

    // Confirm close is returned first.
    auto before = idx.search(query.data(), 1, 10);
    ASSERT_FALSE(before.empty());
    EXPECT_EQ(before[0].second, 1u);

    // Remove the close node.
    idx.remove(1);

    // Now only far should be returned.
    auto after = idx.search(query.data(), 1, 10);
    ASSERT_FALSE(after.empty());
    EXPECT_EQ(after[0].second, 2u);
}

TEST(HnswTombstone, RemoveIdempotent) {
    HnswIndex idx(test_cfg());
    auto v = make_vec(4, 0.5f);
    idx.insert(5, v.data());
    idx.remove(5);
    idx.remove(5);  // second remove should be a no-op
    EXPECT_EQ(idx.size(), 0u);
}

TEST(HnswTombstone, RemoveNonExistentNodeNoOp) {
    HnswIndex idx(test_cfg());
    idx.remove(999);  // should not crash
    EXPECT_EQ(idx.size(), 0u);
}

// ---------------------------------------------------------------------------
// Recall test: HNSW vs brute force on random data
// ---------------------------------------------------------------------------

TEST(HnswRecall, RecallAtLeast90Percent) {
    const size_t dim = 32;
    const size_t N   = 200;
    const int    k   = 10;
    const int    num_queries = 20;

    HnswConfig cfg;
    cfg.dim             = dim;
    cfg.metric          = Metric::L2;
    cfg.M               = 16;
    cfg.M0              = 32;
    cfg.ef_construction = 100;

    HnswIndex idx(cfg);
    std::vector<std::vector<float>> corpus(N);
    for (size_t i = 0; i < N; ++i) {
        corpus[i] = random_vec(dim, i);
        idx.insert(static_cast<NodeId>(i), corpus[i].data());
    }

    int total_hits = 0;
    int total_expected = num_queries * k;

    for (int q = 0; q < num_queries; ++q) {
        auto query = random_vec(dim, 10000 + q);

        auto truth = brute_force_knn(query.data(), corpus, k);
        std::unordered_set<NodeId> truth_set(truth.begin(), truth.end());

        auto result = idx.search(query.data(), k, 50);
        for (auto& [d, id] : result) {
            if (truth_set.count(id)) ++total_hits;
        }
    }

    double recall = static_cast<double>(total_hits) / total_expected;
    EXPECT_GE(recall, 0.9) << "Recall = " << recall;
}

// ---------------------------------------------------------------------------
// Metric coverage: Cosine and InnerProduct
// ---------------------------------------------------------------------------

TEST(HnswMetric, CosineFindsNearestNeighbor) {
    const size_t dim = 8;
    HnswIndex idx(test_cfg(dim, Metric::Cosine));
    auto query = random_vec(dim, 42);
    for (NodeId i = 0; i < 50; ++i)
        idx.insert(i, random_vec(dim, 200 + i).data());
    idx.insert(99, query.data());
    auto result = idx.search(query.data(), 1, 50);
    ASSERT_FALSE(result.empty());
    EXPECT_EQ(result[0].second, 99u);
}

TEST(HnswMetric, InnerProductFindsNearestNeighbor) {
    const size_t dim = 8;
    HnswIndex idx(test_cfg(dim, Metric::InnerProduct));
    auto query = random_vec(dim, 43);
    for (NodeId i = 0; i < 50; ++i)
        idx.insert(i, random_vec(dim, 300 + i).data());
    idx.insert(99, query.data());
    auto result = idx.search(query.data(), 1, 50);
    ASSERT_FALSE(result.empty());
    EXPECT_EQ(result[0].second, 99u);
}

// ---------------------------------------------------------------------------
// ef_search: higher ef_search should not reduce recall
// ---------------------------------------------------------------------------

TEST(HnswEfSearch, HigherEfSearchNeverWorseRecall) {
    const size_t dim = 16;
    const size_t N = 100;
    const int k = 5;

    HnswConfig cfg;
    cfg.dim = dim; cfg.metric = Metric::L2;
    cfg.M = 16; cfg.M0 = 32; cfg.ef_construction = 100;

    HnswIndex idx(cfg);
    std::vector<std::vector<float>> corpus(N);
    for (size_t i = 0; i < N; ++i) {
        corpus[i] = random_vec(dim, i + 500);
        idx.insert(static_cast<NodeId>(i), corpus[i].data());
    }

    auto query = random_vec(dim, 9999);
    auto truth = brute_force_knn(query.data(), corpus, k);
    std::unordered_set<NodeId> truth_set(truth.begin(), truth.end());

    int hits_low_ef = 0, hits_high_ef = 0;
    for (auto& [d, id] : idx.search(query.data(), k, 10))
        if (truth_set.count(id)) ++hits_low_ef;
    for (auto& [d, id] : idx.search(query.data(), k, 100))
        if (truth_set.count(id)) ++hits_high_ef;

    EXPECT_GE(hits_high_ef, hits_low_ef);
}

// ---------------------------------------------------------------------------
// size() accuracy
// ---------------------------------------------------------------------------

TEST(HnswSize, SizeAccurateAfterManyInserts) {
    HnswIndex idx(test_cfg(4));
    for (NodeId i = 0; i < 100; ++i)
        idx.insert(i, random_vec(4, i).data());
    EXPECT_EQ(idx.size(), 100u);
}

TEST(HnswSize, SizeAccurateAfterMixedInsertRemove) {
    HnswIndex idx(test_cfg(4));
    for (NodeId i = 0; i < 50; ++i)
        idx.insert(i, random_vec(4, i).data());
    for (NodeId i = 0; i < 20; ++i)
        idx.remove(i);
    EXPECT_EQ(idx.size(), 30u);
}

// ---------------------------------------------------------------------------
// Search result ids are all valid inserted nodes
// ---------------------------------------------------------------------------

TEST(HnswCorrectness, SearchResultIdsAreValid) {
    const size_t N = 50;
    HnswIndex idx(test_cfg(8));
    std::unordered_set<NodeId> inserted;
    for (NodeId i = 0; i < N; ++i) {
        idx.insert(i, random_vec(8, i + 100).data());
        inserted.insert(i);
    }
    auto query = random_vec(8, 9876);
    auto result = idx.search(query.data(), 10, 30);
    for (auto& [d, id] : result) {
        EXPECT_TRUE(inserted.count(id)) << "returned unknown id: " << id;
    }
}

}  // namespace
}  // namespace vectordb
