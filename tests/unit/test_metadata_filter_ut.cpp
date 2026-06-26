#include <gtest/gtest.h>
#include "server/engine.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

namespace vectordb {
namespace {

class MetadataFilterTest : public ::testing::Test {
protected:
    std::string data_dir_;

    void SetUp() override {
        data_dir_ = std::filesystem::temp_directory_path() /
                    ("test_meta_filter_" + std::to_string(
                        std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(data_dir_);
    }
    void TearDown() override {
        std::filesystem::remove_all(data_dir_);
    }

    static CollectionSchema make_schema(size_t dim = 16) {
        CollectionSchema s;
        s.name   = "test";
        s.dim    = dim;
        s.metric = Metric::L2;
        return s;
    }

    static std::vector<float> random_vec(size_t dim, std::mt19937& rng) {
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<float> v(dim);
        for (auto& x : v) x = d(rng);
        return v;
    }

    // Brute-force filtered search: try all nodes, apply filter, sort by distance.
    static std::vector<uint32_t> brute_force(
        const std::vector<std::vector<float>>& vecs,
        const std::vector<float>& query,
        const std::function<bool(uint32_t)>& passes,
        int top_k)
    {
        size_t dim = query.size();
        std::vector<std::pair<float, uint32_t>> dists;
        for (size_t i = 0; i < vecs.size(); ++i) {
            if (!passes(static_cast<uint32_t>(i))) continue;
            float d = 0;
            for (size_t j = 0; j < dim; ++j) {
                float diff = query[j] - vecs[i][j];
                d += diff * diff;
            }
            dists.push_back({d, static_cast<uint32_t>(i)});
        }
        std::sort(dists.begin(), dists.end());
        std::vector<uint32_t> result;
        for (int k = 0; k < top_k && k < (int)dists.size(); ++k)
            result.push_back(dists[k].second);
        return result;
    }
};

// ---------------------------------------------------------------------------
// Filtered search with eq predicate — compare against brute force
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, EqFilterCorrectness) {
    const size_t dim = 16;
    const size_t N   = 500;
    const int    k   = 5;
    std::mt19937 rng(1);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (size_t i = 0; i < N; ++i) {
        vecs[i] = random_vec(dim, rng);
        MetadataEntry meta;
        meta.strings["color"] = (i % 3 == 0) ? "red" : (i % 3 == 1) ? "blue" : "green";
        engine.insert("test", static_cast<uint32_t>(i), vecs[i].data(), meta);
    }

    auto query = random_vec(dim, rng);

    // Engine filtered search: color == "red"
    SearchRequest req;
    req.collection = "test";
    req.query      = query.data();
    req.top_k      = k;
    req.ef_search  = 100;
    req.filters    = {{"color", FieldFilter::Op::Eq, "red", 0, 0}};
    auto results = engine.search(req);

    // Brute-force ground truth
    auto gt = brute_force(vecs, query,
        [](uint32_t i) { return i % 3 == 0; }, k);

    ASSERT_EQ((int)results.size(), k);
    // All returned IDs must pass the filter.
    for (auto& r : results)
        EXPECT_EQ(r.id % 3, 0u) << "id " << r.id << " does not have color=red";
    // Top result must match ground truth.
    EXPECT_EQ(results[0].id, gt[0]);
}

// ---------------------------------------------------------------------------
// Filtered search with range predicate — compare against brute force
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, RangeFilterCorrectness) {
    const size_t dim = 16;
    const size_t N   = 500;
    const int    k   = 5;
    std::mt19937 rng(2);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (size_t i = 0; i < N; ++i) {
        vecs[i] = random_vec(dim, rng);
        MetadataEntry meta;
        meta.numerics["score"] = static_cast<double>(i);
        engine.insert("test", static_cast<uint32_t>(i), vecs[i].data(), meta);
    }

    auto query = random_vec(dim, rng);

    // Filter: score in [100, 200]
    SearchRequest req;
    req.collection = "test";
    req.query      = query.data();
    req.top_k      = k;
    req.ef_search  = 100;
    req.filters    = {{"score", FieldFilter::Op::Range, "", 100.0, 200.0}};
    auto results = engine.search(req);

    auto gt = brute_force(vecs, query,
        [](uint32_t i) { return i >= 100 && i <= 200; }, k);

    ASSERT_EQ((int)results.size(), k);
    for (auto& r : results)
        EXPECT_TRUE(r.id >= 100 && r.id <= 200)
            << "id " << r.id << " outside range [100,200]";
    EXPECT_EQ(results[0].id, gt[0]);
}

// ---------------------------------------------------------------------------
// AND of two filters
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, AndFilterCorrectness) {
    const size_t dim = 16;
    const size_t N   = 400;
    const int    k   = 5;
    std::mt19937 rng(3);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (size_t i = 0; i < N; ++i) {
        vecs[i] = random_vec(dim, rng);
        MetadataEntry meta;
        meta.strings["type"]    = (i % 2 == 0) ? "A" : "B";
        meta.numerics["weight"] = static_cast<double>(i % 100);
        engine.insert("test", static_cast<uint32_t>(i), vecs[i].data(), meta);
    }

    auto query = random_vec(dim, rng);

    // Filter: type == "A" AND weight <= 50
    SearchRequest req;
    req.collection = "test";
    req.query      = query.data();
    req.top_k      = k;
    req.ef_search  = 100;
    req.filters    = {
        {"type",   FieldFilter::Op::Eq,    "A", 0, 0},
        {"weight", FieldFilter::Op::Range, "",  0, 50.0},
    };
    auto results = engine.search(req);

    for (auto& r : results) {
        EXPECT_EQ(r.id % 2, 0u)       << "id " << r.id << " type != A";
        EXPECT_LE(r.id % 100, 50u)    << "id " << r.id << " weight > 50";
    }
}

// ---------------------------------------------------------------------------
// No filter returns all top_k results (baseline, no regression)
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, NoFilterBaseline) {
    const size_t dim = 16;
    const size_t N   = 200;
    std::mt19937 rng(4);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    for (size_t i = 0; i < N; ++i) {
        auto v = random_vec(dim, rng);
        engine.insert("test", static_cast<uint32_t>(i), v.data());
    }

    auto q = random_vec(dim, rng);
    auto results = engine.search({"test", q.data(), 10, 50});
    EXPECT_EQ((int)results.size(), 10);
}

// ---------------------------------------------------------------------------
// Metadata survives checkpoint + recovery (metadata.bin restored correctly)
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, MetadataSurvivesCheckpointRecovery) {
    const size_t dim = 16;
    const size_t N   = 100;
    std::mt19937 rng(5);

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        for (size_t i = 0; i < N; ++i) {
            MetadataEntry meta;
            meta.strings["label"]  = (i % 2 == 0) ? "even" : "odd";
            meta.numerics["value"] = static_cast<double>(i);
            engine.insert("test", static_cast<uint32_t>(i), vecs[i].data(), meta);
        }
        engine.checkpoint("test");
        // "Crash" after checkpoint.
    }

    Engine engine(data_dir_);
    auto q = random_vec(dim, rng);

    // Filter: label == "even"
    SearchRequest req;
    req.collection = "test";
    req.query      = q.data();
    req.top_k      = 10;
    req.ef_search  = 100;
    req.filters    = {{"label", FieldFilter::Op::Eq, "even", 0, 0}};
    auto results = engine.search(req);

    ASSERT_EQ((int)results.size(), 10);
    for (auto& r : results)
        EXPECT_EQ(r.id % 2, 0u) << "id " << r.id << " is not even after recovery";
}

// ---------------------------------------------------------------------------
// Metadata in WAL survives recovery (no checkpoint — rebuilt from WAL)
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, MetadataSurvivesWalReplay) {
    const size_t dim = 16;
    const size_t N   = 100;
    std::mt19937 rng(6);

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        for (size_t i = 0; i < N; ++i) {
            MetadataEntry meta;
            meta.strings["label"] = (i % 2 == 0) ? "even" : "odd";
            engine.insert("test", static_cast<uint32_t>(i), vecs[i].data(), meta);
        }
        // No checkpoint — metadata must be recovered from WAL.
    }

    Engine engine(data_dir_);
    auto q = random_vec(dim, rng);

    SearchRequest req;
    req.collection = "test";
    req.query      = q.data();
    req.top_k      = 10;
    req.ef_search  = 100;
    req.filters    = {{"label", FieldFilter::Op::Eq, "odd", 0, 0}};
    auto results = engine.search(req);

    ASSERT_EQ((int)results.size(), 10);
    for (auto& r : results)
        EXPECT_EQ(r.id % 2, 1u) << "id " << r.id << " is not odd after WAL replay";
}

// ---------------------------------------------------------------------------
// Remove updates MetadataIndex: removed node must not appear in filtered search
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, RemoveUpdatesMetadata) {
    const size_t dim = 8;
    std::mt19937 rng(7);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    for (uint32_t i = 0; i < 50; ++i) {
        auto v = random_vec(dim, rng);
        MetadataEntry meta;
        meta.strings["tier"] = (i < 25) ? "gold" : "silver";
        engine.insert("test", i, v.data(), meta);
    }

    // Remove all gold nodes.
    for (uint32_t i = 0; i < 25; ++i)
        engine.remove("test", i);

    auto q = random_vec(dim, rng);
    SearchRequest req;
    req.collection = "test";
    req.query      = q.data();
    req.top_k      = 10;
    req.ef_search  = 100;
    req.filters    = {{"tier", FieldFilter::Op::Eq, "gold", 0, 0}};
    auto results = engine.search(req);

    // No gold nodes should appear — they were removed.
    EXPECT_TRUE(results.empty());
}

// ---------------------------------------------------------------------------
// Filter returning 0 candidates → empty search result
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, FilterWithNoCandidatesReturnsEmpty) {
    const size_t dim = 8;
    std::mt19937 rng(8);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    for (uint32_t i = 0; i < 50; ++i) {
        auto v = random_vec(dim, rng);
        MetadataEntry meta;
        meta.strings["color"] = "red";
        engine.insert("test", i, v.data(), meta);
    }

    auto q = random_vec(dim, rng);
    SearchRequest req;
    req.collection = "test";
    req.query      = q.data();
    req.top_k      = 10;
    req.ef_search  = 100;
    req.filters    = {{"color", FieldFilter::Op::Eq, "blue", 0, 0}};
    auto results = engine.search(req);

    EXPECT_TRUE(results.empty());
}

// ---------------------------------------------------------------------------
// Checkpoint + insert more + crash + recover: both metadata.bin (pre-checkpoint)
// and WAL (post-checkpoint) contribute to the recovered MetadataIndex.
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, CheckpointThenInsertThenRecover) {
    const size_t dim = 16;
    std::mt19937 rng(9);

    std::vector<std::vector<float>> vecs(150, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));

        // Insert 0..99, checkpoint (captured in metadata.bin).
        for (uint32_t i = 0; i < 100; ++i) {
            MetadataEntry meta;
            meta.strings["label"] = (i % 2 == 0) ? "even" : "odd";
            engine.insert("test", i, vecs[i].data(), meta);
        }
        engine.checkpoint("test");

        // Insert 100..149 (captured only in WAL).
        for (uint32_t i = 100; i < 150; ++i) {
            MetadataEntry meta;
            meta.strings["label"] = (i % 2 == 0) ? "even" : "odd";
            engine.insert("test", i, vecs[i].data(), meta);
        }
        // "Crash" — engine destroyed without another checkpoint.
    }

    Engine engine(data_dir_);
    auto q = random_vec(dim, rng);

    // Verify pre-checkpoint metadata (from metadata.bin).
    {
        SearchRequest req;
        req.collection = "test";
        req.query      = q.data();
        req.top_k      = 10;
        req.ef_search  = 200;
        req.filters    = {{"label", FieldFilter::Op::Eq, "even", 0, 0}};
        auto results = engine.search(req);
        ASSERT_EQ((int)results.size(), 10);
        for (auto& r : results)
            EXPECT_EQ(r.id % 2, 0u) << "pre-checkpoint: id " << r.id << " is not even";
    }

    // Verify post-checkpoint metadata (from WAL).
    {
        SearchRequest req;
        req.collection = "test";
        req.query      = q.data();
        req.top_k      = 5;
        req.ef_search  = 200;
        req.filters    = {{"label", FieldFilter::Op::Eq, "odd", 0, 0}};
        auto results = engine.search(req);
        ASSERT_EQ((int)results.size(), 5);
        for (auto& r : results)
            EXPECT_EQ(r.id % 2, 1u) << "post-checkpoint: id " << r.id << " is not odd";
    }
}

// ---------------------------------------------------------------------------
// Remove + checkpoint + recover: removed nodes must stay gone after restart.
// ---------------------------------------------------------------------------

TEST_F(MetadataFilterTest, RemovePersistsAcrossRecovery) {
    const size_t dim = 16;
    std::mt19937 rng(10);

    std::vector<std::vector<float>> vecs(50, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));

        for (uint32_t i = 0; i < 50; ++i) {
            MetadataEntry meta;
            meta.strings["tier"] = (i < 25) ? "gold" : "silver";
            engine.insert("test", i, vecs[i].data(), meta);
        }

        // Remove all gold nodes, then checkpoint.
        for (uint32_t i = 0; i < 25; ++i)
            engine.remove("test", i);
        engine.checkpoint("test");
    }

    Engine engine(data_dir_);
    auto q = random_vec(dim, rng);

    SearchRequest req;
    req.collection = "test";
    req.query      = q.data();
    req.top_k      = 10;
    req.ef_search  = 100;
    req.filters    = {{"tier", FieldFilter::Op::Eq, "gold", 0, 0}};
    auto results = engine.search(req);

    EXPECT_TRUE(results.empty()) << "gold nodes should be gone after remove+checkpoint+recovery";
}

}  // namespace
}  // namespace vectordb
