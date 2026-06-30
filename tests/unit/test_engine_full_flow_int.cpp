#include <gtest/gtest.h>
#include "server/engine.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <random>
#include <vector>

namespace vectordb {
namespace {

class EngineFullFlowTest : public ::testing::Test {
protected:
    std::string data_dir_;

    void SetUp() override {
        data_dir_ = std::filesystem::temp_directory_path() /
                    ("test_engine_full_" + std::to_string(
                        std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(data_dir_);
    }
    void TearDown() override {
        std::filesystem::remove_all(data_dir_);
    }

    static std::vector<float> random_vec(size_t dim, std::mt19937& rng) {
        std::uniform_real_distribution<float> d(-1.0f, 1.0f);
        std::vector<float> v(dim);
        for (auto& x : v) x = d(rng);
        return v;
    }
};

// ---------------------------------------------------------------------------
// Full CRUD flow: create → insert → search → delete → search
// ---------------------------------------------------------------------------

TEST_F(EngineFullFlowTest, CreateInsertSearchDeleteSearch) {
    const size_t dim = 32;
    const size_t N   = 200;
    std::mt19937 rng(1);

    Engine engine(data_dir_);

    // Create
    CollectionSchema schema;
    schema.name   = "items";
    schema.dim    = dim;
    schema.metric = Metric::L2;
    engine.create_collection(schema);

    // Insert
    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (size_t i = 0; i < N; ++i) {
        vecs[i] = random_vec(dim, rng);
        engine.insert("items", std::to_string(i), vecs[i].data());
    }

    // Search — query closest to node 0's vector
    auto results = engine.search({"items", vecs[0].data(), 5, 64});
    ASSERT_FALSE(results.empty());
    EXPECT_EQ(results[0].user_id, "0");  // exact match should be nearest

    // Delete half the nodes
    for (uint32_t i = 0; i < N / 2; ++i)
        engine.remove("items", std::to_string(i));

    // Search again — deleted nodes must not appear
    auto results2 = engine.search({"items", vecs[0].data(), 5, 64});
    for (auto& r : results2)
        EXPECT_GE(std::stoi(r.user_id), static_cast<int>(N / 2))
            << "deleted id " << r.user_id << " appeared in results";
}

// ---------------------------------------------------------------------------
// list_collections: returns correct schemas
// ---------------------------------------------------------------------------

TEST_F(EngineFullFlowTest, ListCollectionsReturnsAll) {
    Engine engine(data_dir_);

    CollectionSchema a; a.name = "alpha"; a.dim = 16; a.metric = Metric::L2;
    CollectionSchema b; b.name = "beta";  b.dim = 32; b.metric = Metric::Cosine;
    engine.create_collection(a);
    engine.create_collection(b);

    auto cols = engine.list_collections();
    ASSERT_EQ(cols.size(), 2u);

    // Order not guaranteed — build a name→dim map.
    std::unordered_map<std::string, size_t> found;
    for (auto& c : cols) found[c.name] = c.dim;

    EXPECT_EQ(found["alpha"], 16u);
    EXPECT_EQ(found["beta"],  32u);
}

// ---------------------------------------------------------------------------
// list_collections after drop
// ---------------------------------------------------------------------------

TEST_F(EngineFullFlowTest, ListAfterDrop) {
    Engine engine(data_dir_);

    CollectionSchema s; s.name = "tmp"; s.dim = 8; s.metric = Metric::L2;
    engine.create_collection(s);
    EXPECT_EQ(engine.list_collections().size(), 1u);

    engine.drop_collection("tmp");
    EXPECT_EQ(engine.list_collections().size(), 0u);
}

// ---------------------------------------------------------------------------
// list_collections survives restart
// ---------------------------------------------------------------------------

TEST_F(EngineFullFlowTest, ListSurvivesRestart) {
    {
        Engine engine(data_dir_);
        CollectionSchema a; a.name = "alpha"; a.dim = 16; a.metric = Metric::L2;
        CollectionSchema b; b.name = "beta";  b.dim = 32; b.metric = Metric::Cosine;
        engine.create_collection(a);
        engine.create_collection(b);
    }

    Engine engine(data_dir_);
    auto cols = engine.list_collections();
    ASSERT_EQ(cols.size(), 2u);

    std::unordered_map<std::string, size_t> found;
    for (auto& c : cols) found[c.name] = c.dim;
    EXPECT_EQ(found["alpha"], 16u);
    EXPECT_EQ(found["beta"],  32u);
}

// ---------------------------------------------------------------------------
// Full flow with metadata: insert with meta → filtered search → remove → verify
// ---------------------------------------------------------------------------

TEST_F(EngineFullFlowTest, FullFlowWithMetadata) {
    const size_t dim = 16;
    const size_t N   = 100;
    std::mt19937 rng(2);

    Engine engine(data_dir_);
    CollectionSchema schema; schema.name = "docs"; schema.dim = dim; schema.metric = Metric::L2;
    engine.create_collection(schema);

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (size_t i = 0; i < N; ++i) {
        vecs[i] = random_vec(dim, rng);
        MetadataEntry meta;
        meta.strings["type"] = (i % 2 == 0) ? "article" : "video";
        engine.insert("docs", std::to_string(i), vecs[i].data(), meta);
    }

    auto q = random_vec(dim, rng);

    // Filtered search: type == "article"
    SearchRequest req;
    req.collection = "docs";
    req.query      = q.data();
    req.top_k      = 5;
    req.ef_search  = 100;
    req.filters    = {{"type", FieldFilter::Op::Eq, "article", 0, 0}};
    auto results = engine.search(req);

    ASSERT_EQ((int)results.size(), 5);
    for (auto& r : results)
        EXPECT_EQ(std::stoi(r.user_id) % 2, 0) << "id " << r.user_id << " is not an article";

    // Remove all articles
    for (uint32_t i = 0; i < N; i += 2)
        engine.remove("docs", std::to_string(i));

    // Filtered search should now be empty
    auto results2 = engine.search(req);
    EXPECT_TRUE(results2.empty());
}

// ---------------------------------------------------------------------------
// Multiple collections are independent
// ---------------------------------------------------------------------------

TEST_F(EngineFullFlowTest, MultipleCollectionsIndependent) {
    const size_t dim = 8;
    std::mt19937 rng(3);

    Engine engine(data_dir_);
    CollectionSchema s1; s1.name = "c1"; s1.dim = dim; s1.metric = Metric::L2;
    CollectionSchema s2; s2.name = "c2"; s2.dim = dim; s2.metric = Metric::L2;
    engine.create_collection(s1);
    engine.create_collection(s2);

    auto v1 = random_vec(dim, rng);
    auto v2 = random_vec(dim, rng);
    engine.insert("c1", "1", v1.data());
    engine.insert("c2", "2", v2.data());

    // Searching c1 must not return c2's node and vice versa.
    auto r1 = engine.search({"c1", v1.data(), 1, 10});
    auto r2 = engine.search({"c2", v2.data(), 1, 10});

    ASSERT_EQ(r1.size(), 1u);
    ASSERT_EQ(r2.size(), 1u);
    EXPECT_EQ(r1[0].user_id, "1");
    EXPECT_EQ(r2[0].user_id, "2");
}

}  // namespace
}  // namespace vectordb
