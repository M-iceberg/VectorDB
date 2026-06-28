#include <gtest/gtest.h>
#include "server/engine.h"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <random>
#include <signal.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <unordered_set>
#include <vector>

namespace vectordb {
namespace {

class E2ETest : public ::testing::Test {
protected:
    std::string data_dir_;

    void SetUp() override {
        data_dir_ = std::filesystem::temp_directory_path() /
                    ("test_e2e_" + std::to_string(
                        std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(data_dir_);
    }
    void TearDown() override {
        std::filesystem::remove_all(data_dir_);
    }

    static CollectionSchema make_schema(size_t dim, const std::string& name = "test") {
        CollectionSchema s;
        s.name   = name;
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

    static uint32_t brute_force_nn(
        const std::vector<std::vector<float>>& vecs,
        const std::unordered_set<uint32_t>& live,
        const std::vector<float>& query)
    {
        size_t dim = query.size();
        float best = std::numeric_limits<float>::max();
        uint32_t best_id = 0;
        for (uint32_t id : live) {
            float d = 0;
            for (size_t j = 0; j < dim; ++j) {
                float diff = query[j] - vecs[id][j];
                d += diff * diff;
            }
            if (d < best) { best = d; best_id = id; }
        }
        return best_id;
    }
};

// ---------------------------------------------------------------------------
// insert 10K → search (recall@1 >= 80%) → delete 20% → search again
// (deleted nodes absent, recall@1 still >= 70%)
// ---------------------------------------------------------------------------

TEST_F(E2ETest, Insert10KSearchDeleteSearch) {
    const size_t dim   = 128;
    const size_t N     = 10000;
    const size_t N_del = 2000;
    const int    top_k = 10;
    const int    Q     = 20;
    const int    ef    = 200;   // match ef_construction for good recall
    std::mt19937 rng(42);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (size_t i = 0; i < N; ++i) {
        vecs[i] = random_vec(dim, rng);
        engine.insert("test", static_cast<uint32_t>(i), vecs[i].data());
    }

    std::unordered_set<uint32_t> live;
    for (uint32_t i = 0; i < N; ++i) live.insert(i);

    // Search before delete.
    int hits_before = 0;
    for (int q = 0; q < Q; ++q) {
        auto query = random_vec(dim, rng);
        auto results = engine.search({"test", query.data(), top_k, ef});
        ASSERT_FALSE(results.empty());
        uint32_t gt = brute_force_nn(vecs, live, query);
        for (auto& r : results)
            if (r.id == gt) { ++hits_before; break; }
    }
    EXPECT_GE(hits_before, Q * 8 / 10) << "recall@1 before delete < 80%";

    // Delete nodes 0..N_del-1.
    for (uint32_t i = 0; i < N_del; ++i) {
        engine.remove("test", i);
        live.erase(i);
    }

    // Search after delete.
    int hits_after = 0;
    for (int q = 0; q < Q; ++q) {
        auto query = random_vec(dim, rng);
        auto results = engine.search({"test", query.data(), top_k, ef});
        for (auto& r : results)
            EXPECT_GE(r.id, static_cast<uint32_t>(N_del))
                << "deleted id " << r.id << " appeared in results";
        uint32_t gt = brute_force_nn(vecs, live, query);
        for (auto& r : results)
            if (r.id == gt) { ++hits_after; break; }
    }
    EXPECT_GE(hits_after, Q * 7 / 10) << "recall@1 after delete < 70%";
}

// ---------------------------------------------------------------------------
// Crash recovery at scale: insert N + checkpoint + insert M → SIGKILL →
// restart → all N+M nodes present.
//
// Uses a flag file to synchronize: child writes the flag after checkpoint so
// parent knows exactly when to kill (after M post-checkpoint inserts).
// ---------------------------------------------------------------------------

TEST_F(E2ETest, ScaleCrashRecovery) {
    const size_t dim  = 32;
    const size_t N    = 200;   // pre-checkpoint
    const size_t M    = 100;   // post-checkpoint
    std::mt19937 rng(7);

    std::vector<std::vector<float>> vecs(N + M, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    std::string ckpt_flag = data_dir_ + "/checkpoint_done";
    std::string done_flag = data_dir_ + "/inserts_done";

    pid_t pid = ::fork();
    ASSERT_GE(pid, 0) << "fork failed";

    if (pid == 0) {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        for (size_t i = 0; i < N; ++i)
            engine.insert("test", static_cast<uint32_t>(i), vecs[i].data());
        engine.checkpoint("test");
        std::ofstream(ckpt_flag).close();
        for (size_t i = N; i < N + M; ++i)
            engine.insert("test", static_cast<uint32_t>(i), vecs[i].data());
        // Signal that all post-checkpoint inserts are WAL-synced.
        std::ofstream(done_flag).close();
        ::pause();
        _exit(0);
    }

    // Wait for all inserts to complete (each engine.insert() syncs the WAL
    // before returning, so done_flag means all M records are on disk).
    for (int i = 0; i < 300 && !std::filesystem::exists(done_flag); ++i)
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    ASSERT_TRUE(std::filesystem::exists(done_flag))
        << "post-checkpoint inserts timed out (30s)";
    ::kill(pid, SIGKILL);
    int status;
    ::waitpid(pid, &status, 0);

    // Every insert() that returned successfully is WAL-synced, so all N+M
    // nodes must survive the crash.
    Engine engine(data_dir_);
    auto all = engine.search({"test", vecs[0].data(),
                               static_cast<int>(N + M), static_cast<int>(N + M)});
    EXPECT_EQ(all.size(), N + M) << "not all " << N + M << " nodes recovered";
}

}  // namespace
}  // namespace vectordb
