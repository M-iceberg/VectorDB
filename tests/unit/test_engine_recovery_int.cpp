#include <gtest/gtest.h>
#include "server/engine.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fcntl.h>
#include <random>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

namespace vectordb {
namespace {

class ScopedStorageMode {
public:
    explicit ScopedStorageMode(const char* mode) {
        if (const char* current = std::getenv("VECTORDB_STORAGE_MODE")) {
            had_previous_ = true;
            previous_ = current;
        }
        ::setenv("VECTORDB_STORAGE_MODE", mode, 1);
    }
    ~ScopedStorageMode() {
        if (had_previous_)
            ::setenv("VECTORDB_STORAGE_MODE", previous_.c_str(), 1);
        else
            ::unsetenv("VECTORDB_STORAGE_MODE");
    }
private:
    bool had_previous_ = false;
    std::string previous_;
};

class EngineRecoveryTest : public ::testing::Test {
protected:
    std::string data_dir_;

    void SetUp() override {
        data_dir_ = std::filesystem::temp_directory_path() /
                    ("test_engine_" + std::to_string(
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
};

// ---------------------------------------------------------------------------
// Basic: insert + search in a single session
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, BasicInsertAndSearch) {
    const size_t dim = 16;
    std::mt19937 rng(1);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    for (uint32_t i = 0; i < 100; ++i) {
        auto v = random_vec(dim, rng);
        engine.insert("test", std::to_string(i), v.data());
    }

    auto q = random_vec(dim, rng);
    SearchRequest req{"test", q.data(), 10, 50};
    auto results = engine.search(req);

    EXPECT_EQ(static_cast<int>(results.size()), 10);
    // Results must be sorted ascending by distance.
    for (size_t i = 1; i < results.size(); ++i)
        EXPECT_LE(results[i - 1].distance, results[i].distance);
}

// ---------------------------------------------------------------------------
// WAL replay: reopen without checkpoint; all inserts recovered via WAL
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, WalReplayOnReopen) {
    const size_t dim = 16;
    const size_t N   = 100;
    std::mt19937 rng(2);

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        for (uint32_t i = 0; i < N; ++i)
            engine.insert("test", std::to_string(i), vecs[i].data());
        // Engine destroyed here — WAL on disk, no checkpoint.
    }

    Engine engine(data_dir_);  // recover via WAL replay
    EXPECT_EQ(engine.search({"test", vecs[0].data(), 1, 10}).size(), 1u);

    // All N nodes must be reachable.
    auto q = random_vec(dim, rng);
    auto results = engine.search({"test", q.data(), static_cast<int>(N), 100});
    EXPECT_EQ(results.size(), N);
}

// ---------------------------------------------------------------------------
// Checkpoint + recovery: insert → checkpoint → insert more → reopen
// All N + M vectors must be present after recovery.
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, CheckpointAndRecovery) {
    const size_t dim = 16;
    const size_t N   = 100;   // before checkpoint
    const size_t M   = 50;    // after checkpoint
    std::mt19937 rng(3);

    std::vector<std::vector<float>> vecs(N + M, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));

        for (uint32_t i = 0; i < N; ++i)
            engine.insert("test", std::to_string(i), vecs[i].data());

        engine.checkpoint("test");   // snapshot N vectors, truncate WAL

        for (uint32_t i = N; i < N + M; ++i)
            engine.insert("test", std::to_string(i), vecs[i].data());

        // "Crash": engine destroyed without another checkpoint.
        // On disk: graph.bin (N vectors) + WAL (M records)
    }

    Engine engine(data_dir_);  // recover: load graph.bin + replay M WAL records

    auto q = random_vec(dim, rng);
    auto results = engine.search({"test", q.data(), static_cast<int>(N + M), 200});
    EXPECT_EQ(results.size(), N + M);
}

// ---------------------------------------------------------------------------
// Remove is crash-safe: remove after checkpoint persists through reopen
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, RemovePersistedThroughReopen) {
    const size_t dim = 8;
    std::mt19937 rng(4);

    std::vector<std::vector<float>> vecs(50, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        for (uint32_t i = 0; i < 50; ++i)
            engine.insert("test", std::to_string(i), vecs[i].data());

        engine.checkpoint("test");

        // Remove half the nodes after checkpoint (WAL records).
        for (uint32_t i = 0; i < 50; i += 2)
            engine.remove("test", std::to_string(i));
    }

    Engine engine(data_dir_);

    // Search near node 0's vector — it was deleted, must not appear in results.
    auto results = engine.search({"test", vecs[0].data(), 10, 50});
    for (auto& r : results)
        EXPECT_NE(std::stoi(r.user_id) % 2, 0) << "deleted node " << r.user_id << " appeared in results";
}

// ---------------------------------------------------------------------------
// Recovery idempotency: replaying WAL on an already-populated index
// must not duplicate nodes or corrupt live_count.
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, ReplayIdempotency) {
    const size_t dim = 8;
    const size_t N   = 60;
    std::mt19937 rng(5);

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        for (uint32_t i = 0; i < N; ++i)
            engine.insert("test", std::to_string(i), vecs[i].data());
    }

    // First recovery.
    Engine engine1(data_dir_);
    auto r1 = engine1.search({"test", vecs[0].data(), static_cast<int>(N), 100});
    EXPECT_EQ(r1.size(), N);
    // Destroy and recover again (second recovery replays the same WAL).

    Engine engine2(data_dir_);
    auto r2 = engine2.search({"test", vecs[0].data(), static_cast<int>(N), 100});
    // Must still return exactly N results — not 2N from double-insert.
    EXPECT_EQ(r2.size(), N);
}

// ---------------------------------------------------------------------------
// Multiple checkpoints: WAL truncated after each; recovery uses latest.
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, MultipleCheckpoints) {
    const size_t dim = 8;
    std::mt19937 rng(6);

    auto vec = [&]{ return random_vec(dim, rng); };

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));

        for (uint32_t i = 0; i < 30; ++i) { auto v = vec(); engine.insert("test", std::to_string(i), v.data()); }
        engine.checkpoint("test");  // checkpoint 1

        for (uint32_t i = 30; i < 60; ++i) { auto v = vec(); engine.insert("test", std::to_string(i), v.data()); }
        engine.checkpoint("test");  // checkpoint 2

        for (uint32_t i = 60; i < 80; ++i) { auto v = vec(); engine.insert("test", std::to_string(i), v.data()); }
        // Crash here: 80 vectors, checkpoint at 60, WAL has 20 records.
    }

    Engine engine(data_dir_);
    auto results = engine.search({"test", random_vec(dim, rng).data(), 80, 200});
    EXPECT_EQ(results.size(), 80u);
}

// ---------------------------------------------------------------------------
// Tombstone + re-insert + recover: a node removed after checkpoint then
// re-inserted in the WAL must be live (not tombstoned) after recovery.
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, TombstoneReinsertRecovery) {
    const size_t dim = 8;
    std::mt19937 rng(7);

    auto v0 = random_vec(dim, rng);
    auto v0b = random_vec(dim, rng);  // different vector, same id

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));

        // Insert node 0, checkpoint, remove it, then re-insert with new vec.
        engine.insert("test", "0", v0.data());
        engine.checkpoint("test");
        engine.remove("test", "0");
        engine.insert("test", "0", v0b.data());
        // Crash: graph.bin has node 0 live, WAL has Delete(0) + Insert(0).
    }

    Engine engine(data_dir_);
    // After recovery node 0 must be live with the new vector.
    auto results = engine.search({"test", v0b.data(), 1, 10});
    ASSERT_EQ(results.size(), 1u);
    EXPECT_EQ(results[0].user_id, "0");
}

// ---------------------------------------------------------------------------
// create_collection with duplicate name throws
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, DuplicateCollectionThrows) {
    Engine engine(data_dir_);
    engine.create_collection(make_schema());
    EXPECT_THROW(engine.create_collection(make_schema()), std::runtime_error);
}

// ---------------------------------------------------------------------------
// drop_collection + recreate: new collection is empty
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, DropAndRecreate) {
    const size_t dim = 8;
    std::mt19937 rng(8);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    for (uint32_t i = 0; i < 20; ++i) {
        auto v = random_vec(dim, rng);
        engine.insert("test", std::to_string(i), v.data());
    }

    engine.drop_collection("test");
    engine.create_collection(make_schema(dim));

    // After recreate the collection must be empty.
    auto q = random_vec(dim, rng);
    auto results = engine.search({"test", q.data(), 10, 50});
    EXPECT_TRUE(results.empty());
}

// ---------------------------------------------------------------------------
// search on empty collection returns empty results (no crash)
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, SearchEmptyCollection) {
    Engine engine(data_dir_);
    engine.create_collection(make_schema());

    std::vector<float> query(16, 0.0f);
    auto results = engine.search({"test", query.data(), 10, 50});
    EXPECT_TRUE(results.empty());
}

// ---------------------------------------------------------------------------
// True crash simulation: fork a child that inserts + checkpoints + inserts,
// then SIGKILL it without letting destructors run. Parent recovers and
// verifies all data is present.
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, CrashRecoveryViaKill) {
    const size_t dim = 16;
    const size_t N   = 100;   // before checkpoint
    const size_t M   = 50;    // after checkpoint
    std::mt19937 rng(42);

    std::vector<std::vector<float>> vecs(N + M, std::vector<float>(dim));
    for (auto& v : vecs) v = random_vec(dim, rng);

    // Pipe for child → parent "all inserts done" signal.
    int pipefd[2];
    ASSERT_EQ(::pipe(pipefd), 0);

    pid_t pid = ::fork();
    ASSERT_GE(pid, 0);

    if (pid == 0) {
        // ---- child process ----
        ::close(pipefd[0]);

        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        for (uint32_t i = 0; i < N; ++i)
            engine.insert("test", std::to_string(i), vecs[i].data());
        engine.checkpoint("test");
        for (uint32_t i = N; i < N + M; ++i)
            engine.insert("test", std::to_string(i), vecs[i].data());

        // Every insert called wal.sync() → fsync(), so all data is on disk.
        // Signal parent and wait to be killed.
        char ready = 1;
        ::write(pipefd[1], &ready, 1);
        ::close(pipefd[1]);
        ::pause();       // wait for SIGKILL — destructors never run
        ::_exit(0);
    }

    // ---- parent process ----
    ::close(pipefd[1]);

    // Wait until child has finished all inserts and synced to disk.
    char ready = 0;
    ASSERT_EQ(::read(pipefd[0], &ready, 1), 1);
    ::close(pipefd[0]);

    // Kill the child — simulates SIGKILL crash, no destructors run.
    ASSERT_EQ(::kill(pid, SIGKILL), 0);
    ::waitpid(pid, nullptr, 0);

    // Recover: graph.bin (N vectors) + WAL (M records) → must see all N+M.
    Engine recovered(data_dir_);
    auto q = random_vec(dim, rng);
    auto results = recovered.search({"test", q.data(), static_cast<int>(N + M), 200});
    EXPECT_EQ(results.size(), N + M);
}

// Crash at every checkpoint publication boundary. The previous manifest plus
// WAL, or the newly published manifest, must always describe a recoverable
// state; no destructor is allowed to run in the child.
TEST_F(EngineRecoveryTest, AtomicCheckpointSurvivesEveryPublishPhase) {
    constexpr size_t dim = 16;
    constexpr size_t N = 60;
    constexpr size_t M = 30;
    const std::vector<std::string> phases = {
        "after_snapshot_write",
        "after_snapshot_publish",
        "after_manifest_publish",
        "after_wal_truncate",
    };

    std::mt19937 rng(4242);
    std::vector<std::vector<float>> vecs(N + M, std::vector<float>(dim));
    for (auto& vec : vecs) vec = random_vec(dim, rng);

    for (const auto& phase : phases) {
        std::string case_dir = data_dir_ + "/" + phase;
        std::filesystem::create_directories(case_dir);
        {
            Engine engine(case_dir);
            engine.create_collection(make_schema(dim));
            for (uint32_t i = 0; i < N; ++i)
                engine.insert("test", std::to_string(i), vecs[i].data());
            engine.checkpoint("test");
            for (uint32_t i = N; i < N + M; ++i)
                engine.insert("test", std::to_string(i), vecs[i].data());
        }

        pid_t pid = ::fork();
        ASSERT_GE(pid, 0);
        if (pid == 0) {
            ::setenv("VECTORDB_CHECKPOINT_FAILPOINT", phase.c_str(), 1);
            Engine engine(case_dir);
            engine.checkpoint("test");
            ::_exit(0);  // failpoint should have SIGKILLed us first
        }

        int status = 0;
        ASSERT_EQ(::waitpid(pid, &status, 0), pid);
        ASSERT_TRUE(WIFSIGNALED(status)) << "phase=" << phase;
        EXPECT_EQ(WTERMSIG(status), SIGKILL) << "phase=" << phase;

        Engine recovered(case_dir);
        auto results = recovered.search(
            {"test", vecs[0].data(), static_cast<int>(N + M), 200});
        EXPECT_EQ(results.size(), N + M) << "phase=" << phase;
    }
}

TEST_F(EngineRecoveryTest, CorruptPublishedSnapshotIsRejected) {
    constexpr size_t dim = 8;
    std::mt19937 rng(5150);
    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        for (uint32_t i = 0; i < 20; ++i) {
            auto vec = random_vec(dim, rng);
            engine.insert("test", std::to_string(i), vec.data());
        }
        engine.checkpoint("test");
    }

    std::filesystem::path graph_path;
    for (const auto& entry : std::filesystem::directory_iterator(data_dir_ + "/test")) {
        std::string name = entry.path().filename().string();
        if (entry.is_directory() && name.rfind("checkpoint-", 0) == 0) {
            graph_path = entry.path() / "graph.bin";
            break;
        }
    }
    ASSERT_FALSE(graph_path.empty());

    int fd = ::open(graph_path.c_str(), O_RDWR);
    ASSERT_GE(fd, 0);
    uint8_t byte = 0;
    ASSERT_EQ(::pread(fd, &byte, 1, 40), 1);
    byte ^= 0xFFu;
    ASSERT_EQ(::pwrite(fd, &byte, 1, 40), 1);
    ASSERT_EQ(::fsync(fd), 0);
    ::close(fd);

    EXPECT_THROW({ Engine recovered(data_dir_); }, std::runtime_error);
}

TEST_F(EngineRecoveryTest, WalCommitPrecedesBatchGraphMutation) {
    constexpr size_t dim = 16;
    constexpr size_t count = 80;
    std::mt19937 rng(6161);
    std::vector<float> vecs(count * dim);
    std::vector<std::string> ids(count);
    for (uint32_t i = 0; i < count; ++i) {
        auto vec = random_vec(dim, rng);
        std::copy(vec.begin(), vec.end(), vecs.begin() + i * dim);
        ids[i] = std::to_string(i);
    }
    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
    }

    pid_t pid = ::fork();
    ASSERT_GE(pid, 0);
    if (pid == 0) {
        ::setenv("VECTORDB_INSERT_FAILPOINT", "after_wal_sync", 1);
        Engine engine(data_dir_);
        engine.insert_batch("test", ids, vecs.data(), count, {}, 4);
        ::_exit(0);
    }

    int status = 0;
    ASSERT_EQ(::waitpid(pid, &status, 0), pid);
    ASSERT_TRUE(WIFSIGNALED(status));
    EXPECT_EQ(WTERMSIG(status), SIGKILL);

    Engine recovered(data_dir_);
    auto results = recovered.search(
        {"test", vecs.data(), static_cast<int>(count), 200});
    EXPECT_EQ(results.size(), count);
}

// ---------------------------------------------------------------------------
// Search with top_k > number of nodes returns what's available, not top_k.
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, SearchTopKExceedsNodeCount) {
    const size_t dim = 16;
    const size_t N   = 3;
    std::mt19937 rng(99);

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    for (size_t i = 0; i < N; ++i) {
        auto v = random_vec(dim, rng);
        engine.insert("test", std::to_string(i), v.data());
    }

    std::vector<float> q = random_vec(dim, rng);
    auto results = engine.search({"test", q.data(), 10, 50});
    EXPECT_EQ(results.size(), N);  // only 3 nodes exist, must return 3 not 10
}

TEST_F(EngineRecoveryTest, CompactStorageSurvivesCheckpointAndReopen) {
    ScopedStorageMode compact("compact");
    constexpr size_t dim = 16;
    constexpr size_t count = 256;
    std::mt19937 rng(9090);
    std::vector<float> vecs(count * dim);
    std::vector<std::string> ids(count);
    for (size_t i = 0; i < count; ++i) {
        auto vec = random_vec(dim, rng);
        std::copy(vec.begin(), vec.end(), vecs.begin() + i * dim);
        ids[i] = "compact-" + std::to_string(i);
    }

    {
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        engine.insert_batch("test", ids, vecs.data(), count, {}, 4);
        auto result = engine.search(
            {"test", vecs.data() + 77 * dim, 1, 100});
        ASSERT_EQ(result.size(), 1u);
        EXPECT_EQ(result[0].user_id, ids[77]);
        engine.checkpoint("test");
    }

    Engine reopened(data_dir_);
    auto result = reopened.search(
        {"test", vecs.data() + 177 * dim, 1, 100});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].user_id, ids[177]);
}

TEST_F(EngineRecoveryTest, CheckpointCanSwitchStorageModesOnReopen) {
    constexpr size_t dim = 16;
    std::mt19937 rng(9191);
    auto vec = random_vec(dim, rng);

    {
        ScopedStorageMode performance("performance");
        Engine engine(data_dir_);
        engine.create_collection(make_schema(dim));
        engine.insert("test", "mode-switch", vec.data());
        engine.checkpoint("test");
    }
    {
        ScopedStorageMode compact("compact");
        Engine engine(data_dir_);
        auto result = engine.search({"test", vec.data(), 1, 50});
        ASSERT_EQ(result.size(), 1u);
        EXPECT_EQ(result[0].user_id, "mode-switch");
    }
    {
        ScopedStorageMode performance("performance");
        Engine engine(data_dir_);
        auto result = engine.search({"test", vec.data(), 1, 50});
        ASSERT_EQ(result.size(), 1u);
        EXPECT_EQ(result[0].user_id, "mode-switch");
    }
}

// ---------------------------------------------------------------------------
// Error propagation: insert/search/remove/checkpoint on unknown collection.
// ---------------------------------------------------------------------------

TEST_F(EngineRecoveryTest, UnknownCollectionThrows) {
    Engine engine(data_dir_);
    engine.create_collection(make_schema());

    std::vector<float> v(16, 0.0f);

    EXPECT_THROW(engine.insert("no_such", "1", v.data()), std::runtime_error);
    EXPECT_THROW(engine.remove("no_such", "1"),            std::runtime_error);
    EXPECT_THROW(engine.search({"no_such", v.data(), 1, 10}), std::runtime_error);
    EXPECT_THROW(engine.checkpoint("no_such"),           std::runtime_error);
    EXPECT_THROW(engine.drop_collection("no_such"),      std::runtime_error);
}

}  // namespace
}  // namespace vectordb
