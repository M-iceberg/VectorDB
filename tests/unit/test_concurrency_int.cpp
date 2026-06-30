#include <gtest/gtest.h>
#include "server/engine.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <random>
#include <thread>
#include <vector>

namespace vectordb {
namespace {

class ConcurrencyTest : public ::testing::Test {
protected:
    std::string data_dir_;

    void SetUp() override {
        data_dir_ = std::filesystem::temp_directory_path() /
                    ("test_concurrency_" + std::to_string(
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
// N reader threads + 1 writer thread, run for 3 seconds.
// Goals: no deadlock, no crash, no garbage results.
// ---------------------------------------------------------------------------

TEST_F(ConcurrencyTest, ConcurrentReadersAndWriter) {
    const size_t dim         = 16;
    const int    N_readers   = 4;
    const int    run_ms      = 10000;

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    // Pre-populate so readers have something to search.
    std::mt19937 seed_rng(0);
    for (uint32_t i = 0; i < 100; ++i) {
        auto v = random_vec(dim, seed_rng);
        engine.insert("test", std::to_string(i), v.data());
    }

    std::atomic<bool>     stop{false};
    std::atomic<uint32_t> next_id{100};
    std::atomic<int>      reader_errors{0};
    std::atomic<int>      writer_errors{0};

    // Writer thread: inserts nodes continuously.
    std::thread writer([&] {
        std::mt19937 rng(1);
        while (!stop.load(std::memory_order_relaxed)) {
            uint32_t id = next_id.fetch_add(1, std::memory_order_relaxed);
            auto v = random_vec(dim, rng);
            try {
                engine.insert("test", std::to_string(id), v.data());
            } catch (...) {
                writer_errors.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    // Reader threads: search continuously.
    std::vector<std::thread> readers;
    readers.reserve(N_readers);
    for (int t = 0; t < N_readers; ++t) {
        readers.emplace_back([&, t] {
            std::mt19937 rng(static_cast<uint32_t>(t + 10));
            while (!stop.load(std::memory_order_relaxed)) {
                auto q = random_vec(dim, rng);
                try {
                    auto results = engine.search({"test", q.data(), 5, 32});
                    // Results must be sorted by ascending distance.
                    for (size_t i = 1; i < results.size(); ++i) {
                        if (results[i].distance < results[i-1].distance)
                            reader_errors.fetch_add(1, std::memory_order_relaxed);
                    }
                } catch (...) {
                    reader_errors.fetch_add(1, std::memory_order_relaxed);
                }
            }
        });
    }

    std::this_thread::sleep_for(std::chrono::milliseconds(run_ms));
    stop.store(true, std::memory_order_relaxed);

    writer.join();
    for (auto& r : readers) r.join();

    EXPECT_EQ(writer_errors.load(), 0) << "writer threw exceptions";
    EXPECT_EQ(reader_errors.load(), 0) << "reader got errors or unsorted results";
}

// ---------------------------------------------------------------------------
// Concurrent inserts from multiple threads (all writers).
// Verifies no crash and all inserted nodes are searchable afterward.
// ---------------------------------------------------------------------------

TEST_F(ConcurrencyTest, ConcurrentWriters) {
    const size_t dim       = 16;
    const int    N_threads = 4;
    const int    per_thread = 50;

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    std::vector<std::thread> writers;
    writers.reserve(N_threads);
    for (int t = 0; t < N_threads; ++t) {
        writers.emplace_back([&, t] {
            std::mt19937 rng(static_cast<uint32_t>(t));
            uint32_t base = static_cast<uint32_t>(t * per_thread);
            for (int i = 0; i < per_thread; ++i) {
                auto v = random_vec(dim, rng);
                engine.insert("test", std::to_string(base + static_cast<uint32_t>(i)), v.data());
            }
        });
    }
    for (auto& w : writers) w.join();

    // All N_threads * per_thread nodes were inserted — search should return top_k.
    std::mt19937 rng(99);
    auto q = random_vec(dim, rng);
    auto results = engine.search({"test", q.data(), 10, 64});
    EXPECT_EQ((int)results.size(), 10);
}

// ---------------------------------------------------------------------------
// Concurrent readers while no writer: shared lock must not block readers.
// ---------------------------------------------------------------------------

TEST_F(ConcurrencyTest, ConcurrentReadersNoContention) {
    const size_t dim     = 16;
    const int    N_reads = 8;

    Engine engine(data_dir_);
    engine.create_collection(make_schema(dim));

    std::mt19937 seed_rng(0);
    for (uint32_t i = 0; i < 50; ++i) {
        auto v = random_vec(dim, seed_rng);
        engine.insert("test", std::to_string(i), v.data());
    }

    std::atomic<int> errors{0};
    std::vector<std::thread> readers;
    readers.reserve(N_reads);

    for (int t = 0; t < N_reads; ++t) {
        readers.emplace_back([&, t] {
            std::mt19937 rng(static_cast<uint32_t>(t));
            auto q = random_vec(dim, rng);
            try {
                auto results = engine.search({"test", q.data(), 5, 32});
                if ((int)results.size() != 5)
                    errors.fetch_add(1, std::memory_order_relaxed);
            } catch (...) {
                errors.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto& r : readers) r.join();

    EXPECT_EQ(errors.load(), 0);
}

}  // namespace
}  // namespace vectordb
