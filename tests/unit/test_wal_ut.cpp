// -----------------------------------------------------------------------------
// test_wal_ut.cpp — Unit tests for WAL writer
//
// Tests:
//   WalBasic:   append records, verify LSN assignment, iterate reads them back
//   WalCrc:     CRC covers payload — corruption detected on iterate
//   WalReopen:  reopen after close, LSN continues from where it left off
//   WalSync:    sync() doesn't throw
//   WalCorrupt: truncated tail record is silently skipped on reopen
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "storage/wal.h"
#include "storage/wal_record.h"
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>
#include <cmath>

namespace vectordb {
namespace {

// Creates a temp WAL path that is deleted after each test.
class WalTest : public ::testing::Test {
protected:
    std::string path_;
    void SetUp() override {
        path_ = std::filesystem::temp_directory_path() / "test_wal_XXXXXX";
        // Use a unique name based on test info.
        path_ += std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    void TearDown() override {
        std::filesystem::remove(path_);
        std::filesystem::remove(path_ + ".base");
        std::filesystem::remove(path_ + ".tmp");
    }
};

// ---------------------------------------------------------------------------
// Basic: append records and iterate them back
// ---------------------------------------------------------------------------

TEST_F(WalTest, AppendAndIterate) {
    {
        Wal wal(path_);
        EXPECT_EQ(wal.current_lsn(), 0u);

        uint32_t payload0 = 42;
        Lsn lsn0 = wal.append(WalRecordType::Insert, &payload0, sizeof(payload0));
        EXPECT_EQ(lsn0, 0u);
        EXPECT_EQ(wal.current_lsn(), 1u);

        uint32_t payload1 = 99;
        Lsn lsn1 = wal.append(WalRecordType::Delete, &payload1, sizeof(payload1));
        EXPECT_EQ(lsn1, 1u);
        EXPECT_EQ(wal.current_lsn(), 2u);

        wal.sync();
    }

    // Iterate and verify.
    Wal wal(path_);
    std::vector<std::pair<Lsn, uint32_t>> records;
    wal.iterate(0, [&](Lsn lsn, WalRecordType type, const void* payload, uint32_t len) {
        EXPECT_EQ(len, sizeof(uint32_t));
        uint32_t v;
        std::memcpy(&v, payload, sizeof(v));
        records.push_back({lsn, v});
        (void)type;
    });

    ASSERT_EQ(records.size(), 2u);
    EXPECT_EQ(records[0].first, 0u);
    EXPECT_EQ(records[0].second, 42u);
    EXPECT_EQ(records[1].first, 1u);
    EXPECT_EQ(records[1].second, 99u);
}

// iterate with start_lsn skips earlier records.
TEST_F(WalTest, IterateStartLsn) {
    {
        Wal wal(path_);
        for (uint32_t i = 0; i < 5; ++i)
            wal.append(WalRecordType::Insert, &i, sizeof(i));
        wal.sync();
    }

    Wal wal(path_);
    std::vector<Lsn> seen;
    wal.iterate(3, [&](Lsn lsn, WalRecordType, const void*, uint32_t) {
        seen.push_back(lsn);
    });
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0], 3u);
    EXPECT_EQ(seen[1], 4u);
}

// ---------------------------------------------------------------------------
// LSN continues after reopen
// ---------------------------------------------------------------------------

TEST_F(WalTest, LsnContinuesAfterReopen) {
    {
        Wal wal(path_);
        uint32_t v = 1;
        wal.append(WalRecordType::Insert, &v, sizeof(v));
        wal.append(WalRecordType::Insert, &v, sizeof(v));
        wal.sync();
        EXPECT_EQ(wal.current_lsn(), 2u);
    }

    Wal wal(path_);
    EXPECT_EQ(wal.current_lsn(), 2u);  // picked up from file scan
    uint32_t v = 3;
    Lsn lsn = wal.append(WalRecordType::Insert, &v, sizeof(v));
    EXPECT_EQ(lsn, 2u);
    EXPECT_EQ(wal.current_lsn(), 3u);
}

// ---------------------------------------------------------------------------
// Corrupt tail record is silently skipped on reopen
// ---------------------------------------------------------------------------

TEST_F(WalTest, CorruptTailTruncatedOnReopen) {
    {
        Wal wal(path_);
        uint32_t v = 7;
        wal.append(WalRecordType::Insert, &v, sizeof(v));
        wal.sync();
    }

    // Corrupt the last few bytes of the file (simulate partial write crash).
    {
        std::fstream f(path_, std::ios::in | std::ios::out | std::ios::binary);
        f.seekp(-2, std::ios::end);
        uint8_t garbage[2] = {0xFF, 0xFF};
        f.write(reinterpret_cast<char*>(garbage), 2);
    }

    // Reopen — should silently discard the corrupt record.
    Wal wal(path_);
    EXPECT_EQ(wal.current_lsn(), 0u);  // corrupt record was discarded

    // Can still append new records.
    uint32_t v = 99;
    Lsn lsn = wal.append(WalRecordType::Insert, &v, sizeof(v));
    EXPECT_EQ(lsn, 0u);
}

// ---------------------------------------------------------------------------
// sync() doesn't throw
// ---------------------------------------------------------------------------

TEST_F(WalTest, SyncDoesNotThrow) {
    Wal wal(path_);
    uint32_t v = 1;
    wal.append(WalRecordType::Insert, &v, sizeof(v));
    EXPECT_NO_THROW(wal.sync());
}

// ---------------------------------------------------------------------------
// Empty file: iterate on empty WAL returns nothing
// ---------------------------------------------------------------------------

TEST_F(WalTest, IterateEmptyWal) {
    Wal wal(path_);
    int count = 0;
    wal.iterate(0, [&](Lsn, WalRecordType, const void*, uint32_t) { ++count; });
    EXPECT_EQ(count, 0);
}

// ---------------------------------------------------------------------------
// Large payload round-trips correctly
// ---------------------------------------------------------------------------

TEST_F(WalTest, LargePayloadRoundTrip) {
    std::vector<float> vec(128);
    for (int i = 0; i < 128; ++i) vec[i] = static_cast<float>(i) * 0.5f;

    {
        Wal wal(path_);
        wal.append(WalRecordType::Insert, vec.data(), vec.size() * sizeof(float));
        wal.sync();
    }

    Wal wal(path_);
    std::vector<float> recovered;
    wal.iterate(0, [&](Lsn, WalRecordType, const void* payload, uint32_t len) {
        recovered.resize(len / sizeof(float));
        std::memcpy(recovered.data(), payload, len);
    });

    ASSERT_EQ(recovered.size(), vec.size());
    for (size_t i = 0; i < vec.size(); ++i)
        EXPECT_FLOAT_EQ(recovered[i], vec[i]);
}

// ---------------------------------------------------------------------------
// replay(): insert and delete records are dispatched to the right callbacks
// ---------------------------------------------------------------------------

TEST_F(WalTest, ReplayInsertAndDelete) {
    const size_t dim = 4;
    std::vector<float> vec0 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> vec1 = {5.0f, 6.0f, 7.0f, 8.0f};

    {
        Wal wal(path_);
        auto p0 = make_insert_payload(10, "10", vec0.data(), dim);
        auto p1 = make_insert_payload(20, "20", vec1.data(), dim);
        auto pd = make_delete_payload(10);
        wal.append(WalRecordType::Insert, p0.data(), p0.size());
        wal.append(WalRecordType::Insert, p1.data(), p1.size());
        wal.append(WalRecordType::Delete, pd.data(), pd.size());
        wal.sync();
    }

    Wal wal(path_);
    std::vector<std::pair<uint32_t, std::vector<float>>> inserts;
    std::vector<uint32_t> deletes;

    wal.replay(0, dim,
        [&](uint32_t id, const std::string& /*user_id*/, const float* vec, size_t d, const MetadataEntry&) {
            inserts.push_back({id, std::vector<float>(vec, vec + d)});
        },
        [&](uint32_t id) {
            deletes.push_back(id);
        });

    ASSERT_EQ(inserts.size(), 2u);
    EXPECT_EQ(inserts[0].first, 10u);
    EXPECT_EQ(inserts[0].second, vec0);
    EXPECT_EQ(inserts[1].first, 20u);
    EXPECT_EQ(inserts[1].second, vec1);

    ASSERT_EQ(deletes.size(), 1u);
    EXPECT_EQ(deletes[0], 10u);
}

// ---------------------------------------------------------------------------
// truncate_before(): removes records before lsn, preserves LSN numbering
// ---------------------------------------------------------------------------

TEST_F(WalTest, TruncateBeforeRemovesOldRecords) {
    {
        Wal wal(path_);
        for (uint32_t i = 0; i < 5; ++i)
            wal.append(WalRecordType::Insert, &i, sizeof(i));
        wal.sync();
        EXPECT_EQ(wal.current_lsn(), 5u);

        wal.truncate_before(3);

        // LSNs 3 and 4 remain; next append gets LSN 5.
        EXPECT_EQ(wal.current_lsn(), 5u);
    }

    // Reopen: base_lsn=3, next_lsn=5 (2 records kept).
    Wal wal(path_);
    EXPECT_EQ(wal.current_lsn(), 5u);

    // iterate from 0 sees only records at LSN 3 and 4.
    std::vector<Lsn> seen;
    wal.iterate(0, [&](Lsn lsn, WalRecordType, const void*, uint32_t) {
        seen.push_back(lsn);
    });
    ASSERT_EQ(seen.size(), 2u);
    EXPECT_EQ(seen[0], 3u);
    EXPECT_EQ(seen[1], 4u);
}

TEST_F(WalTest, TruncateBeforeNewAppendContinuesLsn) {
    {
        Wal wal(path_);
        for (uint32_t i = 0; i < 3; ++i)
            wal.append(WalRecordType::Insert, &i, sizeof(i));
        wal.sync();
        wal.truncate_before(2);
    }

    Wal wal(path_);
    uint32_t v = 99;
    Lsn lsn = wal.append(WalRecordType::Insert, &v, sizeof(v));
    EXPECT_EQ(lsn, 3u);  // continues from where it left off
}

// replay() with start_lsn skips earlier records.
TEST_F(WalTest, ReplayStartLsn) {
    const size_t dim = 2;
    std::vector<float> vec = {1.0f, 2.0f};

    {
        Wal wal(path_);
        for (uint32_t i = 0; i < 4; ++i) {
            auto p = make_insert_payload(i, std::to_string(i), vec.data(), dim);
            wal.append(WalRecordType::Insert, p.data(), p.size());
        }
        wal.sync();
    }

    Wal wal(path_);
    std::vector<uint32_t> ids;
    wal.replay(2, dim,
        [&](uint32_t id, const std::string&, const float*, size_t, const MetadataEntry&) { ids.push_back(id); },
        [&](uint32_t) {});

    ASSERT_EQ(ids.size(), 2u);
    EXPECT_EQ(ids[0], 2u);
    EXPECT_EQ(ids[1], 3u);
}

// ---------------------------------------------------------------------------
// Crash simulation: sidecar written (base_lsn=3) but rename not yet done.
// Old WAL still has all 5 records. Verifies that post-checkpoint records
// (old LSN 3 and 4) are not lost — they must appear in replay output.
// ---------------------------------------------------------------------------

TEST_F(WalTest, CrashAfterSidecarBeforeRename) {
    const size_t dim = 2;
    std::vector<float> vec = {1.0f, 2.0f};

    // Write 5 records with node IDs 0..4.
    {
        Wal wal(path_);
        for (uint32_t i = 0; i < 5; ++i) {
            auto p = make_insert_payload(i, std::to_string(i), vec.data(), dim);
            wal.append(WalRecordType::Insert, p.data(), p.size());
        }
        wal.sync();
    }

    // Simulate crash after sidecar write but before rename:
    // manually write base_lsn=3 to sidecar, leave WAL file unchanged.
    {
        uint64_t base = 3;
        std::ofstream f(path_ + ".base", std::ios::binary | std::ios::trunc);
        f.write(reinterpret_cast<const char*>(&base), sizeof(base));
    }

    // Open Wal — old WAL (5 records) + sidecar base_lsn=3.
    Wal wal(path_);

    // replay from checkpoint_lsn=3 must include node IDs 3 and 4
    // (the post-checkpoint records). IDs 0,1,2 may also appear (idempotent).
    std::vector<uint32_t> ids;
    wal.replay(3, dim,
        [&](uint32_t id, const std::string&, const float*, size_t, const MetadataEntry&) { ids.push_back(id); },
        [&](uint32_t) {});

    // IDs 3 and 4 must be present — they were not yet checkpointed.
    EXPECT_NE(std::find(ids.begin(), ids.end(), 3u), ids.end());
    EXPECT_NE(std::find(ids.begin(), ids.end(), 4u), ids.end());
}

// ---------------------------------------------------------------------------
// truncate_before(next_lsn): truncate all records — to_keep is empty.
// Verifies empty WAL + correct base_lsn on reopen, and that the next
// append continues from the right LSN.
// ---------------------------------------------------------------------------

TEST_F(WalTest, TruncateBeforeAllRecords) {
    {
        Wal wal(path_);
        for (uint32_t i = 0; i < 3; ++i)
            wal.append(WalRecordType::Insert, &i, sizeof(i));
        wal.sync();
        EXPECT_EQ(wal.current_lsn(), 3u);

        wal.truncate_before(3);  // truncate everything
        EXPECT_EQ(wal.current_lsn(), 3u);

        // iterate on empty WAL must return nothing.
        size_t count = 0;
        wal.iterate(0, [&](Lsn, WalRecordType, const void*, uint32_t) { ++count; });
        EXPECT_EQ(count, 0u);
    }

    // Reopen: empty file + base_lsn=3.
    Wal wal(path_);
    EXPECT_EQ(wal.current_lsn(), 3u);

    size_t count = 0;
    wal.iterate(0, [&](Lsn, WalRecordType, const void*, uint32_t) { ++count; });
    EXPECT_EQ(count, 0u);

    // Next append must get LSN 3, not 0.
    uint32_t v = 42;
    Lsn lsn = wal.append(WalRecordType::Insert, &v, sizeof(v));
    EXPECT_EQ(lsn, 3u);
    EXPECT_EQ(wal.current_lsn(), 4u);
}

}  // namespace
}  // namespace vectordb
