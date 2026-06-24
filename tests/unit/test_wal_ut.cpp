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
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>

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

}  // namespace
}  // namespace vectordb
