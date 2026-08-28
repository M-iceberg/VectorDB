#include <gtest/gtest.h>
#include "storage/vector_file.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <vector>

namespace vectordb {
namespace {

class VectorFileTest : public ::testing::Test {
protected:
    std::string path_;
    void SetUp() override {
        path_ = std::filesystem::temp_directory_path() / "test_vf_XXXXXX";
        path_ += std::to_string(
            std::chrono::steady_clock::now().time_since_epoch().count());
    }
    void TearDown() override {
        std::filesystem::remove(path_);
    }
};

// ---------------------------------------------------------------------------
// Basic: append + read back in the same session
// ---------------------------------------------------------------------------

TEST_F(VectorFileTest, AppendAndReadBack) {
    const size_t dim = 4;
    VectorFile vf(path_, dim);

    std::vector<float> v0 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> v1 = {5.0f, 6.0f, 7.0f, 8.0f};

    uint32_t s0 = vf.append(v0.data());
    uint32_t s1 = vf.append(v1.data());

    EXPECT_EQ(s0, 0u);
    EXPECT_EQ(s1, 1u);
    EXPECT_EQ(vf.slot_count(), 2u);

    for (size_t i = 0; i < dim; ++i) EXPECT_FLOAT_EQ(vf.read(s0)[i], v0[i]);
    for (size_t i = 0; i < dim; ++i) EXPECT_FLOAT_EQ(vf.read(s1)[i], v1[i]);
}

// ---------------------------------------------------------------------------
// Persistence: close and reopen, read back by slot
// ---------------------------------------------------------------------------

TEST_F(VectorFileTest, PersistsAcrossReopen) {
    const size_t dim = 8;
    const size_t N   = 1000;

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < dim; ++d)
            vecs[i][d] = static_cast<float>(i * dim + d);

    {
        VectorFile vf(path_, dim);
        for (size_t i = 0; i < N; ++i)
            vf.append(vecs[i].data());
        EXPECT_EQ(vf.slot_count(), N);
    }

    VectorFile vf(path_, dim);
    EXPECT_EQ(vf.slot_count(), N);
    for (size_t i = 0; i < N; ++i) {
        const float* p = vf.read(static_cast<uint32_t>(i));
        for (size_t d = 0; d < dim; ++d)
            EXPECT_FLOAT_EQ(p[d], vecs[i][d]);
    }
}

// ---------------------------------------------------------------------------
// Growth: write enough vectors to force multiple remaps
// ---------------------------------------------------------------------------

TEST_F(VectorFileTest, GrowthTriggeredByAppend) {
    const size_t dim = 16;
    // Initial capacity is 1024; write 3000 to force at least two doublings.
    const size_t N = 3000;

    std::vector<VectorFile*> dummy;  // not used; just confirm no crash
    VectorFile vf(path_, dim);
    for (uint32_t i = 0; i < N; ++i) {
        std::vector<float> v(dim, static_cast<float>(i));
        uint32_t slot = vf.append(v.data());
        EXPECT_EQ(slot, i);
    }
    EXPECT_EQ(vf.slot_count(), N);

    // Spot-check a few slots after growth.
    for (uint32_t i : {0u, 1023u, 1024u, 2999u}) {
        const float* p = vf.read(i);
        for (size_t d = 0; d < dim; ++d)
            EXPECT_FLOAT_EQ(p[d], static_cast<float>(i));
    }
}

// ---------------------------------------------------------------------------
// Reopen with wrong dim should throw
// ---------------------------------------------------------------------------

TEST_F(VectorFileTest, DimMismatchThrowsOnReopen) {
    { VectorFile vf(path_, 4); }
    EXPECT_THROW(VectorFile(path_, 8), std::runtime_error);
}

// ---------------------------------------------------------------------------
// slot_count persists correctly after growth + reopen
// ---------------------------------------------------------------------------

TEST_F(VectorFileTest, SlotCountPersistsAfterGrowth) {
    const size_t dim = 4;
    const size_t N   = 2000;

    {
        VectorFile vf(path_, dim);
        std::vector<float> v(dim, 1.0f);
        for (size_t i = 0; i < N; ++i)
            vf.append(v.data());
    }

    VectorFile vf(path_, dim);
    EXPECT_EQ(vf.slot_count(), N);
}

// ---------------------------------------------------------------------------
// Data correctness after grow + reopen: verifies all vector values survive
// a remap and a close/reopen cycle. Would have caught the dangling-pointer
// bug in append() where hdr was fetched before grow() and used after.
// ---------------------------------------------------------------------------

TEST_F(VectorFileTest, DataCorrectAfterGrowAndReopen) {
    const size_t dim = 8;
    const size_t N   = 2000;  // triggers one grow (1024 → 2048)

    std::vector<std::vector<float>> vecs(N, std::vector<float>(dim));
    for (size_t i = 0; i < N; ++i)
        for (size_t d = 0; d < dim; ++d)
            vecs[i][d] = static_cast<float>(i * 100 + d);

    {
        VectorFile vf(path_, dim);
        for (size_t i = 0; i < N; ++i)
            vf.append(vecs[i].data());
    }

    VectorFile vf(path_, dim);
    ASSERT_EQ(vf.slot_count(), N);
    for (size_t i = 0; i < N; ++i) {
        const float* p = vf.read(static_cast<uint32_t>(i));
        for (size_t d = 0; d < dim; ++d)
            EXPECT_FLOAT_EQ(p[d], vecs[i][d]) << "slot=" << i << " dim=" << d;
    }
}

// ---------------------------------------------------------------------------
// Append after reopen: slot numbering continues from where it left off,
// and newly appended vectors are readable and persistent.
// ---------------------------------------------------------------------------

TEST_F(VectorFileTest, AppendAfterReopen) {
    const size_t dim = 4;
    std::vector<float> v0 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> v1 = {5.0f, 6.0f, 7.0f, 8.0f};
    std::vector<float> v2 = {9.0f, 10.0f, 11.0f, 12.0f};

    {
        VectorFile vf(path_, dim);
        EXPECT_EQ(vf.append(v0.data()), 0u);
        EXPECT_EQ(vf.append(v1.data()), 1u);
    }

    // Reopen and append a third vector — must get slot 2, not 0.
    {
        VectorFile vf(path_, dim);
        ASSERT_EQ(vf.slot_count(), 2u);
        EXPECT_EQ(vf.append(v2.data()), 2u);
        EXPECT_EQ(vf.slot_count(), 3u);
    }

    // Reopen again and verify all three vectors.
    VectorFile vf(path_, dim);
    ASSERT_EQ(vf.slot_count(), 3u);
    for (size_t d = 0; d < dim; ++d) EXPECT_FLOAT_EQ(vf.read(0)[d], v0[d]);
    for (size_t d = 0; d < dim; ++d) EXPECT_FLOAT_EQ(vf.read(1)[d], v1[d]);
    for (size_t d = 0; d < dim; ++d) EXPECT_FLOAT_EQ(vf.read(2)[d], v2[d]);
}

TEST_F(VectorFileTest, StableSlotWriteSupportsOverwriteAndGrowth) {
    const size_t dim = 4;
    std::vector<float> initial = {1, 2, 3, 4};
    std::vector<float> updated = {5, 6, 7, 8};
    std::vector<float> distant = {9, 10, 11, 12};

    {
        VectorFile vf(path_, dim);
        vf.write(0, initial.data());
        vf.write(0, updated.data());
        vf.write(2048, distant.data());  // forces multiple mmap growth steps
        vf.sync();
        EXPECT_EQ(vf.slot_count(), 2049u);
        EXPECT_EQ(vf.data(), vf.read(0));
    }

    VectorFile vf(path_, dim);
    ASSERT_EQ(vf.slot_count(), 2049u);
    for (size_t d = 0; d < dim; ++d) {
        EXPECT_FLOAT_EQ(vf.read(0)[d], updated[d]);
        EXPECT_FLOAT_EQ(vf.read(2048)[d], distant[d]);
    }
}

TEST_F(VectorFileTest, RejectsNullStableSlotWrite) {
    VectorFile vf(path_, 4);
    EXPECT_THROW(vf.write(0, nullptr), std::invalid_argument);
}

}  // namespace
}  // namespace vectordb
