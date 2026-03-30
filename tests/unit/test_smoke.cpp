#include <gtest/gtest.h>
#include "core/version.h"
#include "core/distance.h"
#include "core/collection.h"

// Day 1 smoke tests: verify the build system and skeleton interfaces compile.
// Real unit tests start Day 2 (distance scalar baseline).

TEST(Smoke, BuildSystemWorks) {
    EXPECT_EQ(1 + 1, 2);
}

TEST(Smoke, FloatArithmetic) {
    EXPECT_FLOAT_EQ(1.0f + 2.0f, 3.0f);
}

TEST(Smoke, VersionDefined) {
    std::string v = VECTORDB_VERSION;
    EXPECT_FALSE(v.empty());
}

TEST(Smoke, MetricEnumValues) {
    EXPECT_EQ(static_cast<int>(vectordb::Metric::L2),           0);
    EXPECT_EQ(static_cast<int>(vectordb::Metric::Cosine),       1);
    EXPECT_EQ(static_cast<int>(vectordb::Metric::InnerProduct),  2);
}

TEST(Smoke, CollectionSchemaCreation) {
    vectordb::CollectionSchema s;
    s.name   = "test";
    s.dim    = 128;
    s.metric = vectordb::Metric::L2;
    vectordb::Collection col(s);
    EXPECT_EQ(col.schema().name, "test");
    EXPECT_EQ(col.schema().dim,  128u);
}

#if defined(VECTORDB_ARCH_ARM)
TEST(Smoke, PlatformIsARM) { SUCCEED(); }
#elif defined(VECTORDB_ARCH_X86)
TEST(Smoke, PlatformIsX86) { SUCCEED(); }
#else
TEST(Smoke, PlatformIsGeneric) { SUCCEED(); }
#endif
