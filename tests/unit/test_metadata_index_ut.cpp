#include <gtest/gtest.h>
#include "storage/metadata_index.h"
#include <algorithm>

namespace vectordb {
namespace {

// Sort a vector of NodeId for order-independent comparison.
static std::vector<NodeId> sorted(std::vector<NodeId> v) {
    std::sort(v.begin(), v.end());
    return v;
}

// ---------------------------------------------------------------------------
// String index — query_eq
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, StringEqBasic) {
    MetadataIndex idx;
    idx.insert_string("color", "red",   0);
    idx.insert_string("color", "blue",  1);
    idx.insert_string("color", "red",   2);

    EXPECT_EQ(sorted(idx.query_eq("color", "red")),  (std::vector<NodeId>{0, 2}));
    EXPECT_EQ(sorted(idx.query_eq("color", "blue")), (std::vector<NodeId>{1}));
    EXPECT_TRUE(idx.query_eq("color", "green").empty());
}

TEST(MetadataIndexTest, StringEqMissingField) {
    MetadataIndex idx;
    EXPECT_TRUE(idx.query_eq("nonexistent", "any").empty());
}

TEST(MetadataIndexTest, StringEqMissingValue) {
    MetadataIndex idx;
    idx.insert_string("tag", "a", 0);
    EXPECT_TRUE(idx.query_eq("tag", "z").empty());
}

// Simulating "in" query: union of multiple query_eq calls.
TEST(MetadataIndexTest, StringInViaMultipleQueryEq) {
    MetadataIndex idx;
    idx.insert_string("status", "active",   0);
    idx.insert_string("status", "pending",  1);
    idx.insert_string("status", "active",   2);
    idx.insert_string("status", "archived", 3);

    // "status IN ('active', 'pending')"
    auto r1 = idx.query_eq("status", "active");
    auto r2 = idx.query_eq("status", "pending");
    r1.insert(r1.end(), r2.begin(), r2.end());
    EXPECT_EQ(sorted(r1), (std::vector<NodeId>{0, 1, 2}));
}

// ---------------------------------------------------------------------------
// Numeric index — query_range
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, NumericRangeBasic) {
    MetadataIndex idx;
    idx.insert_numeric("price", 10.0, 0);
    idx.insert_numeric("price", 20.0, 1);
    idx.insert_numeric("price", 30.0, 2);
    idx.insert_numeric("price", 40.0, 3);

    EXPECT_EQ(sorted(idx.query_range("price", 15.0, 35.0)),
              (std::vector<NodeId>{1, 2}));
}

TEST(MetadataIndexTest, NumericRangeInclusiveBounds) {
    MetadataIndex idx;
    idx.insert_numeric("score", 1.0, 0);
    idx.insert_numeric("score", 5.0, 1);
    idx.insert_numeric("score", 10.0, 2);

    // Both bounds are inclusive.
    EXPECT_EQ(sorted(idx.query_range("score", 1.0, 10.0)),
              (std::vector<NodeId>{0, 1, 2}));
    EXPECT_EQ(sorted(idx.query_range("score", 5.0, 5.0)),
              (std::vector<NodeId>{1}));
}

TEST(MetadataIndexTest, NumericRangeEmpty) {
    MetadataIndex idx;
    idx.insert_numeric("val", 1.0, 0);
    idx.insert_numeric("val", 2.0, 1);

    EXPECT_TRUE(idx.query_range("val", 5.0, 10.0).empty());
}

TEST(MetadataIndexTest, NumericRangeMissingField) {
    MetadataIndex idx;
    EXPECT_TRUE(idx.query_range("nonexistent", 0.0, 100.0).empty());
}

TEST(MetadataIndexTest, NumericRangeInvertedBoundsReturnsEmpty) {
    MetadataIndex idx;
    idx.insert_numeric("val", 5.0, 0);
    // lo > hi must return empty, not crash.
    EXPECT_TRUE(idx.query_range("val", 100.0, 10.0).empty());
}

TEST(MetadataIndexTest, NumericMultipleNodesWithSameValue) {
    MetadataIndex idx;
    idx.insert_numeric("rank", 5.0, 10);
    idx.insert_numeric("rank", 5.0, 20);
    idx.insert_numeric("rank", 5.0, 30);

    EXPECT_EQ(sorted(idx.query_range("rank", 5.0, 5.0)),
              (std::vector<NodeId>{10, 20, 30}));
}

// ---------------------------------------------------------------------------
// Boolean fields (stored as string "0"/"1")
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, BooleanAsString) {
    MetadataIndex idx;
    idx.insert_string("is_active", "1", 0);
    idx.insert_string("is_active", "0", 1);
    idx.insert_string("is_active", "1", 2);

    EXPECT_EQ(sorted(idx.query_eq("is_active", "1")), (std::vector<NodeId>{0, 2}));
    EXPECT_EQ(sorted(idx.query_eq("is_active", "0")), (std::vector<NodeId>{1}));
}

// ---------------------------------------------------------------------------
// remove
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, RemoveFromStringIndex) {
    MetadataIndex idx;
    idx.insert_string("color", "red", 0);
    idx.insert_string("color", "red", 1);
    idx.insert_string("color", "blue", 1);

    idx.remove(1);

    EXPECT_EQ(sorted(idx.query_eq("color", "red")),  (std::vector<NodeId>{0}));
    EXPECT_TRUE(idx.query_eq("color", "blue").empty());
}

TEST(MetadataIndexTest, RemoveFromNumericIndex) {
    MetadataIndex idx;
    idx.insert_numeric("age", 25.0, 0);
    idx.insert_numeric("age", 30.0, 1);
    idx.insert_numeric("age", 25.0, 2);

    idx.remove(0);

    EXPECT_EQ(sorted(idx.query_range("age", 20.0, 30.0)),
              (std::vector<NodeId>{1, 2}));
}

TEST(MetadataIndexTest, RemoveNonExistentIdNocrash) {
    MetadataIndex idx;
    idx.insert_string("x", "v", 0);
    // Removing an id that was never inserted must not crash.
    idx.remove(999);
    EXPECT_EQ(idx.query_eq("x", "v"), (std::vector<NodeId>{0}));
}

TEST(MetadataIndexTest, RemoveTwiceSameIdNocrash) {
    MetadataIndex idx;
    idx.insert_string("color", "red", 0);
    idx.insert_numeric("price", 50.0, 0);
    idx.remove(0);
    // Second remove must not crash (e.g. WAL replay replays same delete twice).
    idx.remove(0);
    EXPECT_TRUE(idx.query_eq("color", "red").empty());
    EXPECT_TRUE(idx.query_range("price", 0.0, 100.0).empty());
}

TEST(MetadataIndexTest, RemoveAllNodesFromValue) {
    MetadataIndex idx;
    idx.insert_string("tag", "hot", 0);
    idx.insert_string("tag", "hot", 1);

    idx.remove(0);
    idx.remove(1);

    EXPECT_TRUE(idx.query_eq("tag", "hot").empty());
}

// ---------------------------------------------------------------------------
// Multiple fields on one node
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, MultipleFieldsPerNode) {
    MetadataIndex idx;
    idx.insert_string ("category", "electronics", 0);
    idx.insert_numeric("price",     299.99,        0);
    idx.insert_string ("in_stock",  "1",           0);

    EXPECT_EQ(idx.query_eq("category", "electronics"), (std::vector<NodeId>{0}));
    EXPECT_EQ(sorted(idx.query_range("price", 200.0, 400.0)), (std::vector<NodeId>{0}));

    // remove(0) must clean up all three field entries.
    idx.remove(0);
    EXPECT_TRUE(idx.query_eq("category", "electronics").empty());
    EXPECT_TRUE(idx.query_range("price", 0.0, 1000.0).empty());
}

// ---------------------------------------------------------------------------
// Numeric: insert order doesn't matter — sorted list handles any order
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, NumericInsertOutOfOrder) {
    MetadataIndex idx;
    idx.insert_numeric("ts", 300.0, 2);
    idx.insert_numeric("ts", 100.0, 0);
    idx.insert_numeric("ts", 200.0, 1);

    EXPECT_EQ(sorted(idx.query_range("ts", 100.0, 200.0)),
              (std::vector<NodeId>{0, 1}));
    EXPECT_EQ(sorted(idx.query_range("ts", 150.0, 350.0)),
              (std::vector<NodeId>{1, 2}));
}

// ---------------------------------------------------------------------------
// remove then re-insert same id
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, RemoveAndReinsert) {
    MetadataIndex idx;
    idx.insert_string("tier", "gold", 5);
    idx.remove(5);
    EXPECT_TRUE(idx.query_eq("tier", "gold").empty());

    idx.insert_string("tier", "silver", 5);
    EXPECT_EQ(idx.query_eq("tier", "silver"), (std::vector<NodeId>{5}));
    EXPECT_TRUE(idx.query_eq("tier", "gold").empty());
}

}  // namespace
}  // namespace vectordb
