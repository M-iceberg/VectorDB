#include <gtest/gtest.h>
#include "storage/metadata_index.h"
#include <algorithm>
#include <limits>
#include <unordered_set>

namespace vectordb {
namespace {

// Sort a vector of NodeId for order-independent comparison.
static std::vector<NodeId> sorted(std::vector<NodeId> v) {
    std::sort(v.begin(), v.end());
    return v;
}

// Set helpers for AND / OR / NOT filter composition (caller-side operations).
static std::vector<NodeId> set_and(std::vector<NodeId> a, std::vector<NodeId> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::vector<NodeId> result;
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::back_inserter(result));
    return result;
}

static std::vector<NodeId> set_or(std::vector<NodeId> a, std::vector<NodeId> b) {
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::vector<NodeId> result;
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::back_inserter(result));
    return result;
}

// NOT: returns all_ids minus excluded.
static std::vector<NodeId> set_not(std::vector<NodeId> all_ids,
                                   std::vector<NodeId> excluded) {
    std::sort(all_ids.begin(), all_ids.end());
    std::sort(excluded.begin(), excluded.end());
    std::vector<NodeId> result;
    std::set_difference(all_ids.begin(), all_ids.end(),
                        excluded.begin(), excluded.end(),
                        std::back_inserter(result));
    return result;
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

// ---------------------------------------------------------------------------
// gte: lower bound only (price >= 50) — query_range with +infinity as hi
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, NumericGte) {
    MetadataIndex idx;
    idx.insert_numeric("price", 10.0, 0);
    idx.insert_numeric("price", 50.0, 1);
    idx.insert_numeric("price", 99.0, 2);
    idx.insert_numeric("price", 200.0, 3);

    double inf = std::numeric_limits<double>::infinity();
    EXPECT_EQ(sorted(idx.query_range("price", 50.0, inf)),
              (std::vector<NodeId>{1, 2, 3}));
    EXPECT_EQ(sorted(idx.query_range("price", 100.0, inf)),
              (std::vector<NodeId>{3}));
}

// ---------------------------------------------------------------------------
// lte: upper bound only (price <= 50) — query_range with -infinity as lo
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, NumericLte) {
    MetadataIndex idx;
    idx.insert_numeric("price", 10.0, 0);
    idx.insert_numeric("price", 50.0, 1);
    idx.insert_numeric("price", 99.0, 2);
    idx.insert_numeric("price", 200.0, 3);

    double ninf = -std::numeric_limits<double>::infinity();
    EXPECT_EQ(sorted(idx.query_range("price", ninf, 50.0)),
              (std::vector<NodeId>{0, 1}));
    EXPECT_EQ(sorted(idx.query_range("price", ninf, 9.0)),
              (std::vector<NodeId>{}));
}

// ---------------------------------------------------------------------------
// AND: category == "electronics" AND price <= 100
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, FilterAnd) {
    MetadataIndex idx;
    //       id  category        price
    idx.insert_string ("category", "electronics", 0); idx.insert_numeric("price", 80.0,  0);
    idx.insert_string ("category", "electronics", 1); idx.insert_numeric("price", 150.0, 1);
    idx.insert_string ("category", "furniture",   2); idx.insert_numeric("price", 60.0,  2);
    idx.insert_string ("category", "electronics", 3); idx.insert_numeric("price", 50.0,  3);

    double inf = std::numeric_limits<double>::infinity();
    auto electronics = idx.query_eq("category", "electronics");       // [0,1,3]
    auto affordable  = idx.query_range("price", -inf, 100.0);         // [0,2,3]

    // AND: nodes that are electronics AND affordable
    EXPECT_EQ(set_and(electronics, affordable), (std::vector<NodeId>{0, 3}));
}

// ---------------------------------------------------------------------------
// OR: category == "electronics" OR category == "furniture"
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, FilterOr) {
    MetadataIndex idx;
    idx.insert_string("category", "electronics", 0);
    idx.insert_string("category", "furniture",   1);
    idx.insert_string("category", "clothing",    2);
    idx.insert_string("category", "electronics", 3);

    auto a = idx.query_eq("category", "electronics");  // [0,3]
    auto b = idx.query_eq("category", "furniture");    // [1]

    EXPECT_EQ(set_or(a, b), (std::vector<NodeId>{0, 1, 3}));
}

// ---------------------------------------------------------------------------
// NOT: all nodes except those with status == "deleted"
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, FilterNot) {
    MetadataIndex idx;
    idx.insert_string("status", "active",  0);
    idx.insert_string("status", "deleted", 1);
    idx.insert_string("status", "active",  2);
    idx.insert_string("status", "deleted", 3);
    idx.insert_string("status", "active",  4);

    std::vector<NodeId> all_ids = {0, 1, 2, 3, 4};
    auto deleted = idx.query_eq("status", "deleted");  // [1,3]

    // NOT deleted = all_ids minus deleted
    EXPECT_EQ(set_not(all_ids, deleted), (std::vector<NodeId>{0, 2, 4}));
}

// ---------------------------------------------------------------------------
// Combined: (category == "electronics" OR category == "books")
//           AND price >= 20
//           AND NOT status == "deleted"
// ---------------------------------------------------------------------------

TEST(MetadataIndexTest, FilterCombined) {
    MetadataIndex idx;
    // id  category       price  status
    idx.insert_string("category", "electronics", 0); idx.insert_numeric("price", 80.0,  0); idx.insert_string("status", "active",  0);
    idx.insert_string("category", "books",       1); idx.insert_numeric("price", 15.0,  1); idx.insert_string("status", "active",  1);
    idx.insert_string("category", "electronics", 2); idx.insert_numeric("price", 30.0,  2); idx.insert_string("status", "deleted", 2);
    idx.insert_string("category", "clothing",    3); idx.insert_numeric("price", 50.0,  3); idx.insert_string("status", "active",  3);
    idx.insert_string("category", "books",       4); idx.insert_numeric("price", 25.0,  4); idx.insert_string("status", "active",  4);

    double inf = std::numeric_limits<double>::infinity();

    // (category == "electronics" OR category == "books")
    auto cat = set_or(idx.query_eq("category", "electronics"),
                      idx.query_eq("category", "books"));        // [0,1,2,4]

    // AND price >= 20
    auto priced = idx.query_range("price", 20.0, inf);           // [0,2,3,4]
    auto step2 = set_and(cat, priced);                           // [0,2,4]

    // AND NOT status == "deleted"
    auto deleted = idx.query_eq("status", "deleted");            // [2]
    auto result = set_not(step2, deleted);                       // [0,4]

    EXPECT_EQ(result, (std::vector<NodeId>{0, 4}));
}

}  // namespace
}  // namespace vectordb
