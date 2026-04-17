// -----------------------------------------------------------------------------
// test_sift_recall.cpp — HNSW recall test on SIFT-1M subset
//
// SIFT (Scale-Invariant Feature Transform) is a standard ANN benchmark dataset.
// Each vector is a 128-dim float descriptor extracted from an image keypoint.
// SIFT-1M has 1 million such vectors; siftsmall is a 10K subset used here.
//
// Three file roles:
//   base vectors    — the dataset inserted into the HNSW index (10K vectors)
//   query vectors   — search queries ("find my nearest neighbors")
//   ground truth    — brute-force nearest neighbor ids per query, used to
//                     compute recall (how many of HNSW's results are correct)
//
// Test flow:
//   1. Insert all base vectors into HnswIndex
//   2. Search each query vector, get top-k results
//   3. Compare against ground truth, compute recall@10
//
// Skipped automatically if the data files are not present.
// Download: ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz
// Extract to data/sift/ relative to the project root.
//
// Run: ./build/tests/unit/test_sift_recall
// -----------------------------------------------------------------------------
#include <gtest/gtest.h>
#include "core/hnsw_index.h"
#include "load_sift.h"
#include <filesystem>
#include <unordered_set>

namespace vectordb {
namespace {

static const std::string kDataDir = "data/sift";

// Support both siftsmall (10K base, 100 queries) and full SIFT-1M filenames.
bool sift_data_available(std::string& base_path, std::string& query_path, std::string& gt_path) {
    // siftsmall (ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz)
    if (std::filesystem::exists(kDataDir + "/siftsmall_base.fvecs")) {
        base_path  = kDataDir + "/siftsmall_base.fvecs";
        query_path = kDataDir + "/siftsmall_query.fvecs";
        gt_path    = kDataDir + "/siftsmall_groundtruth.ivecs";
        return true;
    }
    // full SIFT-1M (http://corpus-texmex.irisa.fr/sift.tar.gz)
    if (std::filesystem::exists(kDataDir + "/sift_base.fvecs")) {
        base_path  = kDataDir + "/sift_base.fvecs";
        query_path = kDataDir + "/sift_query.fvecs";
        gt_path    = kDataDir + "/sift_groundtruth.ivecs";
        return true;
    }
    return false;
}

TEST(SiftRecall, Recall10KSubset) {
    std::string base_path, query_path, gt_path;
    if (!sift_data_available(base_path, query_path, gt_path)) {
        GTEST_SKIP() << "SIFT data not found at " << kDataDir
                     << ". Download siftsmall.tar.gz from "
                     << "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz "
                     << "and extract to data/sift/.";
    }

    // Load up to 10K base vectors, all queries, ground truth.
    auto base  = tools::load_fvecs(base_path,  10000);
    auto query = tools::load_fvecs(query_path, 0);
    auto gt    = tools::load_ivecs(gt_path,    0);

    ASSERT_FALSE(base.empty())  << "failed to load base vectors";
    ASSERT_FALSE(query.empty()) << "failed to load query vectors";
    ASSERT_FALSE(gt.empty())    << "failed to load ground truth";

    const size_t dim = base[0].size();

    // Build index on 10K base vectors.
    HnswConfig cfg;
    cfg.dim             = dim;
    cfg.M               = 16;
    cfg.M0              = 32;
    cfg.ef_construction = 200;
    cfg.metric          = Metric::L2;

    HnswIndex idx(cfg);
    for (size_t i = 0; i < base.size(); ++i)
        idx.insert(static_cast<NodeId>(i), base[i].data());

    // Measure recall@10 over all queries.
    const int k = 10;
    int found = 0, total = 0;

    for (size_t q = 0; q < query.size(); ++q) {
        // Ground truth: first k neighbor ids for this query.
        std::unordered_set<int> gt_set;
        for (int i = 0; i < k && i < static_cast<int>(gt[q].size()); ++i)
            gt_set.insert(gt[q][i]);

        auto result = idx.search(query[q].data(), k, 64);
        for (auto& [d, id] : result)
            if (gt_set.count(static_cast<int>(id))) ++found;
        total += k;
    }

    double recall = static_cast<double>(found) / total;
    std::cout << "\n[SiftRecall] 10K base, " << query.size()
              << " queries, ef_search=64\n"
              << "  recall@10 = " << recall * 100 << "%\n";

    EXPECT_GE(recall, 0.90) << "recall@10 below 90% on SIFT-1M 10K subset";
}

}  // namespace
}  // namespace vectordb
