// SIFT-1M .fvecs / .bvecs loader — used by bench_hnsw and recall tests.
// Added Day 9 alongside HNSW search implementation.
//
// .fvecs format: [dim:int32][f0:float32][f1:float32]...[dim:int32][f0:float32]...
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace vectordb {
namespace tools {

std::vector<std::vector<float>> load_fvecs(const std::string& path) {
    // TODO Day 9
    (void)path;
    return {};
}

}  // namespace tools
}  // namespace vectordb
