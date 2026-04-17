#pragma once
#include <string>
#include <vector>

namespace vectordb {
namespace tools {

// Load vectors from .fvecs file (float32). max_vecs=0 means load all.
std::vector<std::vector<float>> load_fvecs(const std::string& path, int max_vecs = 0);

// Load vectors from .bvecs file (uint8, cast to float). max_vecs=0 means load all.
std::vector<std::vector<float>> load_bvecs(const std::string& path, int max_vecs = 0);

// Load integer vectors from .ivecs file (used for ground truth neighbor ids).
std::vector<std::vector<int>> load_ivecs(const std::string& path, int max_vecs = 0);

}  // namespace tools
}  // namespace vectordb
