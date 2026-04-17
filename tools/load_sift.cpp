// SIFT-1M .fvecs / .bvecs / .ivecs loader.
//
// File formats (all little-endian):
//   .fvecs: [dim:int32][v0:float32]...[vd-1:float32] repeated per vector
//   .bvecs: [dim:int32][v0:uint8] ...[vd-1:uint8]    repeated per vector
//   .ivecs: [dim:int32][v0:int32] ...[vd-1:int32]    repeated per vector
//
// Usage:
//   auto base   = load_fvecs("sift/sift_base.fvecs");       // 1M x 128
//   auto query  = load_fvecs("sift/sift_query.fvecs");      // 10K x 128
//   auto gt     = load_ivecs("sift/sift_groundtruth.ivecs"); // 10K x 100
#include "load_sift.h"
#include <cstdint>
#include <fstream>
#include <stdexcept>

namespace vectordb {
namespace tools {

std::vector<std::vector<float>> load_fvecs(const std::string& path, int max_vecs) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);

    std::vector<std::vector<float>> result;
    int32_t dim;
    while (f.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        std::vector<float> vec(dim);
        if (!f.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(float)))
            break;
        result.push_back(std::move(vec));
        if (max_vecs > 0 && static_cast<int>(result.size()) >= max_vecs) break;
    }
    return result;
}

std::vector<std::vector<float>> load_bvecs(const std::string& path, int max_vecs) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);

    std::vector<std::vector<float>> result;
    int32_t dim;
    while (f.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        std::vector<uint8_t> raw(dim);
        if (!f.read(reinterpret_cast<char*>(raw.data()), dim))
            break;
        std::vector<float> vec(dim);
        for (int i = 0; i < dim; ++i) vec[i] = static_cast<float>(raw[i]);
        result.push_back(std::move(vec));
        if (max_vecs > 0 && static_cast<int>(result.size()) >= max_vecs) break;
    }
    return result;
}

std::vector<std::vector<int>> load_ivecs(const std::string& path, int max_vecs) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open: " + path);

    std::vector<std::vector<int>> result;
    int32_t dim;
    while (f.read(reinterpret_cast<char*>(&dim), sizeof(dim))) {
        std::vector<int32_t> vec(dim);
        if (!f.read(reinterpret_cast<char*>(vec.data()), dim * sizeof(int32_t)))
            break;
        result.push_back(std::vector<int>(vec.begin(), vec.end()));
        if (max_vecs > 0 && static_cast<int>(result.size()) >= max_vecs) break;
    }
    return result;
}

}  // namespace tools
}  // namespace vectordb
