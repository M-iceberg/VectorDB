// -----------------------------------------------------------------------------
// distance_avx2.cpp — x86 AVX2 SIMD distance implementations  (Day 4)
//
// Compiled only on x86_64 (see src/core/CMakeLists.txt).
// Provides Avx2L2, Avx2Cosine, Avx2IP using 256-bit AVX2 intrinsics:
//   _mm256_fmadd_ps   — fused multiply-add on 8×float lanes
//   _mm256_hadd_ps    — horizontal add for lane reduction
// Handles scalar tail when dim is not a multiple of 8.
//
// Runtime selection between AVX2 and naive is done in distance_dispatch.cpp.
// -----------------------------------------------------------------------------
// x86 AVX2 distance implementations — added Day 4.
// Compiled only on x86_64 (see src/core/CMakeLists.txt).
#include "distance.h"

#if defined(VECTORDB_ARCH_X86)
#include <immintrin.h>
// TODO Day 4: Avx2L2, Avx2Cosine, Avx2IP using _mm256_fmadd_ps.
#endif
