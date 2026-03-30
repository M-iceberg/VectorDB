// -----------------------------------------------------------------------------
// distance_neon.cpp — ARM NEON SIMD distance implementations  (Day 3)
//
// Compiled only on aarch64 (see src/core/CMakeLists.txt).
// Provides NeonL2, NeonCosine, NeonIP using 128-bit NEON intrinsics:
//   vmlaq_f32  — fused multiply-accumulate on 4×float lanes
//   vaddvq_f32 — horizontal sum across a float32x4_t register
// Handles scalar tail when dim is not a multiple of 4.
//
// After Day 3, this file also overrides DistanceCompute::create() to
// return NEON implementations on ARM, replacing the scalar factory in
// distance_naive.cpp.
// -----------------------------------------------------------------------------
// ARM NEON distance implementations — added Day 3.
// Compiled only on aarch64 (see src/core/CMakeLists.txt).
#include "distance.h"

#if defined(VECTORDB_ARCH_ARM)
#include <arm_neon.h>
// TODO Day 3: NeonL2, NeonCosine, NeonIP using vmlaq_f32 / vaddvq_f32.
//             Override DistanceCompute::create() to return NEON impls.
#endif
