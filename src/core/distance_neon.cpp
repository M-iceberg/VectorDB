// ARM NEON distance implementations — added Day 3.
// Compiled only on aarch64 (see src/core/CMakeLists.txt).
#include "distance.h"

#if defined(VECTORDB_ARCH_ARM)
#include <arm_neon.h>
// TODO Day 3: NeonL2, NeonCosine, NeonIP using vmlaq_f32 / vaddvq_f32.
//             Override DistanceCompute::create() to return NEON impls.
#endif
