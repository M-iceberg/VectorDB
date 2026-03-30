// x86 AVX2 distance implementations — added Day 4.
// Compiled only on x86_64 (see src/core/CMakeLists.txt).
#include "distance.h"

#if defined(VECTORDB_ARCH_X86)
#include <immintrin.h>
// TODO Day 4: Avx2L2, Avx2Cosine, Avx2IP using _mm256_fmadd_ps.
#endif
